import pandas as pd
import os
import nibabel as nib
import numpy as np
import json
from tqdm import tqdm
import argparse
import datetime


def save_fullview_like(arr, t1_img, out_path, dtype=None):
    if dtype is not None:
        arr = arr.astype(dtype)
    print('Saving to ', out_path)
    nib.save(nib.Nifti1Image(arr, t1_img.affine, t1_img.header), str(out_path))
    print(f'MIN: {arr.min():.4f}, MAX: {arr.max():.4f}, STD:{arr.std():.4f}')


def save_part_nii(t1_path, seg_path, mask_path, labels, prefix, postfix, save_dir, whole_brain_norm = False):
    # Load seg (label map)
    # Excluding cerebellum labels
    seg_img = nib.load(str(seg_path))
    seg = seg_img.get_fdata().astype(np.int32)
    bg_mask = (seg == 0).astype(np.uint8)
    try:
        head_mask = nib.load(str(mask_path)).get_fdata().astype(np.uint8)
    except Exception as e:
        head_mask = (seg != 0).astype(np.uint8)
    
    t1_img = nib.load(str(t1_path))
    t1 = t1_img.get_fdata().astype(np.float32)

    if whole_brain_norm == True:
        t1 = normalize_mri(t1, head_mask)

    if labels == 'whole':
        part_mask = head_mask.copy()
        part_arr = np.where(part_mask, t1, -1.0)
    elif labels == 'left':
        mid = seg.shape[0] // 2
        part_mask = head_mask.copy()
        part_mask[:mid, :, :] = 0
        part_mask = part_mask.astype(np.uint8)
        part_arr = np.where(part_mask, t1, -1.0)
    elif labels == 'right':
        mid = seg.shape[0] // 2
        part_mask = head_mask.copy()
        part_mask[mid:, :, :] = 0
        part_mask = part_mask.astype(np.uint8)
        part_arr = np.where(part_mask, t1, -1.0)
    elif prefix == 'rhemi_mirror':
        part_mask = np.isin(seg, labels).astype(np.uint8)
        part_arr = np.where(part_mask, t1, -1.0)
        part_mask = part_mask[::-1, :, :]
        part_arr = np.where(part_mask, t1[::-1, :, :], -1.0)
    elif labels == 'right_mirror':
        mid = seg.shape[0] // 2
        part_mask = head_mask.copy()
        part_mask[mid:, :, :] = 0
        part_mask = part_mask.astype(np.uint8)
    else:
        part_mask = np.isin(seg, labels).astype(np.uint8)
        part_arr = np.where(part_mask, t1, -1.0)

    # CROP 
    if prefix == 'rhemi':
        part_arr = part_arr[:96, :, :]
        part_mask = part_mask[:96, :, :]
    if prefix == 'lhemi' or prefix == 'rhemi_mirror':
        part_arr = part_arr[-96:, :, :]
        part_mask = part_mask[-96:, :, :]
    if prefix == 'sub':
        part_arr = part_arr[:, :128, :96]
        part_mask = part_mask[:, :128, :96]
    
    part_vol = part_mask.sum()/head_mask.sum()

    file_name = prefix+t1_path.split('/')[-2]+postfix
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir+'_mask', exist_ok=True)

    save_fullview_like(part_arr, t1_img, f"{save_dir}/{file_name}")
    save_fullview_like(part_mask, t1_img, f"{save_dir+'_mask'}/{'mask_'+file_name}")

    return f"{save_dir}/{file_name}", f"{save_dir+'_mask'}/{'mask_'+file_name}", part_vol



def normalize_mri(vol, mask, pct=(1, 99)):
    """
    vol: np.ndarray (any dtype)
    mask: boolean array same shape as vol (optional)
    robust: if True, use percentiles within mask instead of global min/max
    pct: percentile bounds (low, high) for robust scaling
    """
    x = vol.astype(np.float32)
    mask = mask.astype(bool)

    #x = np.where(mask, x, -1.0).astype(np.float32)
    vals = x[mask] if mask.any() else x.ravel()
    lo, hi = np.percentile(vals, [pct[0], pct[1]])
    # avoid degeneracy
    if hi <= lo: lo, hi = x.min(), x.max()

    v = np.clip(x, lo, hi)
    v = (v - lo) / (hi - lo + 1e-8)      # -> [0,1]
    v = v * 2.0 - 1.0                     # -> [-1,1]

    return v



def get_imageID(data_path):
    return str(data_path.split('_')[-2])



def main():
    PARTS = {'left_hemi':{'prefix':'lhemi','labels':[2,3,4,5,10,11,12,13,17,18,26,28]},
            'right_hemi':{'prefix':'rhemi', 'labels':[41,42,43,44,49,50,51,52,53,54,58,60]},
            'right_hemi_mirror':{'prefix':'rhemi_mirror', 'labels':[41,42,43,44,49,50,51,52,53,54,58,60]},
            'cerebellum':{'prefix':'cerebellum', 'labels':[7,8,46,47]},
            'stem':{'prefix':'stem', 'labels':[14,15,16, 24]},
            'sub':{'prefix':'sub', 'labels':[7,8,46,47, 14,15,16]},
            'ventricles':{'prefix':'ventricles', 'labels':[14,15]},
            'cerebral':{'prefix':'cerebral', 'labels':[2,41,3,42]},
            'hemi':{'prefix':'hemi', 'labels':'hemi'},
            'whole_brain':{'prefix':'whole_brain', 'labels':'whole'},
            'left':{'prefix':'left', 'labels':'left'},
            'right':{'prefix':'right', 'labels':'right'},
            'right_mirror':{'prefix':'right_mirror', 'labels':'right_mirror'},
            }
    SEX = {'F':0, 'M':1, 'X':2}
    GROUP = {'CN':0, 'MCI':1, 'AD':2, 'SMC':3, 'EMCI':4, 'LMCI':5 }

    ap = argparse.ArgumentParser(description="Prepare ADNI turboprep output data into parts nifti files and csv metadata.")
    ap.add_argument("--root", default='data/turboprep_out_1114', help="Root directory to tuboprep output nifti files.")
    ap.add_argument("--part", default="whole_brain,left,right", help=str(list(PARTS.keys())))
    ap.add_argument("--outdir", default="data", help="Root output directory for processed data.")
    ap.add_argument("--postfix", default=str(datetime.datetime.now().date()), help="Postfix for outdir. Default: current date; Ex: ADNI_turboprepout_/{part/}_2026-01-07")
    ap.add_argument("--combine", default=True)
    ap.add_argument('--hemi_csv', default=True)
    args = ap.parse_args()

    root = args.root

    with open('data/MPRAGE-all-1213_1_07_2026.json', 'r') as f:
        conditions = json.load(f)

    parts = list(args.part.split(','))
    print("Processing parts: ", parts)

    for part in parts:
        prefix = PARTS[part]['prefix']
        labels = PARTS[part]['labels']
        save_dir = os.path.join(args.outdir, f'ADNI_turboprepout_{prefix}_{args.postfix}')
        os.makedirs(save_dir, exist_ok=True)
        print('Saving to ', save_dir)
        part_df = []
        if part == 'hemi':
            hemis = ['lhemi', 'rhemi_mirror']
            for hemi in hemis:
                csv_path = os.path.join(args.outdir, f'{hemi}_{args.postfix}.csv')
                hemi_df = pd.read_csv(csv_path)
                part_df.append(hemi_df)
            part_df = pd.concat(part_df, axis=0, ignore_index=True)
        else:
            for nii in tqdm(os.listdir(root)):
                try:
                    t1_path = os.path.join(root, nii, 'normalized.nii.gz')
                    seg_path = os.path.join(root, nii, 'segm.nii.gz')
                    mask_path = os.path.join(root, nii, 'mask.nii.gz')
                    postfix = '_normalized.nii.gz'
                    part_path, mask_path, part_vol = save_part_nii(t1_path, seg_path, None, labels, 
                                                                   prefix, postfix, save_dir, 
                                                                   whole_brain_norm=False)
                    image_id = get_imageID(part_path)
                    part_df.append({'imageID': image_id, 
                                    'image': part_path, 'mask': mask_path, 
                                    'part': part,
                                    'age': conditions[str(image_id)]['Age']/100, 
                                    'sex': SEX[conditions[str(image_id)]['Sex']], 
                                    'group': GROUP[conditions[str(image_id)]['Group']], 
                                    'vol': part_vol})
                except Exception as e:
                    print(f"Error processing {nii}: {e}")
                
            part_df = pd.DataFrame(part_df)

        df_path = os.path.join(args.outdir, f'{prefix}_{args.postfix}.csv')
        part_df.to_csv(df_path, index=False)
        print('Saved to ', df_path)

        print(part_df.head())


    if args.combine == True:
        df = pd.DataFrame()
        for part in parts:
            part = PARTS[part]['prefix']
            csv_path = os.path.join(args.outdir, f'{part}_{args.postfix}.csv')
            part_df = pd.read_csv(csv_path)
            part_df.sort_values(by='imageID', ascending=True, inplace=True, ignore_index=True)
            df['imageID'] = list(part_df['imageID'])
            df['age'] = list(part_df['age'])    
            df['sex'] = list(part_df['sex'])    
            df['group'] = list(part_df['group'])
            df[part] = list(part_df['image'])
            df[part+'_mask'] = list(part_df['mask'])
            df[part+'_part'] = list(part_df['part'])
            df[part+'_vol'] = list(part_df['vol'])
        df.to_csv(os.path.join(args.outdir, f'whole_brain+3parts+masks_{args.postfix}.csv'))
        print(df.head())

    if args.hemi_csv == True:
        df = pd.read_csv(os.path.join(args.outdir, f'whole_brain+3parts+masks_{args.postfix}.csv'))
        imgs = df.set_index("imageID").to_dict(orient="index")
        df_paired = []
        for id in imgs.keys():
            img = imgs[id]
            for part, pair in [('lhemi', 'rhemi_mirror'), ('rhemi_mirror', 'lhemi')]:
                row = {'imageID': id, 'age': img['age'], 'sex': img['sex'], 'group': img['group']}
                row['image'] = img[part]
                row['mask'] = img[part+'_mask']
                row['part'] = 0 if part =='lhemi' else 1
                row['vol'] = img[part+'_vol']
                row['pair'] = img[pair]
                row['pair_mask'] = img[pair+'_mask']
                row['pair_part'] = 0 if pair == 'lhemi' else 1
                row['pair_vol'] = img[pair+'_vol']
                df_paired.append(row)
        df_paired = pd.DataFrame(df_paired)
        df_paired.to_csv(os.path.join(args.outdir, f'hemi_{args.postfix}.csv'))
        print(df_paired.head())
        

if __name__ == "__main__":
   main()
