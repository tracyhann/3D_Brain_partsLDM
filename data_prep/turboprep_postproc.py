#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import nibabel as nib
import argparse
from tqdm import tqdm
import shutil
from monai import transforms


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



def main():
    ap = argparse.ArgumentParser(description="Normalize ADNI turboprep output data.")
    ap.add_argument("--root", default='data/turboprep_out_1114', help="Root directory to tuboprep output nifti files.")
    ap.add_argument("--outdir", default='data/processed_turboprep_out_1114', help="Output directory for processed data.")
    ap.add_argument("--n_samples", default='ALL', help='Number of volumes to be processed.')

    args = ap.parse_args()

    root = args.root
    try: 
        n = int(args.n_samples)
    except:
        n = 'ALL'
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    entries = os.listdir(root)
    if n != 'ALL':
        entries = entries[:n]
    for nii in tqdm(entries):
        try:
            t1_path = os.path.join(root, nii, 'normalized.nii.gz')
            mask_path = os.path.join(root, nii, 'mask.nii.gz')

            if not os.path.exists(t1_path) or not os.path.exists(mask_path):
                print('Missing normalized or mask for: ', nii)
                continue

            t1_img = nib.load(t1_path)
            t1 = t1_img.get_fdata().astype(np.float32)
            mask = nib.load(mask_path).get_fdata().astype(np.uint8)

            print(nii)
            
            norm_t1 = normalize_mri(t1, mask)

            segm_path = os.path.join(root, nii, 'segm.nii.gz')
            if not os.path.exists(segm_path):
                print('Missing segm for: ', nii)
                continue

            os.makedirs(os.path.join(outdir, nii), exist_ok=True)
            out_norm_path = os.path.join(outdir, nii, 'normalized.nii.gz')
            out_segm_path = os.path.join(outdir, nii, 'segm.nii.gz')
            out_mask_path = os.path.join(outdir, nii, 'mask.nii.gz')

            nib.save(nib.Nifti1Image(norm_t1, t1_img.affine, t1_img.header), out_norm_path)
            shutil.copy2(mask_path, out_mask_path)  # copy2 keeps metadata
            shutil.copy2(segm_path, out_segm_path)  # copy2 keeps metadata

            keys = ["normalized", "mask", "segm"]

            data_transforms = transforms.Compose(
                [
                    transforms.LoadImaged(keys=keys),
                    transforms.EnsureChannelFirstd(keys=keys),
                    transforms.EnsureTyped(keys=keys),
                    transforms.CropForegroundd(keys=keys, source_key="mask"),
                    transforms.DivisiblePadd(keys=keys, k=32, mode="constant", constant_values=-1.0),
                    transforms.SpatialPadd(keys=keys, spatial_size=(160,192,160), mode="constant", constant_values=-1.0),
                ]
            )

            # inside your loop for each subject:
            data = {
                "normalized": out_norm_path,
                "mask": out_mask_path,
                "segm": out_segm_path,
            }
            out = data_transforms(data)

            # save results
            norm_out = np.asarray(out["normalized"])
            mask_out = np.asarray(out["mask"])
            norm_out = np.where(mask_out, norm_out, -1.0).astype(np.float32)
            segm_out = np.asarray(out["segm"])
            if norm_out.ndim == 4 and norm_out.shape[0] == 1:
                norm_out = norm_out[0]
            if mask_out.ndim == 4 and mask_out.shape[0] == 1:
                mask_out = mask_out[0]
            if segm_out.ndim == 4 and segm_out.shape[0] == 1:
                segm_out = segm_out[0]

            nib.save(nib.Nifti1Image(norm_out, t1_img.affine, t1_img.header), out_norm_path)
            nib.save(nib.Nifti1Image(mask_out, t1_img.affine, t1_img.header), out_mask_path)
            nib.save(nib.Nifti1Image(segm_out,  t1_img.affine, t1_img.header), out_segm_path)

            print('Processed: ', nii)
        except:
            print('Failed to process: ', nii)


if __name__ == "__main__":
   main()
