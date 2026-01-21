import os, pandas as pd 
import nibabel as nib
import numpy as np
import json
from tqdm import tqdm

root = 'data'
postfix = '0120'
parts = ['whole_brain', 'left', 'right_mirror']
CODES = {'left': 0, 'right_mirror': 1, 'right': 1}


df = pd.DataFrame()
for part in parts:
    csv_path = os.path.join(root, f'{part}_{postfix}.csv')
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

imgs = df.set_index("imageID").to_dict(orient="index")
df_paired = []
for id in imgs.keys():
    img = imgs[id]
    for part, pair in [('left', 'right_mirror'), ('right_mirror', 'left')]:
        row = {'imageID': id, 'age': img['age'], 'sex': img['sex'], 'group': img['group']}
        wb = 'whole_brain'
        row[wb] = img[wb]
        row[wb+'_mask'] = img[wb+'_mask']
        row[wb+'_vol'] = img[wb+'_vol']
        row['image'] = img[part]
        row['mask'] = img[part+'_mask']
        row['part'] = CODES[img[part+'_part']]
        row['vol'] = img[part+'_vol']
        row['pair'] = img[pair]
        row['pair_mask'] = img[pair+'_mask']
        row['pair_part'] = CODES[img[pair+'_part']]
        row['pair_vol'] = img[pair+'_vol']
        df_paired.append(row)

df_paired = pd.DataFrame(df_paired)
print(df_paired.head())
print(len(df_paired))
df_paired.to_csv('data/left_right_paired_0120.csv')
