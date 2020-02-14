from fastprogress import progress_bar
import pandas as pd
import os
import shutil

df = pd.read_csv('../data/landmarks.csv')
df = df.rename(columns={'_file':'filename'})
df['filename'] = df['filename'].str.split('/').str[-1]
df = df.set_index('filename')
df =df[[
    'box/part/0/_x', 
    'box/part/0/_y', 
    'box/part/1/_x', 
    'box/part/1/_y',
    'box/part/2/_x', 
    'box/part/2/_y',
    'box/part/3/_x', 
    'box/part/3/_y',
    'box/part/4/_x', 
    'box/part/4/_y',
    'box/part/5/_x', 
    'box/part/5/_y',
    'box/part/6/_x', 
    'box/part/6/_y',
    'box/part/7/_x', 
    'box/part/7/_y',
    'box/part/8/_x', 
    'box/part/8/_y',
    'box/part/9/_x', 
    'box/part/9/_y',
    'box/part/10/_x', 
    'box/part/10/_y',
]]

old_cols = list(df.columns)
splits = [f.split('/') for f in old_cols]
new_cols = [s[-2]+s[-1][-1] for s in splits]

dic = {}
for i in range(len(old_cols)):
    dic[old_cols[i]] = new_cols[i]

df = df.rename(columns=dic)

df.to_csv('../data/cleaned_labels.csv')

if not os.path.isdir('../data/cleaned_images'):
    os.mkdir('../data/cleaned_images')

for image in progress_bar(os.listdir('../data/images')):
    if image in list(y.index):
        shutil.copyfile(f'../data/images/{image}', f'../data/cleaned_images/{image}')