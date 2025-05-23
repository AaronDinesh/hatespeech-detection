import os
import numpy as np
from typing import Optional, List
from datasets import PartitionDataset, FeatureWiseDataset
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from torch import Tensor
import pandas as pd
import json


def load_img_text(row_id):
    file_path = os.path.join(json_folder, f"{row_id}.json")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("img_text", None)
    except Exception as e:
        print(f"Warning: Could not read {file_path} — {e}")
        return None


def preprocessing(data_dir):
    df = pd.read_json(os.path.join(data_dir, 'MMHS150K_GT.json'),\
                      lines=False, orient='index', convert_dates=False)

    df = df.reset_index()

    df['id'] = df['tweet_url'].str.extract(r'/status/(\d+)')

    df['img'] = 'img/'+df['id']+'.jpg'

    # Folder containing the JSON files
    json_folder = os.path.join(data_dir,'img_txt')

    # Function to load "img_text" from a given ID's JSON file
    def load_img_text(row_id):
        file_path = os.path.join(json_folder, f"{row_id}.json")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("img_text", None)
        except Exception as e:
            print(f"Warning: Could not read {file_path} — {e}")
            return None

    # Apply the function to the 'id' column to create a new column
    df['img_text'] = df['id'].apply(load_img_text)

    def label_agg(row):
        res = 0
        for x in row:
            if x != 0:
                res+= 1
        return res

    df['label'] = df['labels'].apply(label_agg)

    MM_df = df[['img', 'img_text','tweet_text','label','id']].copy()
    # print(MM_df[MM_df['img_text'].notna()].head())
    return MM_df

def data_splitting(MM_df, data_dir):

    # Load split files
    def load_ids(filepath):
        with open(filepath, 'r') as f:
            return set(line.strip() for line in f)

    train_ids = load_ids(os.path.join(data_dir, 'splits/train_ids.txt'))
    test_ids = load_ids(os.path.join(data_dir, 'splits/test_ids.txt'))
    val_ids = load_ids(os.path.join(data_dir, 'splits/val_ids.txt'))

    # Filter the DataFrame
    train_df =  MM_df[ MM_df['id'].isin(train_ids)].copy()
    test_df =  MM_df[ MM_df['id'].isin(test_ids)].copy()
    val_df =  MM_df[ MM_df['id'].isin(val_ids)].copy()

    return train_df, test_df, val_df



class MMHSDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['img']              # maybe preprocess here
        img_text = row['img_text']        # tokenize or embed
        tweet_text = row['tweet_text']    # tokenize or embed
        label = int(row['label'])         # convert to int/long
        return (image_path, img_text, tweet_text), label