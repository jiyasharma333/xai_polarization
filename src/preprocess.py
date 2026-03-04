import os
import re
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import logging
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'@\w+', '', text)
    text = text.replace('@USER', '')
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(input_csv, output_dir):
    logger.info(f"Loading data from {input_csv}")
    if not os.path.exists(input_csv):
        logger.error("Dataset not found. Please place Trial_Data.csv in the data folder.")
        return

    df = pd.read_csv(input_csv)
    allowed_langs = ['eng', 'hin', 'tel']
    df = df[df['lang'].isin(allowed_langs)].copy()
    
    logger.info("Class distribution per language:")
    distribution = df.groupby(['lang', 'polarization']).size()
    logger.info(f"\n{distribution}")
    
    logger.info("Cleaning text...")
    df['text_clean'] = df['text'].apply(clean_text)
    df = df[df['text_clean'].str.len() > 0]
    
    df['stratify_col'] = df['lang'].astype(str) + "_" + df['polarization'].astype(str)
    
    # Determine if we can stratify
    min_class_counts = df['stratify_col'].value_counts().min()
    stratify_param = df['stratify_col'] if min_class_counts >= 4 else None
    
    if stratify_param is None:
        logger.warning(f"Warning: Disabling stratified split because some classes have fewer than 4 members in this trial data (min is {min_class_counts}).")

    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=stratify_param, random_state=42)
    
    # Check again for the second split
    if stratify_param is not None:
        min_temp = temp_df['stratify_col'].value_counts().min()
        stratify_param2 = temp_df['stratify_col'] if min_temp >= 2 else None
    else:
        stratify_param2 = None
        
    dev_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=stratify_param2, random_state=42)
    
    for d in [train_df, dev_df, test_df]:
        d.drop(columns=['stratify_col'], inplace=True)
        
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    dev_df.to_csv(os.path.join(output_dir, 'dev.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    logger.info(f"Saved splits to {output_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, 'data', 'Trial_Data.csv')
    output_dir = os.path.join(base_dir, 'data')
    preprocess_data(input_path, output_dir)
