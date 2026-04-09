import os
import re
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import logging
import torch
import numpy as np
import random
import glob

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
    # Remove @USERmentions
    text = re.sub(r'@\w+', '', text)
    text = text.replace('@USER', '')
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(input_dir, output_dir):
    logger.info(f"Loading data from {input_dir}")
    if not os.path.exists(input_dir):
        logger.error(f"Dataset directory not found: {input_dir}. Please place the dataset here.")
        return

    # Find all CSV files recursively, skipping Trial_Data.csv
    csv_files = glob.glob(os.path.join(input_dir, '**', '*.csv'), recursive=True)
    csv_files = [f for f in csv_files if 'Trial_Data' not in f and os.path.basename(f) not in ['train.csv', 'dev.csv', 'test.csv']]
    
    if not csv_files:
        logger.error(f"No valid CSV files found in {input_dir} (excluding Trial_Data.csv).")
        return

    logger.info(f"Found {len(csv_files)} CSV files to process.")
    
    df_list = []
    for f in csv_files:
        try:
            temp_df = pd.read_csv(f)
            if 'lang' not in temp_df.columns:
                temp_df['lang'] = os.path.basename(f).split('.')[0]
            df_list.append(temp_df)
        except Exception as e:
            logger.warning(f"Could not read {f}: {e}")

    df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Total rows loaded: {len(df)}")

    # Ensure required columns exist
    required_cols = {'text', 'lang', 'polarization'}
    if not required_cols.issubset(set(df.columns)):
        logger.error(f"Missing required columns in dataset. Found: {df.columns}, Required: {required_cols}")
        return

    # Filter languages
    allowed_langs = ['eng', 'hin', 'tel']
    df = df[df['lang'].isin(allowed_langs)].copy()
    
    # Map polarization string labels to binary if needed
    if df['polarization'].dtype == object:
        polarization_mapping = {'not polarized': 0, 'polarized': 1, 'not_polarized': 0}
        df['polarization'] = df['polarization'].str.lower().str.strip().map(polarization_mapping).fillna(df['polarization'])
    
    # Convert to int, drop NaNs
    df['polarization'] = pd.to_numeric(df['polarization'], errors='coerce')
    df.dropna(subset=['polarization'], inplace=True)
    df['polarization'] = df['polarization'].astype(int)

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
        logger.warning(f"Warning: Disabling stratified split because some classes have fewer than 4 members (min is {min_class_counts}).")

    # 70/15/15 split
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=stratify_param, random_state=42)
    
    if stratify_param is not None:
        min_temp = temp_df['stratify_col'].value_counts().min()
        stratify_param2 = temp_df['stratify_col'] if min_temp >= 2 else None
    else:
        stratify_param2 = None
        
    dev_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=stratify_param2, random_state=42)
    
    for d in [train_df, dev_df, test_df]:
        d.drop(columns=['stratify_col'], inplace=True, errors='ignore')
        
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    dev_df.to_csv(os.path.join(output_dir, 'dev.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    logger.info(f"Saved splits to {output_dir}:")
    logger.info(f"  train.csv: {len(train_df)} rows")
    logger.info(f"  dev.csv: {len(dev_df)} rows")
    logger.info(f"  test.csv: {len(test_df)} rows")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Input dir dynamically reads from the data-public-main folder
    input_dir = os.path.join(base_dir, 'data', 'data-public-main')
    # Output dir for splits
    output_dir = os.path.join(base_dir, 'data')
    preprocess_data(input_dir, output_dir)
