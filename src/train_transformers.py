import os
import pandas as pd
import numpy as np
import torch
import random
import logging
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
PATIENCE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PolarDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = [str(t) for t in texts]
        # Safely extract and cast labels to integers to prevent DataLoader crashes
        self.labels = [int(float(l)) if pd.notna(l) else 0 for l in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self): return len(self.texts)
    
    def __getitem__(self, item):
        encoding = self.tokenizer(self.texts[item], add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(self.labels[item], dtype=torch.long)}

def evaluate(model, data_loader):
    model.eval()
    val_preds, val_labels, val_probs = [], [], []
    with torch.no_grad():
        for d in data_loader:
            outputs = model(input_ids=d["input_ids"].to(DEVICE), attention_mask=d["attention_mask"].to(DEVICE))
            val_probs.extend(torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy())
            val_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            val_labels.extend(d["labels"].cpu().numpy())
    try: auc = roc_auc_score(val_labels, val_probs)
    except: auc = 0.0
    return accuracy_score(val_labels, val_preds), precision_score(val_labels, val_preds, zero_division=0), recall_score(val_labels, val_preds, zero_division=0), f1_score(val_labels, val_preds, average='macro', zero_division=0), auc

def train_model(model_name, train_df, dev_df, save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(DEVICE)
    train_loader = DataLoader(PolarDataset(train_df['text_clean'], train_df['polarization'], tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(PolarDataset(dev_df['text_clean'], dev_df['polarization'], tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*len(train_loader)*EPOCHS), num_training_steps=len(train_loader)*EPOCHS)
    best_dev_f1, patience_counter = 0, 0
    for epoch in range(EPOCHS):
        model.train()
        for d in train_loader:
            optimizer.zero_grad()
            loss = model(input_ids=d["input_ids"].to(DEVICE), attention_mask=d["attention_mask"].to(DEVICE), labels=d["labels"].to(DEVICE)).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        _, _, _, dev_f1, _ = evaluate(model, dev_loader)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} - Dev Macro F1: {dev_f1:.4f}")
        
        if dev_f1 >= best_dev_f1:
            best_dev_f1, patience_counter = dev_f1, 0
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE: break
    return save_path

def evaluate_on_test(model_path, test_df, model_display_name, lang="All"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    acc, prec, rec, f1, auc = evaluate(model, DataLoader(PolarDataset(test_df['text_clean'], test_df['polarization'], tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False))
    return {"Model": model_display_name, "Language": lang, "Accuracy": acc, "Precision": prec, "Recall": rec, "Macro F1": f1, "AUC-ROC": auc, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv')).dropna(subset=['text_clean', 'polarization'])
    dev_df = pd.read_csv(os.path.join(data_dir, 'dev.csv')).dropna(subset=['text_clean', 'polarization'])
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv')).dropna(subset=['text_clean', 'polarization'])
    
    results = []

    # Model C: Monolingual BERT
    train_en = train_df[train_df['lang'] == 'eng']
    dev_en = dev_df[dev_df['lang'] == 'eng']
    test_en = test_df[test_df['lang'] == 'eng']
    
    bert_save = os.path.join(models_dir, 'bert_monolingual')
    train_model('bert-base-uncased', train_en, dev_en, bert_save)
    if len(test_en) > 0:
        results.append(evaluate_on_test(bert_save, test_en, 'Monolingual BERT', 'English'))

    # Model D: mBERT
    mbert_save = os.path.join(models_dir, 'mbert_multilingual')
    train_model('bert-base-multilingual-cased', train_df, dev_df, mbert_save)
    results.append(evaluate_on_test(mbert_save, test_df, 'Multilingual mBERT', 'En+Hi+Te'))

    # Model E: XLM-R
    xlmr_save = os.path.join(models_dir, 'xlmr_multilingual')
    train_model('xlm-roberta-base', train_df, dev_df, xlmr_save)
    results.append(evaluate_on_test(xlmr_save, test_df, 'XLM-R Base', 'En+Hi+Te'))
    
    results_path = os.path.join(base_dir, 'results', 'results_transformers.csv')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    pd.DataFrame(results).to_csv(results_path, index=False)
    logger.info("Saved transformer results.")

if __name__ == "__main__":
    main()
