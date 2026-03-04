import pandas as pd
import numpy as np
import os
import random
from datetime import datetime
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
set_seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.0
    logger.info(f"Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")
    return {"Model": model_name, "Language": "English", "Accuracy": acc, "Precision": prec, "Recall": rec, "Macro F1": macro_f1, "AUC-ROC": auc, "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

def train_and_eval(data_dir, output_dir, models_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    train_en = train_df[train_df['lang'] == 'eng'].dropna(subset=['text_clean', 'polarization']).copy()
    test_en = test_df[test_df['lang'] == 'eng'].dropna(subset=['text_clean', 'polarization']).copy()
    X_train, y_train = train_en['text_clean'], train_en['polarization']
    X_test, y_test = test_en['text_clean'], test_en['polarization']
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
    
    results = []
    lr_model = LogisticRegression(class_weight='balanced', random_state=42)
    lr_model.fit(X_train_vec, y_train)
    results.append(evaluate_model(lr_model, X_test_vec, y_test, "TF-IDF + LR"))
    joblib.dump(lr_model, os.path.join(models_dir, "lr_model.pkl"))
    
    svm_model = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42)
    svm_model.fit(X_train_vec, y_train)
    results.append(evaluate_model(svm_model, X_test_vec, y_test, "TF-IDF + SVM"))
    joblib.dump(svm_model, os.path.join(models_dir, "svm_model.pkl"))
    
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "results_classical_ml.csv"), index=False)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_and_eval(os.path.join(base_dir, 'data'), os.path.join(base_dir, 'results'), os.path.join(base_dir, 'models', 'classical_ml'))
