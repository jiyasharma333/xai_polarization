import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    print("Please install lime: pip install lime")
    LimeTextExplainer = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_lime_explanations(model_path, data_path, output_path, n_samples=50):
    if LimeTextExplainer is None: return

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()

    df = pd.read_csv(data_path)
    # Subset of 50 samples due to LIME being slow
    df_sample = pd.concat([
        df[df['polarization'] == 1].sample(n_samples//2, random_state=42, replace=True),
        df[df['polarization'] == 0].sample(n_samples//2, random_state=42, replace=True)
    ]).reset_index(drop=True)

    explainer = LimeTextExplainer(class_names=["Not Polarized", "Polarized"])

    def predictor(texts):
        # LIME calls this predictor with a list of strings
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        return probs

    results = []
    print(f"Extracting LIME explanations for {n_samples} samples...")
    for idx, row in df_sample.iterrows():
        text = str(row['text_clean'])
        target_class = int(row['polarization'])

        # 100 perturbations per sample as requested
        exp = explainer.explain_instance(text, predictor, num_features=20, num_samples=100)
        
        # Get explanation for the polarized class (class 1)
        lime_weights = exp.as_list(label=1)

        results.append({
            "id": row.get('id', idx),
            "text": text,
            "label": target_class,
            "lang": row['lang'],
            "all_scores": lime_weights
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved LIME explanations to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'mbert_multilingual')
    data_path = os.path.join(base_dir, 'data', 'test.csv')
    output_path = os.path.join(base_dir, 'explanations', 'explanations_lime.json')
    
    if os.path.exists(model_path) and os.path.exists(data_path):
        get_lime_explanations(model_path, data_path, output_path)
    else:
        print("Model or dataset missing.")
