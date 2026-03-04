import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_attention_explanations(model_path, data_path, output_path, k=5):
    """
    Extracts attention weights from the final Transformer layer.
    Highlights top-k tokens with the highest attention weight toward the [CLS] token.
    """
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, output_attentions=True).to(DEVICE)
    model.eval()

    df = pd.read_csv(data_path)
    
    # We select a sample of 200 instances (balanced across target label and hopefully languages).
    # Since we need 100 polarized and 100 non-polarized:
    df_sample = pd.concat([
        df[df['polarization'] == 1].sample(100, random_state=42, replace=True),
        df[df['polarization'] == 0].sample(100, random_state=42, replace=True)
    ]).reset_index(drop=True)

    results = []

    print("Extracting attention...")
    for idx, row in df_sample.iterrows():
        text = str(row['text_clean'])
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Get attentions from the last layer (tuple of layers, each [batch, heads, seq, seq])
        attentions = outputs.attentions[-1]
        
        # Mean pool across attention heads
        attentions = attentions.mean(dim=1) # [batch, seq, seq]
        
        # Get attention towards the [CLS] token (index 0) from all other tokens
        # We look at how much CLS attends to each token: attentions[0, 0, :]
        cls_attention = attentions[0, 0, :].cpu().numpy()
        
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Exclude special tokens if needed, but we keep them to strictly align indices
        # We just find top-k directly from the attention array
        token_scores = [(tokens[i], float(cls_attention[i])) for i in range(len(tokens)) if tokens[i] not in tokenizer.all_special_tokens]
        
        # Sort by importance and get top k
        token_scores.sort(key=lambda x: x[1], reverse=True)
        top_k = token_scores[:k]

        results.append({
            "id": row.get('id', idx), 
            "text": text,
            "label": int(row['polarization']),
            "lang": row['lang'],
            "top_tokens": top_k,
            "all_scores": token_scores
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved attention explanations to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Typically we use the best multilingual model (e.g. mBERT)
    model_path = os.path.join(base_dir, 'models', 'mbert_multilingual')
    data_path = os.path.join(base_dir, 'data', 'test.csv')
    output_path = os.path.join(base_dir, 'explanations', 'explanations_attention.json')
    
    if os.path.exists(model_path) and os.path.exists(data_path):
        get_attention_explanations(model_path, data_path, output_path)
    else:
        print("Model or dataset missing. Train transformers first.")
