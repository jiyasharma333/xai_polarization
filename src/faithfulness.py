import os
import json
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_confidence(model, tokenizer, texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
    return probs

def deletion_test(model, tokenizer, explanations, k_values=[3, 5]):
    drops = {k: [] for k in k_values}
    
    for item in explanations:
        text = item['text']
        scores = item['all_scores']
        
        # Sort tokens by importance descending
        sorted_tokens = sorted(scores, key=lambda x: x[1], reverse=True)
        
        orig_conf = compute_confidence(model, tokenizer, [text])[0]
        
        for k in k_values:
            top_k_tokens = [t[0] for t in sorted_tokens[:k]]
            
            # Simple string deletion of token (heuristic, subwords might need detokenization)
            mod_text = text
            for t in top_k_tokens:
                # Remove ## from subwords
                clean_t = t.replace('##', '')
                mod_text = mod_text.replace(clean_t, '')
            
            mod_conf = compute_confidence(model, tokenizer, [mod_text])[0]
            drops[k].append(max(0, orig_conf - mod_conf))
            
    return {k: np.mean(v) for k, v in drops.items()}

def insertion_test(model, tokenizer, explanations, data_df):
    increases = []
    
    polarized_exps = [e for e in explanations if e['label'] == 1]
    non_polarized_texts = data_df[data_df['polarization'] == 0]['text_clean'].tolist()
    
    # Take 50 samples
    sample_size = min(50, len(polarized_exps), len(non_polarized_texts))
    
    for i in range(sample_size):
        pol_item = polarized_exps[i]
        non_pol_text = non_polarized_texts[i]
        
        orig_conf = compute_confidence(model, tokenizer, [non_pol_text])[0]
        
        top_3_tokens = [t[0].replace('##', '') for t in sorted(pol_item['all_scores'], key=lambda x: x[1], reverse=True)[:3]]
        
        # Insert at the beginning randomly
        mod_text = " ".join(top_3_tokens) + " " + non_pol_text
        mod_conf = compute_confidence(model, tokenizer, [mod_text])[0]
        
        increases.append(max(0, mod_conf - orig_conf))
        
    return np.mean(increases)

def evaluate_faithfulness(model_path, data_path, exp_dir, output_path):
    print(f"Loading model for faithfulness testing: {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()
    
    df = pd.read_csv(data_path)
    
    results = []
    methods = ['attention', 'ig']
    
    for method in methods:
        exp_file = os.path.join(exp_dir, f'explanations_{method}.json')
        if not os.path.exists(exp_file):
            print(f"Skipping {method}, explanations not found.")
            continue
            
        with open(exp_file, 'r') as f:
            explanations = json.load(f)
            
        print(f"Running Deletion Test for {method}...")
        del_results = deletion_test(model, tokenizer, explanations, k_values=[3, 5])
        
        print(f"Running Insertion Test for {method}...")
        ins_result = insertion_test(model, tokenizer, explanations, df)
        
        results.append({
            "Method": method.upper(),
            "Deletion_Drop_k3": del_results[3],
            "Deletion_Drop_k5": del_results[5],
            "Insertion_Increase": ins_result
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_path, index=False)
    print(f"Saved faithfulness results to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'mbert_multilingual')
    data_path = os.path.join(base_dir, 'data', 'test.csv')
    exp_dir = os.path.join(base_dir, 'explanations')
    output_path = os.path.join(base_dir, 'results', 'faithfulness_results.csv')
    
    evaluate_faithfulness(model_path, data_path, exp_dir, output_path)
