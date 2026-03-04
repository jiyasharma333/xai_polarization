import os
import json
import pandas as pd
from collections import defaultdict

def identify_category(token):
    token = token.lower()
    # Simplified mock categories
    identity_terms = ['muslim', 'hindu', 'christian', 'white', 'black', 'indian', 'american', 'women', 'men', 'gay', 'straight', 'group', 'people', 'they', 'them']
    political_terms = ['liberty', 'liberal', 'conservative', 'party', 'vote', 'election', 'minister', 'president', 'government', 'left', 'right', 'bjp', 'congress', 'democrat', 'republican']
    emotional_terms = ['hate', 'angry', 'disgust', 'fear', 'threat', 'ruin', 'destroy', 'love', 'happy']
    
    if token in identity_terms:
        return 'Identity'
    elif token in political_terms:
        return 'Political'
    elif token in emotional_terms:
        return 'Emotional'
    return 'Other'

def run_bias_audit(data_path, ig_exp_path, output_path, threshold=0.15):
    print("Running Bias Audit...")
    try:
        with open(ig_exp_path, 'r') as f:
            explanations = json.load(f)
    except FileNotFoundError:
        print("IG explanations not found. Run xai_ig.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # 1. Compute corpus frequency (% of times token appears in corpus)
    # We will compute frequency over all explanations provided.
    token_doc_counts = defaultdict(int)
    total_docs = len(explanations)
    
    # 2. Compute flagged frequency (% of times flagged as important for polarized class)
    # Important: > threshold
    polarized_exps = [e for e in explanations if e['label'] == 1]
    total_pol_docs = len(polarized_exps)
    token_flag_counts = defaultdict(int)

    for item in explanations:
        seen = set([t[0].lower().replace('##', '') for t in item['all_scores']])
        for t in seen:
            token_doc_counts[t] += 1
            
    for item in polarized_exps:
        seen_flagged = set([t[0].lower().replace('##', '') for t in item['all_scores'] if t[1] > threshold])
        for t in seen_flagged:
            token_flag_counts[t] += 1

    results = []
    
    for token, flag_count in token_flag_counts.items():
        # Minimum threshold to avoid noise
        if flag_count < 2:
            continue
            
        doc_count = token_doc_counts.get(token, 1)
        
        prob_flag = flag_count / total_pol_docs
        prob_corpus = doc_count / total_docs
        
        bias_ratio = prob_flag / prob_corpus if prob_corpus > 0 else 0
        
        category = identify_category(token)
        
        results.append({
            "Token": token,
            "Category": category,
            "Flagged_Count": flag_count,
            "Corpus_Count": doc_count,
            "Bias_Ratio": bias_ratio,
            "Over_Flagged": bias_ratio > 2.0
        })
        
    if not results:
        results_df = pd.DataFrame(columns=["Token", "Category", "Flagged_Count", "Corpus_Count", "Bias_Ratio", "Over_Flagged"])
    else:
        results_df = pd.DataFrame(results).sort_values(by="Bias_Ratio", ascending=False)
        
    results_df.to_csv(output_path, index=False)
    print(f"Saved Bias Audit report to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'test.csv')
    ig_exp_path = os.path.join(base_dir, 'explanations', 'explanations_ig.json')
    output_path = os.path.join(base_dir, 'results', 'bias_audit.csv')
    
    run_bias_audit(data_path, ig_exp_path, output_path)
