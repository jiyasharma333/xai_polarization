import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_ig_explanations(model_path, data_path, output_path):
    """
    Computes token importance using Integrated Gradients via Captum.
    """
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()

    # Define a custom forward function that accepts embeddings
    def custom_forward(inputs_embeds, attention_mask):
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return torch.softmax(outputs.logits, dim=1)

    ig = IntegratedGradients(custom_forward)

    df = pd.read_csv(data_path)
    df_sample = pd.concat([
        df[df['polarization'] == 1].sample(100, random_state=42, replace=True),
        df[df['polarization'] == 0].sample(100, random_state=42, replace=True)
    ]).reset_index(drop=True)

    results = []

    print("Extracting IG explanations...")
    for idx, row in df_sample.iterrows():
        text = str(row['text_clean'])
        target_class = int(row['polarization'])

        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).to(DEVICE)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Get embeddings for the input
        embeddings = model.get_input_embeddings()(input_ids)
        
        # Baseline: zero embeddings
        baseline_embeddings = torch.zeros_like(embeddings).to(DEVICE)

        with torch.backends.cudnn.flags(enabled=False): # Captum IG sometimes requires this
            attributions, delta = ig.attribute(
                inputs=embeddings,
                baselines=baseline_embeddings,
                additional_forward_args=(attention_mask,),
                target=target_class,
                n_steps=50,
                return_convergence_delta=True
            )
        
        # Aggregate across embedding dimensions by L2 norm
        attributions_sum = torch.norm(attributions.squeeze(0), p=2, dim=1).cpu().detach().numpy()
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        token_scores = [(tokens[i], float(attributions_sum[i])) for i in range(len(tokens)) if tokens[i] not in tokenizer.all_special_tokens]
        token_scores.sort(key=lambda x: x[1], reverse=True)

        results.append({
            "id": row.get('id', idx), 
            "text": text,
            "label": target_class,
            "lang": row['lang'],
            "all_scores": token_scores
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved IG explanations to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'mbert_multilingual')
    data_path = os.path.join(base_dir, 'data', 'test.csv')
    output_path = os.path.join(base_dir, 'explanations', 'explanations_ig.json')
    
    if os.path.exists(model_path) and os.path.exists(data_path):
        get_ig_explanations(model_path, data_path, output_path)
    else:
        print("Model or dataset missing.")
