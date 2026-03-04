import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'Trial_Data.csv')
results_dir = os.path.join(base_dir, 'results')
models_dir = os.path.join(base_dir, 'models')
exp_dir = os.path.join(base_dir, 'explanations')

st.set_page_config(page_title="POLAR XAI", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Polarization Detector", "Dataset Explorer", "Model Performance", "XAI Comparison", "Bias Audit"])

@st.cache_resource
def load_model_and_tokenizer():
    path = os.path.join(models_dir, 'mbert_multilingual')
    if os.path.exists(path):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return tokenizer, model
    return None, None

def render_heatmap(tokens, scores, title="Heatmap"):
    # Normalize scores
    import numpy as np
    m_val = max(scores) if scores else 1
    scores = [s / m_val if m_val > 0 else 0 for s in scores]
    
    html = f"<div><h4>{title}</h4><p style='line-height: 2;'>"
    for t, s in zip(tokens, scores):
        color = f"rgba(255, 0, 0, {s})"
        html += f"<span style='background-color: {color}; padding: 2px; border-radius: 3px; font-size: 16px;'>{t}</span> "
    html += "</p></div>"
    st.markdown(html, unsafe_allow_html=True)

if page == "Polarization Detector":
    st.title("Polarization Detector 🕵️")
    text = st.text_area("Enter text for analysis:")
    lang = st.selectbox("Language", ["English", "Hindi", "Telugu", "Auto-detect"])
    
    if st.button("Predict"):
        tokenizer, model = load_model_and_tokenizer()
        if model is None:
            st.error("Model not found. Please train first.")
        elif not text:
            st.warning("Please enter text.")
        else:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0].numpy()
                pred = "Polarized" if probs[1] > 0.5 else "Not Polarized"
            
            st.subheader(f"Prediction: {pred} (Confidence: {probs[1]:.2f})")
            st.progress(float(probs[1]))
            
            # Simple attention extraction for demo
            st.write("Generating Explanations...")
            
            from captum.attr import IntegratedGradients
            def custom_forward(inputs_embeds):
                outs = model(inputs_embeds=inputs_embeds, attention_mask=inputs['attention_mask'])
                return torch.softmax(outs.logits, dim=1)
            
            ig = IntegratedGradients(custom_forward)
            embeddings = model.get_input_embeddings()(inputs['input_ids'])
            baseline = torch.zeros_like(embeddings)
            
            try:
                attr, _ = ig.attribute(inputs=embeddings, baselines=baseline, target=1, n_steps=20, return_convergence_delta=True)
                attr_sum = torch.norm(attr.squeeze(0), p=2, dim=1).detach().numpy()
                tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                
                # Filter special tokens
                clean_t, clean_s = [], []
                for t, s in zip(tokens, attr_sum):
                    if t not in tokenizer.all_special_tokens:
                        clean_t.append(t)
                        clean_s.append(s)
                
                st.write("### Token Importance (Integrated Gradients)")
                render_heatmap(clean_t, clean_s, "IG Token Importance")
            except Exception as e:
                st.error(f"Error computing IG: {e}")

elif page == "Dataset Explorer":
    st.title("Dataset Explorer 📊")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Class Distribution")
            fig = px.histogram(df, x="polarization", color="lang", barmode='group')
            st.plotly_chart(fig)
            
        with col2:
            st.subheader("Data Samples")
            filt_lang = st.selectbox("Filter Lang", ['All'] + list(df['lang'].unique()))
            if filt_lang != 'All':
                df = df[df['lang'] == filt_lang]
            st.dataframe(df.head(50))
    else:
        st.error("Dataset not found.")

elif page == "Model Performance":
    st.title("Model Performance Dashboard 📈")
    res_path = os.path.join(results_dir, 'results_transformers.csv')
    clas_path = os.path.join(results_dir, 'results_classical_ml.csv')
    
    dfs = []
    if os.path.exists(res_path): dfs.append(pd.read_csv(res_path))
    if os.path.exists(clas_path): dfs.append(pd.read_csv(clas_path))
    
    if dfs:
        df = pd.concat(dfs)
        st.dataframe(df)
        
        fig = px.bar(df, x="Model", y="Macro F1", color="Language", barmode="group", title="Macro F1 across Models and Languages")
        st.plotly_chart(fig)
    else:
        st.info("No results found. Run training scripts.")

elif page == "XAI Comparison":
    st.title("XAI Comparison 🔍")
    ig_path = os.path.join(exp_dir, 'explanations_ig.json')
    att_path = os.path.join(exp_dir, 'explanations_attention.json')
    
    if os.path.exists(ig_path) and os.path.exists(att_path):
        with open(ig_path) as f: ig_data = json.load(f)
        with open(att_path) as f: att_data = json.load(f)
        
        sample_ids = [str(item['id']) for item in ig_data]
        selected_id = st.selectbox("Select Sample ID", sample_ids)
        
        ig_sample = next(item for item in ig_data if str(item['id']) == selected_id)
        att_sample = next(item for item in att_data if str(item['id']) == selected_id)
        
        st.write("**Text:**", ig_sample['text'])
        st.write("**Label:**", "Polarized" if ig_sample['label'] == 1 else "Not Polarized")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Integrated Gradients")
            t_ig, s_ig = zip(*ig_sample['all_scores'])
            render_heatmap(t_ig, s_ig, "IG Heatmap")
        with col2:
            st.subheader("Attention")
            t_att, s_att = zip(*att_sample['all_scores'])
            render_heatmap(t_att, s_att, "Attention Heatmap")
            
        f_path = os.path.join(results_dir, 'faithfulness_results.csv')
        if os.path.exists(f_path):
            st.subheader("Faithfulness Metrics")
            st.dataframe(pd.read_csv(f_path))
    else:
        st.info("XAI explanations not found.")

elif page == "Bias Audit":
    st.title("Bias Audit ⚖️")
    b_path = os.path.join(results_dir, 'bias_audit.csv')
    if os.path.exists(b_path):
        df = pd.read_csv(b_path)
        st.dataframe(df.style.applymap(lambda x: 'background-color: #ffcccc' if x == True else '', subset=['Over_Flagged']))
        
        over = df[df['Over_Flagged'] == True].head(10)
        if not over.empty:
            fig = px.bar(over, x="Token", y="Bias_Ratio", color="Category", title="Top 10 Over-flagged Terms")
            st.plotly_chart(fig)
    else:
        st.info("Bias audit results not found.")
