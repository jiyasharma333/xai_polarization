# Explainable Multilingual Polarization Detection on POLAR (SemEval-2026)

This project builds a complete end-to-end NLP pipeline for binary polarization detection on multilingual social media text, paired with Explainable AI (XAI) methods to interpret model predictions.

## Setup Instructions
1. Install requirements: `pip install -r requirements.txt`
2. Run data processing: `python src/preprocess.py`
3. Train classical baselines: `python src/train_classical.py`
4. Train transformer models: `python src/train_transformers.py`
5. Run XAI scripts: `python src/xai_attention.py`, `python src/xai_ig.py` (LIME optional)
6. Evaluate Faithfulness: `python src/faithfulness.py`
7. Run Bias Audit: `python src/bias_audit.py`
8. Launch Streamlit Web App: `streamlit run streamlit_app.py`
