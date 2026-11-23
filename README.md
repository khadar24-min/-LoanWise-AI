# LoanWise AI

An explainable machine-learning dashboard for personal-loan campaign insights, built with Streamlit.

## Features

- Personal-loan acceptance prediction using the existing trained model
- Customer profile inputs and sample profiles
- Model explanation and analytics
- Interactive 3D fintech hero visual powered by Three.js

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app expects these files in the repository root:
- `loanwise_model.pkl`
- `Bank_Personal_Loan_Modelling.xlsx`

## Free deployment: Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with GitHub.
2. Choose **Create app**.
3. Select repository `khadar24-min/-LoanWise-AI`, branch `main`, and main file path `app.py`.
4. In **Advanced settings**, choose Python 3.11 or 3.12 if needed, then deploy.
5. After the first build, open the app URL ending in `.streamlit.app`.

Community Cloud deploys directly from GitHub and automatically rebuilds when the connected branch changes. The app's 3D hero loads Three.js from a public CDN, so the visitor's browser needs internet access for that visual.

## Notes

- The existing model and prediction pipeline are retained.
- The 3D hero is an additional visual layer; it does not alter model predictions.
- Model artifacts are loaded with joblib. Keep the training scikit-learn version compatible with the installed runtime if deserialization reports a version warning or error.
