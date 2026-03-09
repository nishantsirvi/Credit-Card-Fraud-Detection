# Credit Card Fraud Detection

Machine learning project for detecting fraudulent credit card transactions using classification algorithms on highly imbalanced data.

## What it does
Binary classification to identify fraudulent credit card transactions. Uses the Kaggle Credit Card Fraud dataset (~284k transactions, 0.17% fraud rate) with PCA-transformed features (V1-V28), Amount, and Time columns.

## How to run it

**Setup:**
```bash
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the `data/` folder.

**Train models:**
```bash
python main.py
```

**Launch web app:**
```bash
streamlit run app.py
```

## Models
- **Logistic Regression** - Fast baseline
- **Random Forest** - Handles non-linear patterns
- **Isolation Forest** - Unsupervised anomaly detection (optional)

## License
MIT License

