# Credit Card Fraud Detection

Machine learning project for detecting fraudulent credit card transactions using classification algorithms on highly imbalanced data.

## Project Objective
- Binary classification: Fraud (1) vs Legitimate (0) transactions
- Focus on detecting fraud cases with high recall
- Handle highly imbalanced dataset (~0.17% fraud rate)

## Dataset
- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: ~284,000 transactions
- **Fraud rate**: 0.17% (highly imbalanced)
- **Features**:
  - V1-V28: PCA-transformed anonymized features
  - Amount: Transaction amount
  - Time: Seconds elapsed between transactions
  - Class: Target (0 = Legitimate, 1 = Fraud)

## Technologies
- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- imbalanced-learn (SMOTE)
- Streamlit

## Project Structure
```
Credit Card Fraud Detection/
├── data/
│   └── creditcard.csv          # Dataset (download separately)
├── src/
│   ├── data_loader.py          # Data loading
│   ├── eda.py                  # Exploratory analysis
│   ├── preprocessing.py        # Data preprocessing
│   ├── models.py               # Model building
│   ├── evaluation.py           # Model evaluation
│   └── utils.py                # Utilities
├── plots/                      # Generated visualizations
├── results/                    # Trained models and metrics
├── main.py                     # Training pipeline
├── app.py                      # Streamlit web app
├── requirements.txt            # Dependencies
└── README.md
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Credit Card Fraud Detection"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download dataset**
   - Get the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   - Place `creditcard.csv` in the `data/` folder

## Usage

**Train models:**
```bash
python main.py
```

**Launch web app:**
```bash
streamlit run app.py
```

## Models
1. **Logistic Regression** - Fast, interpretable baseline
2. **Random Forest** - Ensemble method for complex patterns
3. **Isolation Forest** - Unsupervised anomaly detection (optional)

## Evaluation Metrics
- **Precision** - Minimize false fraud alerts
- **Recall** - Maximize fraud detection (priority metric)
- **F1-Score** - Balance precision and recall
- **ROC-AUC** - Overall performance
- **Confusion Matrix** - Detailed prediction breakdown

## Key Features
- Handles class imbalance with class weights and SMOTE
- Feature scaling with StandardScaler
- Comprehensive EDA and visualizations
- Multiple model comparison
- Threshold tuning for optimal detection
- Interactive Streamlit deployment

## Results
Models are compared on:
- **Recall** (priority for fraud detection)
- **F1-Score** (precision-recall balance)
- **ROC-AUC** (discrimination ability)

Results are saved in `results/model_comparison.csv`

## License
MIT License
