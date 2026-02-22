# 🏦 Bank Marketing Prediction

A machine learning dashboard that predicts if a customer will subscribe to a term deposit.

##  What It Does

- **Predicts** whether a bank customer will subscribe to a term deposit based on their profile
- **Explores** data with interactive visualizations
- **Analyzes** model performance with detailed metrics

##  Quick Start

### Requirements
- Python 3.8+
- pip

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd bank-marketing-prediction

# Install dependencies
pip install -r requirements.txt

# Run exploratory analysis (generates visualizations)
python eda_analysis.py

# Launch the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`

##  Features

- **5 Dashboard Pages:**
  -  Home - Project overview & key metrics
  -  Data Exploration - Distributions, correlations, target analysis
  -  Make Prediction - Predict for new customers
  -  Model Performance - Metrics, confusion matrix, feature importance
  -  About - Project information

##  Dataset

- **Source:** Portuguese Banking Institution
- **Records:** 4,521 customers
- **Target:** Term deposit subscription (Yes/No)
- **Features:** 16 (age, job, education, balance, duration, etc.)

##  Model

- **Algorithm:** Random Forest Classifier
- **Accuracy:** ~89%
- **Precision:** ~50%
- **ROC-AUC:** ~0.89

##  Files

```
├── app.py                      # Streamlit dashboard
├── eda_analysis.py             # Data analysis & model training
├── bank.csv                    # Dataset
├── best_model.pkl              # Trained model
├── label_encoders.pkl          # Categorical encoders
├── requirements.txt            # Dependencies
└── README.md                   # Full documentation
```

##  How to Use

1. **Run the app:** `streamlit run app.py`
2. **Explore data:** Go to "Data Exploration" tab
3. **Make predictions:** Enter customer info in "Make Prediction" tab
4. **View performance:** Check "Model Performance" for metrics

##  Key Insights

- Only 11.5% of customers subscribe to term deposits
- Call duration is the most important feature
- Previous campaign outcome strongly influences subscription

##  Tech Stack

- **Python** - Programming language
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **Matplotlib/Seaborn** - Visualization
- **Streamlit** - Web dashboard

##  Workflow

```
Raw Data → EDA & Preprocessing → Model Training → Interactive Dashboard
```

##  License

This project is for educational purposes.

##  Contributing

Feel free to fork and submit pull requests!

---

**Last Updated:** February 2026  
**Status:** Production Ready

