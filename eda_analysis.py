"""
=================================================================================
EXPLORATORY DATA ANALYSIS - Portuguese Bank Marketing Campaigns
=================================================================================
Dataset: bank.csv (10% of bank-full.csv with 4,521 records)
Goal: Predict if a client will subscribe to a term deposit (variable y)
Campaign Type: Direct phone call marketing
=================================================================================
"""

# ==============================================================================
# SECTION 1: IMPORTS AND INITIAL SETUP
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("\n" + "="*80)
print("STARTING EXPLORATORY DATA ANALYSIS")
print("="*80 + "\n")

# ==============================================================================
# SECTION 2: LOAD AND INSPECT DATASET
# ==============================================================================

# Load the dataset
print("[STEP 1] Loading dataset...")
import csv
try:
    # Try to detect delimiter using csv.Sniffer on a small sample
    with open('bank.csv', 'r', encoding='utf-8') as fh:
        sample = fh.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=",	;")
        delim = dialect.delimiter
    df = pd.read_csv('bank.csv', sep=delim)
except Exception:
    # Fallback: let pandas infer (engine='python') or default to comma
    try:
        df = pd.read_csv('bank.csv', sep=None, engine='python')
    except Exception:
        df = pd.read_csv('bank.csv')

# Normalize column names
df.columns = [c.strip() for c in df.columns]

# If target column 'y' is missing, try common alternatives
if 'y' not in df.columns:
    cols_lower = [c.lower().strip() for c in df.columns]
    if 'y' in cols_lower:
        # map exact-lowercase match back to original name
        idx = cols_lower.index('y')
        real = df.columns[idx]
        df.rename(columns={real: 'y'}, inplace=True)
    else:
        candidates = ['subscribed', 'response', 'subscription', 'deposit', 'y.', 'yes/no', 'class']
        found = False
        for cand in candidates:
            if cand in cols_lower:
                real = df.columns[cols_lower.index(cand)]
                df.rename(columns={real: 'y'}, inplace=True)
                found = True
                break
        if not found:
            print(f"ERROR: target column 'y' not found. Columns: {list(df.columns)}")
            raise KeyError("Target column 'y' not found in CSV. Please rename the target column to 'y' or set it manually in the script.")

# Display basic information
print(f"✓ Dataset loaded successfully!")
print(f"  - Shape: {df.shape} (rows, columns)")
print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())
print("\n")

# ==============================================================================
# SECTION 3: DATA TYPES AND MISSING VALUES
# ==============================================================================

print("[STEP 2] Data types and missing values analysis...")
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
missing = df.isnull().sum()
print(missing)
print(f"Total missing values: {missing.sum()}")
print("\n")

# ==============================================================================
# SECTION 4: STATISTICAL SUMMARY
# ==============================================================================

print("[STEP 3] Statistical summary of numerical features...")
print(df.describe())
print("\n")

# Display unique values for categorical features
print("🏷️  Categorical Features - Unique Values:")
for col in df.select_dtypes(include='object').columns:
    print(f"  {col}: {df[col].nunique()} unique values - {df[col].unique()[:5]}")
print("\n")

# ==============================================================================
# SECTION 5: TARGET VARIABLE ANALYSIS
# ==============================================================================

print("[STEP 4] Target Variable ('y') - Term Deposit Subscription Analysis...")
target_counts = df['y'].value_counts()
target_pct = df['y'].value_counts(normalize=True) * 100

print(f"\nSubscription Status Distribution:")
print(f"  {'No':>15}: {target_counts['no']:>6} ({target_pct['no']:.2f}%)")
print(f"  {'Yes':>15}: {target_counts['yes']:>6} ({target_pct['yes']:.2f}%)")
print(f"  Class Imbalance Ratio: {target_counts['no']/target_counts['yes']:.2f}:1\n")

# Create visualization for target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
sns.countplot(x='y', data=df, ax=axes[0], palette=['#0e1c2e', '#3d80cd'])
axes[0].set_title('Distribution of Term Deposit Subscriptions', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Subscription Status', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xticklabels(['No', 'Yes'])

# Pie chart
colors_pie = ['#0e1c2e', '#3d80cd']
axes[1].pie(target_counts.values, labels=['No', 'Yes'], autopct='%1.1f%%', 
            colors=colors_pie, startangle=90, textprops={'fontsize': 12})
axes[1].set_title('Subscription Percentage', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('01_target_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 01_target_distribution.png\n")
plt.close()

# ==============================================================================
# SECTION 6: NUMERICAL FEATURES ANALYSIS
# ==============================================================================

print("[STEP 5] Analyzing numerical features...")

# Identify numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numerical columns: {numerical_cols}\n")

# Distribution plots for numerical features (dynamic grid)
n_num = len(numerical_cols)
if n_num == 0:
    print("No numerical columns to plot.")
else:
    ncols = 2
    nrows = int(np.ceil(n_num / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
    axes = np.array(axes).reshape(-1)

    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        ax.hist(df[col], bins=30, color='#3d80cd', alpha=0.7, edgecolor='black')
        ax.set_title(f'Distribution of {col.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)

        # Add statistics text
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='#0e1c2e', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        ax.legend()

        print(f"  {col}:")
        print(f"    - Mean: {mean_val:.2f}")
        print(f"    - Median: {median_val:.2f}")
        print(f"    - Std Dev: {df[col].std():.2f}")
        print(f"    - Min: {df[col].min():.2f}, Max: {df[col].max():.2f}\n")

    # Hide any unused axes
    for j in range(n_num, len(axes)):
        axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('02_numerical_distributions.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 02_numerical_distributions.png\n")
plt.close()

# ==============================================================================
# SECTION 7: BOX PLOTS - OUTLIERS DETECTION
# ==============================================================================

print("[STEP 6] Detecting outliers using box plots...")

# Dynamic grid for box plots
n_num = len(numerical_cols)
if n_num == 0:
    print("No numerical columns for boxplots.")
else:
    ncols = 3
    nrows = int(np.ceil(n_num / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    axes = np.array(axes).reshape(-1)

    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        sns.boxplot(y=df[col], ax=ax, color='#3d80cd')
        ax.set_title(f'Box Plot - {col.upper()}', fontsize=12, fontweight='bold')
        ax.set_ylabel(col, fontsize=11)

        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        print(f"  {col}: {len(outliers)} outliers detected")

    for j in range(n_num, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('03_outliers_boxplots.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 03_outliers_boxplots.png\n")
    plt.close()

# ==============================================================================
# SECTION 8: CATEGORICAL FEATURES ANALYSIS
# ==============================================================================

print("[STEP 7] Analyzing categorical features...")

categorical_cols = df.select_dtypes(include='object').columns.tolist()
print(f"Categorical columns: {categorical_cols}\n")

for col in categorical_cols:
    print(f"  {col}: {df[col].nunique()} unique values")
    print(f"    Distribution:\n{df[col].value_counts()}\n")

# Create bar plots for categorical features (dynamic grid)
cat_cols_to_plot = [c for c in categorical_cols if c != 'y']
n_cat = len(cat_cols_to_plot)
if n_cat == 0:
    print("No categorical columns to plot.")
else:
    ncols = min(4, n_cat)
    nrows = int(np.ceil(n_cat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = np.array(axes).reshape(-1)

    for idx, col in enumerate(cat_cols_to_plot):
        counts = df[col].value_counts()
        ax = axes[idx]
        ax.bar(range(len(counts)), counts.values, color='#3d80cd', edgecolor='#0e1c2e', linewidth=1.5)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha='right')
        ax.set_title(f'{col.upper()} Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11)

    for j in range(n_cat, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('04_categorical_distributions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 04_categorical_distributions.png\n")
    plt.close()

# ==============================================================================
# SECTION 9: RELATIONSHIP WITH TARGET VARIABLE
# ==============================================================================

print("🔗 [STEP 8] Analyzing relationship between features and target variable...")

# Numerical features vs Target (dynamic grid)
n_num = len(numerical_cols)
if n_num == 0:
    print("No numerical columns to compare with target.")
else:
    ncols = 2
    nrows = int(np.ceil(n_num / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 4))
    axes = np.array(axes).reshape(-1)

    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        df.boxplot(column=col, by='y', ax=ax)
        ax.set_title(f'{col.upper()} vs Subscription', fontsize=12, fontweight='bold')
        ax.set_xlabel('Subscription Status', fontsize=11)
        ax.set_ylabel(col, fontsize=11)
        plt.sca(ax)
        plt.xticks([1, 2], ['No', 'Yes'])

    for j in range(n_num, len(axes)):
        axes[j].set_visible(False)
plt.suptitle('')
plt.tight_layout()
plt.savefig('05_numerical_vs_target.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 05_numerical_vs_target.png\n")
plt.close()

# Categorical features vs Target (dynamic grid)
cat_cols_to_plot = [c for c in categorical_cols if c != 'y']
n_cat = len(cat_cols_to_plot)
if n_cat == 0:
    print("No categorical columns to compare with target.")
else:
    ncols = min(4, n_cat)
    nrows = int(np.ceil(n_cat / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = np.array(axes).reshape(-1)

    for idx, col in enumerate(cat_cols_to_plot):
        # Create cross-tabulation
        crosstab = pd.crosstab(df[col], df['y'])
        crosstab_pct = pd.crosstab(df[col], df['y'], normalize='index') * 100
        
        x_pos = np.arange(len(crosstab.index))
        width = 0.35
        ax = axes[idx]
        ax.bar(x_pos - width/2, crosstab_pct['no'], width, label='No', color='#0e1c2e')
        ax.bar(x_pos + width/2, crosstab_pct['yes'], width, label='Yes', color='#3d80cd')
        
        ax.set_title(f'{col.upper()} vs Subscription (%)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(crosstab.index, rotation=45, ha='right')
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.legend()

    for j in range(n_cat, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('06_categorical_vs_target.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 06_categorical_vs_target.png\n")
    plt.close()

# ==============================================================================
# SECTION 10: CORRELATION ANALYSIS
# ==============================================================================

print("[STEP 9] Correlation analysis...")

# Prepare data for correlation (encode categorical variables)
df_encoded = df.copy()

# Encode target variable
df_encoded['y'] = (df_encoded['y'] == 'yes').astype(int)

# Encode categorical features
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    le_dict[col] = le

# Calculate correlation matrix
correlation_matrix = df_encoded.corr()

# Plot correlation heatmap with blue color scheme
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            cbar_kws={'label': 'Correlation Coefficient'}, ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('07_correlation_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 07_correlation_matrix.png\n")
plt.close()

# Display top correlations with target variable
print("Top features correlated with target variable (y):")
target_corr = correlation_matrix['y'].sort_values(ascending=False)
for feature, corr in target_corr.items():
    if feature != 'y':
        print(f"  {feature}: {corr:.4f}")
print("\n")

# ==============================================================================
# SECTION 11: DATA CLEANING AND PREPROCESSING
# ==============================================================================

print("🧹 [STEP 10] Data cleaning and preprocessing...")

# Create a copy for model building
df_model = df.copy()

# Convert target to binary (1 for 'yes', 0 for 'no')
df_model['y'] = (df_model['y'] == 'yes').astype(int)

# Encode categorical variables
print("  - Encoding categorical features...")
for col in categorical_cols:
    if col != 'y':
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])

print("✓ Data preprocessing completed!\n")

# ==============================================================================
# SECTION 12: FEATURE ENGINEERING INSIGHTS
# ==============================================================================

print("🔧 [STEP 11] Feature importance insights...")

# Prepare data for model
X = df_model.drop('y', axis=1)
y = df_model['y']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"  - Training set: {X_train.shape[0]} samples")
print(f"  - Test set: {X_test.shape[0]} samples")
print(f"  - Training set subscription rate: {y_train.mean():.2%}")
print(f"  - Test set subscription rate: {y_test.mean():.2%}\n")

# Train Random Forest for feature importance
print("  - Training Random Forest for feature importance...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
print("\n")

# Plot feature importance
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature', 
            palette=['#87CEFA' if i == 0 else '#3d80cd' for i in range(10)], ax=ax)
ax.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('08_feature_importance.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 08_feature_importance.png\n")
plt.close()

# ==============================================================================
# SECTION 13: MODEL TRAINING AND EVALUATION
# ==============================================================================

print("[STEP 12] Training machine learning models...")

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = {}

for model_name, model in models.items():
    print(f"\n  Training {model_name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    results[model_name] = {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"    ✓ Accuracy:  {results[model_name]['accuracy']:.4f}")
    print(f"    ✓ Precision: {results[model_name]['precision']:.4f}")
    print(f"    ✓ Recall:    {results[model_name]['recall']:.4f}")
    print(f"    ✓ F1-Score:  {results[model_name]['f1']:.4f}")
    print(f"    ✓ ROC-AUC:   {results[model_name]['roc_auc']:.4f}")

print("\n")

# ==============================================================================
# SECTION 14: MODEL COMPARISON VISUALIZATIONS
# ==============================================================================

print("[STEP 13] Creating model comparison visualizations...")

# Performance metrics comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

metrics_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
})

# Plot 1: Metrics comparison
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics_to_plot))
width = 0.35

for idx, model_name in enumerate(results.keys()):
    values = [metrics_df[metrics_df['Model'] == model_name][m].values[0] for m in metrics_to_plot]
    axes[0].bar(x + idx*width, values, width, label=model_name, 
                color=['#3d80cd', '#0e1c2e'][idx])

axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_xticks(x + width/2)
axes[0].set_xticklabels(metrics_to_plot, rotation=45, ha='right')
axes[0].legend()
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: ROC Curves
for model_name in results.keys():
    fpr, tpr, _ = roc_curve(y_test, results[model_name]['y_pred_proba'])
    roc_auc = results[model_name]['roc_auc']
    color = '#3d80cd' if model_name == 'Logistic Regression' else '#0e1c2e'
    axes[1].plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', 
                linewidth=2, color=color)

axes[1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
axes[1].set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('09_models_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 09_models_comparison.png\n")
plt.close()

# ==============================================================================
# SECTION 15: CONFUSION MATRICES
# ==============================================================================

print("[STEP 14] Confusion matrices for each model...")

fig, axes = plt.subplots(1, len(results), figsize=(12, 4))

for idx, (model_name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['y_pred'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                cbar=False, annot_kws={'fontsize': 14})
    axes[idx].set_title(f'{model_name} Confusion Matrix', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=11)
    axes[idx].set_xlabel('Predicted Label', fontsize=11)
    axes[idx].set_xticklabels(['No', 'Yes'])
    axes[idx].set_yticklabels(['No', 'Yes'])
    
    print(f"\n{model_name} Confusion Matrix:")
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")
    
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, result['y_pred'], target_names=['No', 'Yes']))

plt.tight_layout()
plt.savefig('10_confusion_matrices.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 10_confusion_matrices.png\n")
plt.close()

# ==============================================================================
# SECTION 16: SAVE MODELS AND DATA
# ==============================================================================

print("💾 [STEP 15] Saving models and preprocessed data...")

import pickle

# Save the best model (Random Forest has highest ROC-AUC in most cases)
best_model = results['Random Forest']['model']
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Saved: best_model.pkl")

# Save the label encoders
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(le_dict, f)
print("✓ Saved: label_encoders.pkl")

# Save preprocessed test data for Streamlit
df_model.to_csv('df_model_preprocessed.csv', index=False)
print("✓ Saved: df_model_preprocessed.csv")

print("\n" + "="*80)
print("✅ EDA COMPLETED SUCCESSFULLY!")
print("="*80 + "\n")

print("Generated Files:")
print("  1. 01_target_distribution.png")
print("  2. 02_numerical_distributions.png")
print("  3. 03_outliers_boxplots.png")
print("  4. 04_categorical_distributions.png")
print("  5. 05_numerical_vs_target.png")
print("  6. 06_categorical_vs_target.png")
print("  7. 07_correlation_matrix.png")
print("  8. 08_feature_importance.png")
print("  9. 09_models_comparison.png")
print("  10. 10_confusion_matrices.png")
print("  11. best_model.pkl")
print("  12. label_encoders.pkl")
print("  13. df_model_preprocessed.csv")
print("\n" + "="*80 + "\n")
