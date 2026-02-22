"""
=================================================================================
STREAMLIT DASHBOARD - BANK MARKETING PREDICTION MODEL
=================================================================================
Interactive dashboard for bank term deposit subscription prediction
Features: Data exploration, model predictions, and performance metrics
Color Scheme: Green-White-Dark Green
=================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM STYLING - GREEN, WHITE, DARK GREEN THEME
# ==============================================================================

st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #ffffff;
        color: #0e1c2e;
    }
    
    /* Sidebar styling - Dark background with white text */
    [data-testid="stSidebar"] {
        background-color: #0e1c2e;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: #ffffff !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #0e1c2e;
    }
    
    /* Primary button */
    .stButton > button {
        background-color: #3d80cd;
        color: white;
        border: 2px solid #0e1c2e;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #87CEFA;
        color: #0e1c2e;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3d80cd;
    }
    
    /* Info box */
    .stInfo {
        background-color: #e6f2ff;
        border-left: 5px solid #3d80cd;
        color: #0e1c2e;
    }
    
    /* Success message */
    .stSuccess {
        background-color: #d4e8ff;
        color: #0e1c2e;
    }
    
    /* Select box */
    .stSelectbox {
        color: #0e1c2e;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {
        color: #0e1c2e;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #3d80cd;
        color: white;
        border-bottom: 3px solid #87CEFA;
    }
    
    /* Container styling */
    .stContainer {
        background-color: #ffffff;
        color: #0e1c2e;
    }
    
    /* Input fields */
    .stTextInput, .stNumberInput, .stSelectbox, .stMultiSelect {
        color: #0e1c2e;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# LOAD DATA AND MODELS
# ==============================================================================

@st.cache_resource
def load_data():
    """Load data and trained models"""
    # Load original dataframe for display and form options (real category values)
    df_original = pd.read_csv('bank.csv')
    
    # Load preprocessed dataframe for model (encoded values)
    df_model = pd.read_csv('df_model_preprocessed.csv')
    
    # Load model
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load label encoders
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    return df_original, df_model, model, label_encoders

# Load data
df, df_model_data, model, le_dict = load_data()

# Convert target column to numeric for calculations (keep original for display)
df_numeric = df.copy()
df_numeric['y'] = (df_numeric['y'] == 'yes').astype(int)

# ==============================================================================
# SIDEBAR - NAVIGATION
# ==============================================================================

st.sidebar.markdown("""
<h1 style='color: #87CEFA; text-align: center; font-size: 28px;'>🏦 BANK MARKETING</h1>
<h3 style='color: #ffffff; text-align: center;'>Term Deposit Prediction</h3>
<hr style='border: 1px solid #3d80cd;'>
""", unsafe_allow_html=True)

# Navigation menu
page = st.sidebar.radio(
    "📌 Select Page",
    ["Home", "Data Exploration", "Make Prediction", "Model Performance", "About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<hr style='border: 1px solid #3d80cd;'>", unsafe_allow_html=True)

# Dataset info in sidebar
st.sidebar.markdown("""
### 📋 Dataset Information
""")
st.sidebar.info(f"""
**Total Records:** {len(df):,}
**Features:** {df.shape[1]}
**Subscription Rate:** {df_numeric['y'].mean():.1%}
""")

# ==============================================================================
# PAGE 1: HOME
# ==============================================================================

if page == "Home":
    st.title("🏦 Bank Marketing - Term Deposit Prediction")
    
    st.markdown("""
    ---
    
    ### 🎯 Project Overview
    
    This machine learning dashboard predicts whether a client will subscribe to a term deposit 
    based on direct marketing campaign data from a Portuguese banking institution.
    
    ####  Dataset Details
    - **Source:** Portuguese Banking Institution
    - **Records:** 4,521 (10% sample of original dataset)
    - **Campaign Type:** Direct phone call marketing
    - **Objective:** Binary classification (Subscribe / Not Subscribe)
    
    ####  Key Features
    - **Personal Information:** Age, Job, Marital Status, Education, Credit Default
    - **Financial Data:** Account Balance, Housing Loan, Personal Loan
    - **Contact Data:** Contact Type, Duration, Campaign Details
    - **Previous Campaign:** Outcome, Days Since Last Contact, Previous Contacts
    
    ####  Machine Learning Model
    - **Algorithm:** Random Forest Classifier
    - **Training Data:** 80% of dataset
    - **Test Data:** 20% of dataset
    - **Performance Metric:** ROC-AUC Score
    
    ---
    
    ### 🚀 Navigation Guide
    
    - **Data Exploration:** Visualize distributions, correlations, and relationships in the dataset
    
    - **Make Prediction:** Predict subscription probability for a new client
    
    - **Model Performance:** Review model metrics, confusion matrix, and feature importance
    
    - **About:** Learn more about the project
    
    ---
    
    ### 💡 Key Insights
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "📞 Total Clients",
            f"{len(df):,}",
            "From Marketing Campaigns"
        )
    
    with col2:
        subscription_rate = df_numeric['y'].mean() * 100
        st.metric(
            "✅ Subscription Rate",
            f"{subscription_rate:.1f}%",
            f"{int(df_numeric['y'].sum())} Subscriptions"
        )
    
    with col3:
        avg_age = df['age'].mean()
        st.metric(
            "👤 Average Age",
            f"{avg_age:.1f}",
            "years"
        )
    
    st.markdown("---")


# ==============================================================================
# PAGE 2: DATA EXPLORATION
# ==============================================================================
elif page == "Data Exploration":
    st.title("Exploratory Data Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Distributions", "🔗 Correlations", "🎯 Target Analysis", "📊 Feature Relations"]
    )
    
    # TAB 1: DISTRIBUTIONS
    with tab1:
        st.subheader("Numerical Features Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature = st.selectbox(
                "Select Feature",
                df.select_dtypes(include=[np.number]).columns
            )
        
        with col2:
            n_bins = st.slider("Number of Bins", 10, 50, 30)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df[feature], bins=n_bins, color='#52b788', edgecolor='#2d5016', alpha=0.8)
        ax.axvline(df[feature].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[feature].mean():.2f}')
        ax.axvline(df[feature].median(), color='#2d5016', linestyle='--', linewidth=2, label=f'Median: {df[feature].median():.2f}')
        ax.set_title(f'Distribution of {feature.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{df[feature].mean():.2f}")
        with col2:
            st.metric("Median", f"{df[feature].median():.2f}")
        with col3:
            st.metric("Min", f"{df[feature].min():.2f}")
        with col4:
            st.metric("Max", f"{df[feature].max():.2f}")
    
    # TAB 2: CORRELATIONS
    with tab2:
        st.subheader("Feature Correlations")
        
        # Prepare data for correlation
        df_corr = df.copy()
        numerical_only = df_corr.select_dtypes(include=[np.number]).columns
        correlation_matrix = df_corr[numerical_only].corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax, linewidths=0.5)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    # TAB 3: TARGET ANALYSIS
    with tab3:
        st.subheader("Term Deposit Subscription Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Count plot
            fig, ax = plt.subplots(figsize=(8, 5))
            subscription_counts = df_numeric['y'].value_counts()
            colors_bar = ['#0e1c2e', '#3d80cd']
            bars = ax.bar(['No', 'Yes'], subscription_counts.values, color=colors_bar, edgecolor='black', linewidth=2)
            ax.set_title('Subscription Distribution', fontsize=14, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 5))
            colors_pie = ['#0e1c2e', '#3d80cd']
            wedges, texts, autotexts = ax.pie(
                subscription_counts.values, 
                labels=['No', 'Yes'],
                autopct='%1.1f%%',
                colors=colors_pie,
                startangle=90,
                textprops={'fontsize': 12, 'fontweight': 'bold'}
            )
            ax.set_title('Subscription Percentage', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        st.metric("Subscription Rate", f"{(df_numeric['y'].sum() / len(df) * 100):.2f}%")
    
    # TAB 4: FEATURE RELATIONS WITH TARGET
    with tab4:
        st.subheader("Feature Relationship with Subscription")
        
        feature_type = st.radio(
            "Select Feature Type",
            ["Numerical", "Categorical"],
            horizontal=True
        )
        
        if feature_type == "Numerical":
            feature = st.selectbox(
                "Select Numerical Feature",
                df.select_dtypes(include=[np.number]).columns
            )
            
            fig, ax = plt.subplots(figsize=(10, 5))
            df.boxplot(column=feature, by='y', ax=ax)
            ax.set_title(f'{feature.upper()} vs Subscription', fontsize=14, fontweight='bold')
            ax.set_xlabel('Subscription Status', fontsize=12)
            ax.set_ylabel(feature, fontsize=12)
            plt.suptitle('')
            st.pyplot(fig)
            plt.close()
        
        else:
            categorical_cols = df.select_dtypes(include='object').columns.tolist()
            if 'y' in categorical_cols:
                categorical_cols.remove('y')
            
            feature = st.selectbox("Select Categorical Feature", categorical_cols)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            crosstab = pd.crosstab(df[feature], df_numeric['y'], normalize='index') * 100
            crosstab.plot(kind='bar', ax=ax, color=['#0e1c2e', '#3d80cd'], edgecolor='black', linewidth=1.5)
            ax.set_title(f'{feature.upper()} vs Subscription (%)', fontsize=14, fontweight='bold')
            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.legend(title='Subscription', labels=['No', 'Yes'])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)
            plt.close()

# ==============================================================================
# PAGE 3: MAKE PREDICTION
# ==============================================================================

elif page == "Make Prediction":
    st.title("Predict Client Subscription")
    
    st.markdown("""
    Enter client information to predict the probability of term deposit subscription.
    """)
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("👤 Age", int(df['age'].min()), int(df['age'].max()), 30)
    
    with col2:
        balance = st.slider("💰 Account Balance ($)", int(df['balance'].min()), int(df['balance'].max()), 1000)
    
    with col3:
        duration = st.slider("⏱️ Call Duration (seconds)", int(df['duration'].min()), int(df['duration'].max()), 180)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        job = st.selectbox("💼 Job", df['job'].unique())
    
    with col5:
        marital = st.selectbox("💑 Marital Status", df['marital'].unique())
    
    with col6:
        education = st.selectbox("🎓 Education", df['education'].unique())
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        default = st.selectbox("⚠️ Credit Default", df['default'].unique())
    
    with col8:
        housing = st.selectbox("🏠 Housing Loan", df['housing'].unique())
    
    with col9:
        loan = st.selectbox("💳 Personal Loan", df['loan'].unique())
    
    col10, col11, col12 = st.columns(3)
    
    with col10:
        contact = st.selectbox("📞 Contact Type", df['contact'].unique())
    
    with col11:
        day = st.slider("📅 Day of Month", 1, 31, 15)
    
    with col12:
        month = st.selectbox("📆 Month", df['month'].unique())
    
    col13, col14, col15 = st.columns(3)
    
    with col13:
        campaign = st.slider("📊 Campaign Contacts", int(df['campaign'].min()), int(df['campaign'].max()), 1)
    
    with col14:
        pdays = st.slider("📈 Days Since Last Contact (-1 if never)", int(df['pdays'].min()), int(df['pdays'].max()), -1)
    
    with col15:
        previous = st.slider("🔄 Previous Contacts", int(df['previous'].min()), int(df['previous'].max()), 0)
    
    col16, col17 = st.columns(2)
    
    with col16:
        poutcome = st.selectbox("✅ Previous Outcome", df['poutcome'].unique())
    
    # Prepare input data
    input_data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }
    
    # Prediction button
    if st.button("Make Prediction", use_container_width=True):
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        for col in input_df.select_dtypes(include='object').columns:
            if col in le_dict:
                input_df[col] = le_dict[col].transform(input_df[col])
        
        # Make prediction
        probability = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            if prediction == 1:
                st.success(f"✅ **LIKELY TO SUBSCRIBE**")
                result_text = "**YES**"
            else:
                st.warning(f"❌ **UNLIKELY TO SUBSCRIBE**")
                result_text = "**NO**"
            
            st.metric("Predicted Subscription", result_text)
        
        with col_result2:
            st.metric(
                "Subscription Probability",
                f"{probability * 100:.1f}%"
            )
        
        # Probability gauge
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Create a horizontal bar
        colors_gauge = ['#0e1c2e', '#3d80cd']
        threshold = 0.5
        
        # Background bar
        ax.barh([0], [1], color='#e0e0e0', height=0.3)
        
        # Probability bar
        ax.barh([0], [probability], color=colors_gauge[int(prediction)], height=0.3)
        
        # Threshold line
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label='Decision Threshold (50%)')
        
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.5, 0.5])
        ax.set_xlabel('Subscription Probability', fontsize=12)
        ax.set_title('Subscription Probability Gauge', fontsize=14, fontweight='bold')
        ax.set_yticks([])
        
        # Add percentage labels
        ax.text(probability/2, 0, f'{probability*100:.1f}%', 
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        st.pyplot(fig)
        plt.close()

# ==============================================================================
# PAGE 4: MODEL PERFORMANCE
# ==============================================================================

elif page == "Model Performance":
    st.title("Model Performance Analysis")
    
    # Use preprocessed (encoded) data for model predictions
    df_test = df_model_data.sample(frac=0.2, random_state=42)
    
    tab1, tab2, tab3 = st.tabs([" Metrics", " Feature Importance", " Model Details"])
    
    with tab1:
        st.subheader("Model Performance Metrics")
        
        # Get predictions from the model
        X_test = df_test.drop('y', axis=1)
        y_test = df_test['y']
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}", "Overall correctness")
        with col2:
            st.metric("Precision", f"{precision:.4f}", "Positive accuracy")
        with col3:
            st.metric("Recall", f"{recall:.4f}", "True positive rate")
        with col4:
            st.metric("F1-Score", f"{f1:.4f}", "Harmonic mean")
        with col5:
            st.metric("ROC-AUC", f"{roc_auc:.4f}", "Model discrimination")
        
        st.markdown("---")
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, 
                        cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_xticklabels(['No', 'Yes'])
            ax.set_yticklabels(['No', 'Yes'])
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # ROC Curve
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', 
                   linewidth=2, color='#52b788')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
            ax.fill_between(fpr, tpr, alpha=0.2, color='#52b788')
            ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.legend(loc='lower right')
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Classification Report
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred, target_names=['No', 'Yes'], output_dict=True)
        
        report_df = pd.DataFrame(report).transpose().round(4)
        st.markdown("### Classification Report")
        st.dataframe(report_df, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Importance")
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            colors_imp = ['#52b788' if i == 0 else '#2d5016' for i in range(len(feature_importance.head(15)))]
            bars = ax.barh(range(len(feature_importance.head(15))), 
                           feature_importance.head(15)['Importance'].values,
                           color=colors_imp, edgecolor='black', linewidth=1.5)
            ax.set_yticks(range(len(feature_importance.head(15))))
            ax.set_yticklabels(feature_importance.head(15)['Feature'].values)
            ax.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
            ax.set_xlabel('Importance Score', fontsize=12)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2., f'{width:.4f}',
                       ha='left', va='center', fontsize=10, fontweight='bold')
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("### Top Features")
            for idx, row in feature_importance.head(10).iterrows():
                st.text(f"{row['Feature']}: {row['Importance']:.4f}")
    
    with tab3:
        st.subheader("Model Information")
        
        st.markdown("""
        ###  Model Details
        
        **Algorithm:** Random Forest Classifier
        
        **Purpose:** Binary classification to predict term deposit subscription
        
        **Training Data:** 80% of dataset
        
        **Test Data:** 20% of dataset
        
        **Features:** 16 input features
        
        **Training Samples:** ~3,617
        
        **Test Samples:** ~904
        
        ---
        
        ### 📊 Dataset Statistics
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Subscription Rate", f"{(df_numeric['y'].sum() / len(df) * 100):.2f}%")
        with col3:
            st.metric("Features Used", df.shape[1] - 1)

# ==============================================================================
# PAGE 5: ABOUT
# ==============================================================================

elif page == "About":
    st.title("About This Project")
    
    st.markdown("""
    ## 📚 Project Information
    
    ###  Objective
    Predict whether a Portuguese bank client will subscribe to a term deposit product
    based on direct marketing campaign data.
    
    ###  Dataset Overview
    - **Source:** Portuguese Banking Institution
    - **Total Records:** 4,521 (10% sample of bank-full.csv)
    - **Time Period:** Marketing campaigns data
    - **Campaign Method:** Direct phone calls
    - **Target Variable:** Subscription to term deposit (Binary: Yes/No)
    
    ###  Features Description
    
    **Personal Information:**
    - `age`: Client's age in years
    - `job`: Type of employment
    - `marital`: Marital status
    - `education`: Education level
    - `default`: Credit default status
    
    **Financial Information:**
    - `balance`: Account balance in euros
    - `housing`: Has housing loan (Yes/No)
    - `loan`: Has personal loan (Yes/No)
    
    **Campaign Contact Information:**
    - `contact`: Type of contact (cellular, telephone, unknown)
    - `day`: Day of month for last contact
    - `month`: Month of last contact
    - `duration`: Duration of last contact in seconds
    - `campaign`: Number of contacts during this campaign
    - `pdays`: Days since previous contact (-1 if never contacted)
    - `previous`: Number of previous contacts
    - `poutcome`: Outcome of previous campaign
    
    ###  Machine Learning Model
    
    **Algorithm:** Random Forest Classifier
    - Ensemble method with multiple decision trees
    - Excellent for classification tasks
    - Handles both numerical and categorical features
    - Provides feature importance rankings
    
    **Model Evaluation Metrics:**
    - **Accuracy:** Overall correctness of predictions
    - **Precision:** Ratio of correct positive predictions
    - **Recall:** Ratio of actual positives correctly identified
    - **F1-Score:** Harmonic mean of precision and recall
    - **ROC-AUC:** Area under the ROC curve (discrimination ability)
    
    ###  Dashboard Features
    
    **📊 Data Exploration:**
    - Interactive visualizations of feature distributions
    - Correlation matrix heatmap
    - Relationship analysis between features and target
    - Statistical summaries
    
    **🤖 Prediction Tool:**
    - User-friendly interface to input client information
    - Real-time predictions with probability scores
    - Visual probability gauge
    
    **📈 Model Performance:**
    - Comprehensive performance metrics
    - Confusion matrix visualization
    - ROC curve analysis
    - Feature importance rankings
    
    ###  Technology Stack
    - **Python 3.x:** Programming language
    - **Pandas:** Data manipulation and analysis
    - **NumPy:** Numerical computations
    - **Scikit-learn:** Machine learning algorithms
    - **Matplotlib & Seaborn:** Data visualization
    - **Streamlit:** Interactive web dashboard

    
    **GitHub Repository:**
    - All code is available on GitHub for easy access and collaboration
    - Includes documentation and usage instructions
    
    ###  Key Insights from EDA
    
    1. **Class Imbalance:** Only ~11% of clients subscribe to term deposits
    2. **Age Distribution:** Average client age is ~41 years
    3. **Balance Variability:** Account balances show significant variation
    4. **Duration Impact:** Longer call durations correlate with subscriptions
    5. **Campaign Contacts:** Number of contacts affects subscription likelihood
    
    ###  Important Notes
    
    - Model predictions are based on learned patterns from historical data
    - Actual subscription depends on many factors beyond those captured
    - Regular model retraining recommended with new data
    - Consider business context when interpreting predictions
    
    ###  Developer
    
    This project demonstrates end-to-end machine learning workflow including:
    - Data exploration and visualization
    - Feature engineering and preprocessing
    - Model training and evaluation
    - Interactive deployment with Streamlit
    
    ### 📧 Contact & Support
    
    For questions or suggestions regarding this project, please refer to the GitHub repository.
    
    ---
    
    **Last Updated:** 2026
    **Model Version:** 1.0
    """)

# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #2d5016; padding: 20px;'>
    <p><strong>🏦 Bank Marketing Prediction Dashboard</strong></p>
    <p>Powered by Machine Learning | Built with Streamlit</p>
    <p style='font-size: 12px; color: #52b788;'>© 2026 | Portuguese Bank Marketing Analysis</p>
</div>
""", unsafe_allow_html=True)
