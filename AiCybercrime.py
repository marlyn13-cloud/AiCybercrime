import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from transformers import pipeline

# Page config
st.set_page_config(page_title="Cybercrime Detection Dashboard", layout="wide")
st.title("🧠 AI Cybercrime Detection Dashboard")

# Load models
@st.cache_resource
def load_summarizer():
    return pipeline("summarization")

@st.cache_resource
def load_image_captioner():
    return pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

summarizer = load_summarizer()
image_captioner = load_image_captioner()

# Helper: format classification report to readable text
def format_classification_summary(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    lines = []

    for label in [str(c) for c in np.unique(y_test)]:
        class_report = report.get(label, {})
        if class_report:
            lines.append(
                f"Class {label}: Precision {class_report['precision']:.2f}, "
                f"Recall {class_report['recall']:.2f}, F1-score {class_report['f1-score']:.2f}."
            )

    accuracy = report.get("accuracy", None)
    if accuracy is not None:
        lines.append(f"Overall Accuracy: {accuracy * 100:.2f}%")

    return " ".join(lines)

# Helper: Detect outcome column
def detect_outcome_column(df):
    candidates = ['outcome', 'label', 'target', 'class', 'result', 'status']
    for col in df.columns:
        if col.lower() in candidates:
            return col
    return None

# Tabs
tab1, = st.tabs(["📊 Data & Model"])

# ===  Data & Model Training ===
with tab1:
    uploaded_csv = st.file_uploader("Upload your cybersecurity CSV file", type=["csv"])
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv, encoding='windows-1252')
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Try to detect the outcome column
        outcome_col = detect_outcome_column(df)

        if not outcome_col:
            st.warning("Could not automatically detect the outcome column.")
            outcome_col = st.selectbox("Please select the outcome column:", df.columns)

        st.success(f"Using outcome column: `{outcome_col}`")

        # Preprocessing
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        df[numeric_cols] = SimpleImputer(strategy='mean').fit_transform(df[numeric_cols])
        df[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[categorical_cols])

        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col])

        df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])

        # PCA
        st.subheader("PCA Explained Variance")
        X = df.drop(outcome_col, axis=1)
        y = df[outcome_col]
        pca = PCA(n_components=10)
        X_pca = pca.fit_transform(X)

        fig, ax = plt.subplots()
        ax.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
        ax.set_xlabel("Number of Principal Components")
        ax.set_ylabel("Cumulative Explained Variance")
        ax.set_title("PCA Variance Explained")
        st.pyplot(fig)

        threshold = 0.90
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(cumulative_variance >= threshold) + 1
        explained = cumulative_variance[num_components - 1] * 100
        st.success(f"The first {num_components} principal components explain {explained:.2f}% of the dataset variance.")

        # Train model
        st.subheader("🧪 Train Random Forest Classifier")
        test_size = st.slider("Test Set Size (%)", 10, 50, 20, 5)
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=test_size/100, random_state=42)

        with st.spinner("Training model..."):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        st.metric("✅ Accuracy", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

        # Classification Report
        st.subheader("Classification Report")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report_dict).transpose())

        # AI Summary
        st.subheader("🧠 AI Summary of Model Performance")
        readable_report = format_classification_summary(y_test, y_pred)
        try:
            with st.spinner("Summarizing performance..."):
                summary = summarizer(readable_report, max_length=80, min_length=30, do_sample=False)
                st.info(summary[0]['summary_text'])
        except Exception as e:
            st.warning("AI summarizer failed. Showing manual summary instead.")
            st.text(readable_report)
