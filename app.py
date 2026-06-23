import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(
    page_title="Healthcare Fraud Detection",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Healthcare Fraud Detection Dashboard")
st.markdown("Machine Learning based Healthcare Claim Fraud Analysis")

# Sidebar
st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully!")

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Dataset Information
    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    # Statistics
    st.subheader("Statistical Summary")
    st.write(df.describe())

    # Missing Values
    st.subheader("Missing Values")

    missing = df.isnull().sum()

    if missing.sum() > 0:
        st.bar_chart(missing)
    else:
        st.success("No Missing Values Found")

    # Fraud Distribution
    target_cols = [
        col for col in df.columns
        if 'fraud' in col.lower()
        or 'target' in col.lower()
        or 'label' in col.lower()
    ]

    if len(target_cols) > 0:

        target = target_cols[0]

        st.subheader("Fraud Distribution")

        fig, ax = plt.subplots()

        df[target].value_counts().plot(
            kind='bar',
            ax=ax
        )

        plt.title("Fraud vs Non-Fraud Claims")

        st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] > 1:

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            numeric_df.corr(),
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig)

    # Feature Explorer
    st.subheader("Feature Analysis")

    numeric_columns = numeric_df.columns.tolist()

    if numeric_columns:

        feature = st.selectbox(
            "Select Feature",
            numeric_columns
        )

        fig, ax = plt.subplots()

        sns.histplot(
            df[feature],
            kde=True,
            ax=ax
        )

        st.pyplot(fig)

else:
    st.info("Upload a healthcare fraud dataset to begin analysis.")