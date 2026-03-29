import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="HAR: Research-to-Product", page_icon="🏃", layout="wide")

# 2. LOAD ASSETS (Cached)
@st.cache_resource
def load_assets():
    model = joblib.load('activity_model.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return model, encoder

model, encoder = load_assets()

# 3. SIDEBAR: PROFESSIONAL CONTEXT & ACTIVITY MAP
with st.sidebar:
    st.header("🔬 Project Context")
    st.info("""
    **Target Audience:** Technical Recruiters & Researchers.
    
    **Problem:** Transforming high-frequency raw smartphone sensor noise into discrete human actions.
    
    **The Logic:** This app uses a **Logistic Regression** model trained on the UCI HAR Dataset, which translates 561 engineered features (Mean, Std, FFT, Signal Entropy) into 6 activity classes.
    """)
    
    st.divider()
    st.markdown("### 🏃 Activity Mapping")
    st.write({
        0: "Laying", 1: "Sitting", 2: "Standing", 
        3: "Walking", 4: "Walking Downstairs", 5: "Walking Upstairs"
    })
    
    st.divider()
    st.markdown("### 📚 Learn More")
    st.markdown("[Mathematical Theory of FFT](https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/)")
    st.markdown("[Signal Processing for Sensors](https://www.mdpi.com/2079-9292/11/3/322)")
    

# 4. MAIN UI: HEADER & SYSTEM ARCHITECTURE
st.title("🏃 Human Activity Recognition Dashboard")
st.subheader("A Multi-Class Classification Pipeline for Smartphone Sensor Data")

# Professional explanation of the Pipeline
st.markdown("""
---
### ⚙️ System Architecture
In a production environment, the data flows as follows:
**Raw Sensor (X,Y,Z)** ->
            **Sliding Window Filtering** ->
             **Fast Fourier Transform (FFT)** ->**561 Feature Extraction** ->**This Model**.
""")

# PLACEHOLDER FOR FLOW DIAGRAM (You can upload an image file named 'flow.png' to your folder)
# st.image("flow.png", caption="The End-to-End Signal Processing Pipeline")

st.divider()

# 5. DATA INGESTION: DEMO OR UPLOAD
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### 📤 1. Upload Your Own Data")
    uploaded_file = st.file_uploader("Upload CSV (561 Features)", type=["csv"])
    
with col2:
    st.markdown("#### 🧪 2. No Data? Use the Lab Sample")
    # Providing the sample download button
    try:
        with open("sample_data.csv", "rb") as f:
            st.download_button("Download Sample CSV", f, "sample_har_data.csv", "text/csv")
    except FileNotFoundError:
        st.warning("Sample CSV file not found in directory.")

# 6. INFERENCE LOGIC
data_to_predict = None

# If user uploads a file
if uploaded_file is not None:
    data_to_predict = pd.read_csv(uploaded_file)
# If user clicks a "Run Demo" button (Optional extra)
if st.button("🚀 Run Analysis on Loaded Data"):
    if uploaded_file is None:
        # Load the sample data automatically if no file is uploaded
        try:
            data_to_predict = pd.read_csv("sample_data.csv")
        except:
            st.error("Please upload a file or ensure sample_data.csv exists.")
    
    if data_to_predict is not None:
        st.write("### 🔍 Data Snapshot", data_to_predict.head(5))
        
        # Perform Prediction
        preds = model.predict(data_to_predict)
        labels = encoder.inverse_transform(preds)
        
        # Display Results
        st.divider()
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.success("🎯 Analysis Complete")
            results_df = pd.DataFrame({"Activity": labels})
            counts = results_df['Activity'].value_counts()
            st.table(counts)
            
        with res_col2:
            st.markdown("#### Activity Distribution")
            st.bar_chart(counts)

st.divider()
st.caption("Developed by [Pushpam](https://github.com/Kripu-star)")