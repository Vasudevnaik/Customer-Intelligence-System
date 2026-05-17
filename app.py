import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(layout="wide")
st.title("Customer Intelligence Tracking System")

# --- Load Models and Scalers/Encoders ---
@st.cache_resource
def load_models():
    # Clustering
    kmeans_model = joblib.load('k_means_model.pkl')
    kmeans_scaler = joblib.load('kmeans_scaler.pkl')
    cluster_names = {
        0: "Occasional Shopper",
        1: "Loyal Customer",
        2: "Low Value Customer",
        3: "VIP Customer"
    }
    rfm_data_for_plot = pd.read_pickle('rfm_df_for_plot.pkl')

    # Churn Prediction
    lr_model = joblib.load('logistic_regression.pkl')
    churn_scaler = joblib.load('churn_scaler.pkl')
    churn_label_encoders = joblib.load('churn_label_encoders.pkl')

    # CLV Prediction
    linr_model = joblib.load('linear_regression.pkl')
    clv_scaler = joblib.load('clv_scaler.pkl')

    return kmeans_model, kmeans_scaler, cluster_names, rfm_data_for_plot, \
           lr_model, churn_scaler, churn_label_encoders, \
           linr_model, clv_scaler

kmeans_model, kmeans_scaler, cluster_names, rfm_data_for_plot, \
lr_model, churn_scaler, churn_label_encoders, \
linr_model, clv_scaler = load_models()

# --- Customer Clustering Section ---
st.header("1. Customer Clustering (RFM)")
st.markdown("Enter customer's Recency, Frequency, and Monetary values to determine their segment.")

with st.expander("Input for Clustering"):
    col1, col2, col3 = st.columns(3)
    recency_c = col1.number_input("Recency (Days since last purchase)", min_value=0, value=100)
    frequency_c = col2.number_input("Frequency (Number of purchases)", min_value=1, value=5)
    monetary_c = col3.number_input("Monetary (Total spend)", min_value=0.0, value=200.0)

    if st.button("Predict Customer Segment"):
        # Ensure consistent transformations
        monetary_log_c = np.log1p(monetary_c)
        frequency_log_c = np.log1p(frequency_c)

        cluster_input = pd.DataFrame([{
            'Recency': recency_c,
            'Frequency': frequency_c,
            'Monetary': monetary_c,
            'Monetary_log': monetary_log_c,
            'Frequency_log': frequency_log_c
        }])

        # Scale the input features using the saved scaler
        scaled_cluster_input = kmeans_scaler.transform(cluster_input)

        # Predict cluster
        cluster_id = kmeans_model.predict(scaled_cluster_input)[0]
        segment = cluster_names.get(cluster_id, "Unknown Segment")

        st.success(f"The customer belongs to the: **{segment}** segment (Cluster {cluster_id})")

        # Plotting the clusters and the new customer
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=rfm_data_for_plot,
            x='Monetary_log',
            y='Frequency_log',
            hue='Segment',
            palette='viridis',
            ax=ax
        )
        ax.scatter(
            monetary_log_c,
            frequency_log_c,
            color='red',
            marker='X',
            s=200,
            label=f'New Customer ({segment})',
            edgecolor='black',
            zorder=5
        )
        ax.set_title("Customer Segments (RFM) with New Customer")
        ax.set_xlabel("Monetary (Log Transformed)")
        ax.set_ylabel("Frequency (Log Transformed)")
        ax.legend()
        st.pyplot(fig)


# --- Churn Prediction Section ---
st.header("2. Churn Prediction")
st.markdown("Predict whether a customer is likely to churn based on their telecom service details.")

with st.expander("Input for Churn Prediction"):
    col1, col2, col3 = st.columns(3)
    tenure = col1.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
    monthly_charges = col2.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = col3.number_input("Total Charges", min_value=0.0, value=600.0)

    col4, col5, col6 = st.columns(3)
    contract = col4.selectbox("Contract Type", churn_label_encoders['Contract'].classes_)
    payment_method = col5.selectbox("Payment Method", churn_label_encoders['PaymentMethod'].classes_)
    internet_service = col6.selectbox("Internet Service", churn_label_encoders['InternetService'].classes_)

    col7, col8 = st.columns(2)
    tech_support = col7.selectbox("Tech Support", churn_label_encoders['TechSupport'].classes_)
    online_security = col8.selectbox("Online Security", churn_label_encoders['OnlineSecurity'].classes_)

    if st.button("Predict Churn"):
        # Create DataFrame from inputs
        churn_input = pd.DataFrame([{
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Contract': churn_label_encoders['Contract'].transform([contract])[0],
            'PaymentMethod': churn_label_encoders['PaymentMethod'].transform([payment_method])[0],
            'InternetService': churn_label_encoders['InternetService'].transform([internet_service])[0],
            'TechSupport': churn_label_encoders['TechSupport'].transform([tech_support])[0],
            'OnlineSecurity': churn_label_encoders['OnlineSecurity'].transform([online_security])[0]
        }])

        # Scale the input features
        scaled_churn_input = churn_scaler.transform(churn_input)

        # Predict churn
        churn_prediction = lr_model.predict(scaled_churn_input)[0]
        churn_prob = lr_model.predict_proba(scaled_churn_input)[0][1]

        result = "Churn" if churn_prediction == 1 else "No Churn"
        st.success(f"The customer is predicted to: **{result}** (Churn Probability: {churn_prob:.2f})")


# --- Customer Lifetime Value (CLV) Prediction Section ---
st.header("3. Customer Lifetime Value (CLV) Prediction")
st.markdown("Predict the potential future revenue a customer will bring to the business.")

with st.expander("Input for CLV Prediction"):
    col1, col2, col3 = st.columns(3)
    recency_clv = col1.number_input("Recency (Days)", min_value=0, value=100)
    frequency_clv = col2.number_input("Frequency (Purchases)", min_value=1, value=5)
    avg_order_value = col3.number_input("Average Order Value", min_value=0.0, value=150.0)

    col4, col5, col6 = st.columns(3)
    order_std = col4.number_input("Order Value Standard Deviation", min_value=0.0, value=50.0)
    lifespan = col5.number_input("Lifespan (Days as customer)", min_value=0, value=300)
    purchase_rate = col6.number_input("Purchase Rate (Frequency / (Recency + 1))", min_value=0.0, value=0.05, format="%.4f")

    if st.button("Predict CLV"):
        clv_input = pd.DataFrame([{
            'Recency': recency_clv,
            'Frequency': frequency_clv,
            'AvgOrderValue': avg_order_value,
            'OrderStd': order_std,
            'Lifespan': lifespan,
            'PurchaseRate': purchase_rate
        }])

        # Scale the input features
        scaled_clv_input = clv_scaler.transform(clv_input)

        # Predict Monetary_log and inverse transform
        monetary_log_prediction = linr_model.predict(scaled_clv_input)[0]
        predicted_clv = np.expm1(monetary_log_prediction) # Inverse of log1p

        st.success(f"Predicted Customer Lifetime Value (CLV): **${predicted_clv:.2f}**")

st.markdown("--- ")
st.info("Note: This is a demo. Predictions are based on the trained models and input values.")
