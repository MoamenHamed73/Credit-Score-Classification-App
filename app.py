import streamlit as st
import pandas as pd
import pickle
import numpy as np

# إعداد الصفحة
st.set_page_config(page_title="Credit Score Prediction", page_icon="💳", layout="centered")
st.title("💳 Credit Score Prediction App")
st.markdown("Predict your credit score and see the probability for each class.")
st.sidebar.markdown("Developed by **Moamen Mohamed**")

# ===== تحميل الملفات =====
with open("Credit_Mix_encoder.pkl", "rb") as f:
    le_credit = pickle.load(f)
with open("Payment_of_Min_Amount_encoder.pkl", "rb") as f:
    le_pay = pickle.load(f)
with open("Type_of_Loan_mlb.pkl", "rb") as f:
    mlb = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# ===== الأعمدة =====
numeric_features = [
    "Age","Annual_Income","Monthly_Inhand_Salary","Num_Bank_Accounts",
    "Num_Credit_Card","Interest_Rate","Num_of_Loan","Delay_from_due_date",
    "Num_of_Delayed_Payment","Changed_Credit_Limit","Num_Credit_Inquiries",
    "Outstanding_Debt","Credit_Utilization_Ratio","Total_EMI_per_month",
    "Amount_invested_monthly","Monthly_Balance","Credit_History_Age"
]

categorical_features = [
    "Occupation","Credit_Mix","Payment_Behaviour","Payment_of_Min_Amount","Type_of_Loan"
]

# ===== Default numeric values =====
default_values = {
    "Age": 35,
    "Annual_Income": 50000,
    "Monthly_Inhand_Salary": 4000,
    "Num_Bank_Accounts": 2,
    "Num_Credit_Card": 1,
    "Interest_Rate": 12,
    "Num_of_Loan": 1,
    "Delay_from_due_date": 0,
    "Num_of_Delayed_Payment": 0,
    "Changed_Credit_Limit": 0,
    "Num_Credit_Inquiries": 1,
    "Outstanding_Debt": 5000,
    "Credit_Utilization_Ratio": 0.3,
    "Total_EMI_per_month": 1500,
    "Amount_invested_monthly": 500,
    "Monthly_Balance": 3000,
    "Credit_History_Age": 5
}

# ===== Numeric inputs =====
numeric_inputs = {}
for feature in numeric_features:
    numeric_inputs[feature] = st.number_input(
        f"{feature}", 
        value=default_values.get(feature, 0.0)
    )

# ===== Categorical options =====
occupation_options = ['Salaried','Self_Employed','Others']
credit_mix_options = ['Good','Standard','Bad']
payment_behaviour_options = ['On_time','Delayed','Early']
payment_of_min_options = ['Yes','No']
type_of_loan_options = ['Car','Education','Home','Personal','Other']

categorical_inputs = {}
categorical_inputs['Occupation'] = st.selectbox("Occupation", occupation_options)
categorical_inputs['Credit_Mix'] = st.selectbox("Credit Mix", credit_mix_options)
categorical_inputs['Payment_Behaviour'] = st.selectbox("Payment Behaviour", payment_behaviour_options)
categorical_inputs['Payment_of_Min_Amount'] = st.selectbox("Payment of Minimum Amount", payment_of_min_options)
categorical_inputs['Type_of_Loan'] = st.selectbox("Type of Loan", type_of_loan_options)

# ===== تحويل الأعمدة النصية =====
categorical_inputs['Credit_Mix'] = le_credit.transform([categorical_inputs['Credit_Mix']])[0]
categorical_inputs['Payment_of_Min_Amount'] = le_pay.transform([categorical_inputs['Payment_of_Min_Amount']])[0]

# ===== Type_of_Loan One-Hot =====
loan_df = pd.DataFrame(
    mlb.transform([[categorical_inputs['Type_of_Loan']]]),
    columns=mlb.classes_
)
categorical_inputs.pop('Type_of_Loan')

# ===== تجهيز DataFrame =====
input_data = pd.DataFrame([numeric_inputs])
input_data = pd.concat([input_data, loan_df], axis=1)
for col in ['Occupation','Credit_Mix','Payment_Behaviour','Payment_of_Min_Amount']:
    input_data[col] = [categorical_inputs[col]]

# ===== التعامل مع الأعمدة العددية (RobustScaler) =====
scaler_features = scaler.feature_names_in_
for col in scaler_features:
    if col not in input_data.columns:
        input_data[col] = 0
input_data[scaler_features] = scaler.transform(input_data[scaler_features])

# ===== التعامل مع كل الأعمدة المطلوبة للموديل =====
required_cols = model.feature_names_in_
for col in required_cols:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[required_cols]

# ===== Prediction =====
if st.button("Predict Credit Score"):
    pred = model.predict(input_data)
    pred_proba = model.predict_proba(input_data)[0]
    
    # تحويل الرقم للتصنيف النهائي
    pred_class = target_encoder.inverse_transform(pred.reshape(-1,1))[0][0]
    
    st.success(f"✅ Predicted Credit Score: **{pred_class}**")
    
    # Probability chart
    proba_df = pd.DataFrame({
        'Class': target_encoder.categories_[0],
        'Probability': pred_proba
    })
    st.bar_chart(proba_df.set_index('Class'))
