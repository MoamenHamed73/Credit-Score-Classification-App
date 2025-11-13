# -Credit-Score-Classification-App
# 💳 Credit Score Prediction App

A **Streamlit web application** to predict credit scores (Poor / Standard / Good) based on customer financial and personal data.  
Built using **Random Forest** model with preprocessed data, encoding, and scaling.  

---

## 🚀 Features

- Predict credit score using customer details.
- Handles both **numerical** and **categorical** inputs.
- Shows **probability of each credit class** using bar chart.
- User-friendly interface with **default values** for numeric inputs.
- Robust handling of missing or extra features.
- Fully reproducible using Pickle files and Streamlit.

---

## 🛠️ Technologies Used

- Python 3.13
- Streamlit
- Pandas & NumPy
- scikit-learn
- Pickle (for saving model and encoders)
- Matplotlib / Seaborn (for plots if needed)

---

## 🗂️ Files in Repository

| File | Description |
|------|-------------|
| `streamlit_credit_app.py` | Main Streamlit app code |
| `rf_model.pkl` | Trained Random Forest model |
| `scaler.pkl` | Scaler for numeric features |
| `Credit_Mix_encoder.pkl` | LabelEncoder for Credit_Mix |
| `Payment_of_Min_Amount_encoder.pkl` | LabelEncoder for Payment_of_Min_Amount |
| `Type_of_Loan_mlb.pkl` | MultiLabelBinarizer for Type_of_Loan |
| `target_encoder.pkl` | OrdinalEncoder for target variable |
| `requirements.txt` | Python dependencies |

---

## 🎯 How to Run

1. Clone the repository:

```bash
git clone https://github.com/USERNAME/Credit-Score-Streamlit.git
cd Credit-Score-Streamlit
pip install -r requirements.txt
streamlit run streamlit_credit_app.py
Enter customer details and click Predict Credit Score.
🧩 Usage

Numeric fields have default values to make it easier for testing.

Categorical fields are dropdowns with predefined options.

The app displays the predicted credit score and a probability chart for each class.

👨‍💻 Developer

Moamen Mohamed

GitHub:https://github.com/MoamenHamed73
