# Loan Approval Prediction using Machine Learning

This project predicts whether a loan application will be approved or not based on applicant details such as income, loan amount, credit history, and other financial and demographic factors.  
It uses a **Random Forest Classifier** trained on historical loan application data and provides an interactive **Streamlit web application** for easy predictions.

---

## Features

- **Machine Learning Model** – Trained using Random Forest for high accuracy.
- **Interactive Web App** – Built with Streamlit for real-time predictions.
- **User-Friendly Input Form** – Simple interface for entering applicant details.
- **Field Information Table** – Explains how each input affects loan approval.
- **Deployed on Streamlit Cloud** – No setup required for users.

---

## Dataset

The dataset contains the following key features:

| Feature           | Description                                     | Unit / Values           |
| ----------------- | ----------------------------------------------- | ----------------------- |
| Gender            | Gender of applicant                             | Male, Female            |
| Married           | Marital status                                  | Yes, No                 |
| Dependents        | Number of dependents                            | 0, 1, 2, 3+             |
| Education         | Education level                                 | Graduate, Not Graduate  |
| Self_Employed     | Self-employment status                          | Yes, No                 |
| ApplicantIncome   | Monthly income of applicant                     | In Indian Rupees (₹)    |
| CoapplicantIncome | Monthly income of co-applicant                  | In Indian Rupees (₹)    |
| LoanAmount        | Loan amount (divided by 1000 for normalization) | ₹ Thousands             |
| Loan_Amount_Term  | Loan repayment term                             | Months                  |
| Credit_History    | Past credit repayment record                    | 1.0 = Good, 0.0 = Bad   |
| Property_Area     | Type of property location                       | Urban, Semiurban, Rural |

---

## How It Works

1. User enters loan application details in the Streamlit form.
2. Data is preprocessed to match the training format.
3. The trained model predicts whether the loan will be **Approved** or **Rejected**.
4. The result is displayed instantly.

---

## Tech Stack

- **Python** – Data processing & ML model
- **Pandas, NumPy** – Data handling
- **Scikit-learn** – Machine Learning
- **Streamlit** – Web app deployment
- **Joblib** – Model saving & loading

---

## Installation

```bash
# Clone the repository
git clone https://github.com/NAYAK-Shivam/loan-approval-prediction.git
cd loan-approval-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```
