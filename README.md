# Fake-News-Detection

# 📰 Fake News Detection Web App

This project aims to **detect fake news articles using machine learning** techniques. The model analyzes the content of a given news text and classifies it as either *Fake* or *Real*. It provides a simple and interactive web interface built with **Streamlit** for real-time predictions.

---

## 🚀 Features
- Detects **Fake vs Real** news instantly.  
- Simple and user-friendly **Streamlit web app**.  
- Uses **Wikipedia integration** to cross-check facts (optional).  
- Fast and accurate prediction using a trained ML model.

---

## 🧠 Technologies Used
- **Python** (Core language)  
- **Pandas, NumPy** – Data preprocessing  
- **Scikit-learn** – Machine learning model  
- **Joblib** – Model serialization  
- **Streamlit** – Frontend web app  
- **Wikipedia API** – Optional real-time verification  

---

## 📊 Dataset
- CSV dataset containing two main columns:  
  - **text** → The news content  
  - **label** → Fake or Real  
- Dataset used for model training and testing.

---

## ⚙️ How to Run
```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the Streamlit app
streamlit run app.py
