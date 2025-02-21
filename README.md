# 📝 Sentiment Analysis using Machine Learning & Transformers

This project performs **sentiment analysis** on customer feedback using **Machine Learning (ML) models** and a **Transformer-based model (DistilBERT)**. The trained models are deployed using **Streamlit** and **Gradio** for easy user interaction.

---

## 🚀 Project Overview
This project involves:
- **Data Preprocessing**: Cleaning and tokenizing text data.
- **Traditional ML Models**: Logistic Regression, Random Forest, XGBoost.
- **Transformer Model**: Fine-tuned **DistilBERT** for improved accuracy.
- **Model Evaluation**: Comparing ML models with transformer-based models.
- **Deployment**:
  - **Streamlit Web App** → [Live Demo](https://sentiment-analysis-app-amdari.streamlit.app/)


---

## 📂 Project Structure
📁 Sentiment-Analysis
│── app.py                 # Streamlit Web App
│── requirements.txt       # Dependencies
│── README.md              # Documentation
│── 📂 models/             # Saved Models (pkl, transformers)

---

## 📊 Dataset
- The dataset contains **3.5 million** customer reviews labeled as **Positive (1)** or **Negative (0)**.
- It is split into **train (80%)**, **validation (10%)**, and **test (10%)**.

---

## 🔧 Installation

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/boiBASH/sentiment-analysis-app.git
