# 🏏 End-to-End IPL Match Prediction

<p align="center">
  <img src="https://img.shields.io/badge/ML-Pipeline-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Deployment-FastAPI-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/MLOps-MLflow-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Sports-Analytics-red?style=for-the-badge" />
</p>

---

## 📌 Overview

This project is an **end-to-end Machine Learning pipeline** that predicts the outcome of IPL (Indian Premier League) cricket matches.  
It covers everything from **data preprocessing, feature engineering, model training, evaluation, experiment tracking, to deployment**.

Cricket is not just a game in India — it's an **emotion ❤️**. Predicting match results requires analyzing complex patterns in **team performance, player stats, and match conditions**.  
This project demonstrates how **ML techniques and MLOps practices** can be applied to **real-world sports analytics**.

---

## 🚀 Features

- ✅ Data preprocessing & feature engineering on IPL dataset
- ✅ ML models for match outcome prediction (Logistic Regression, Random Forest, XGBoost, etc.)
- ✅ Model evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC Curve
- ✅ Experiment tracking & reproducibility with **MLflow**
- ✅ Deployment-ready with **FastAPI**
- ✅ Well-structured modular pipeline (configs, utils, components)

---

## 📂 Project Structure

```
├── data/                # Raw & processed data
├── notebooks/           # Jupyter notebooks for EDA & experimentation
├── src/                 # Source code (pipeline, utils, training)
│   ├── components/
│   ├── pipeline/
│   ├── utils/
├── app.py               # FastAPI app for deployment
├── config.yaml          # Configuration file
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## 🔧 Tech Stack

- **Python 3.9+**
- **Pandas, NumPy, Scikit-learn**
- **RandomForest**
- **MLflow** for experiment tracking
- **FastAPI** for deployment
- **Heroku / AWS** (optional deployment)

---

## 📊 Results

## 📊 Model Performance

The model was evaluated on test data using multiple classification metrics:

- **Accuracy**: `72.00%`
- **Precision**: `72.02%`
- **Recall**: `72.00%`
- **F1-Score**: `71.98%`

These results indicate a well-balanced performance across all metrics, showing the model is reliable for predicting IPL match outcomes

---

## 🌍 Deployment

### ⚡ Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/suyashambule/End-to-End-IPL-prediction.git
   cd End-to-End-IPL-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   `

---

## 📘 Learning Outcomes

Through this project, I learned:

- Designing a complete **ML pipeline** (raw data → deployment)
- Applying **MLOps practices** like experiment tracking & model versioning
- Building & serving ML models with **FastAPI**
- Evaluating models rigorously in a **sports analytics context**

---

## 🎯 Future Improvements

- 🔹 Integrate **real-time match data APIs**
- 🔹 Use **deep learning models (LSTM, Transformers)** for sequence predictions
- 🔹 Build an interactive **Streamlit dashboard**
- 🔹 Deploy with **AWS/GCP + CI/CD automation**
- 🔹 Add explainability using **SHAP/LIME**

---

## 👨‍💻 Author

**Suyash Ambule**  
📧 [suyashambule1234@gmail.com](mailto:suyashambule1234@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/suyashambule/) | [GitHub](https://github.com/suyashambule)

---

⭐ If you found this project interesting, don’t forget to give it a **star** on GitHub!
