# 🍷 Wine Quality MLOps Pipeline

A modular Machine Learning pipeline to predict wine quality using chemical properties. Built with production-grade MLOps practices.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```bash
   python src/components/model_trainer.py
   ```

3. **Run the app:**
   ```bash
   python app.py
   ```
   Visit `http://localhost:5000` to start predicting.

## 📂 Project Structure
- `src/components/`: Data ingestion, transformation, and model training.
- `src/pipeline/`: Prediction pipeline logic.
- `app.py`: Flask web application.
- `artifacts/`: Saved models and preprocessors.

## 🛠️ Tech Stack
Python, Scikit-Learn, XGBoost, Flask, Pandas.

---
**Author:** Rudresh Trivedi  
**License:** MIT
