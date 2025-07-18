# california-housing-predictor
California Housing Predictor is a machine learning pipeline that predicts house prices using income, location, and proximity features. It includes stratified sampling, preprocessing with scikit-learn, Random Forest regression, and saved models for seamless inference from input to output CSV.
<br> 
 🏡 Housing Price Predictor
<br> 
A complete machine learning pipeline built to predict housing prices in California using the classic housing dataset. This project covers preprocessing, stratified sampling, pipeline creation, model training, and prediction storage — all in a clean, modular way.
<br> 
                                       ---
<br> 
## 🚀 Features
<br> 
- Stratified Train-Test Splitting<br> 
- Numerical & Categorical Preprocessing Pipelines<br> 
- Model Training using Random Forest Regressor<br> 
- Model and Pipeline Persistence with `joblib`<br> 
- Inference system that predicts and stores results in `output.csv`<br> 
<br> 
                                      ---


---

 🧠 How it Works

1. **First Run**:
    - Reads `housing.csv`
    - Stratified splitting based on `median_income`
    - Trains a Random Forest model on the training data
    - Saves `model.pkl` and `pipeline.pkl`

2. **Subsequent Runs**:
    - Loads saved model and pipeline
    - Reads `input.csv` for new data
    - Performs prediction and writes to `output.csv`

---

 🛠️ Requirements

- Python 3.7+
- `scikit-learn`, `pandas`, `numpy`, `joblib`

Install with:
```bash
pip install -r requirements.txt


