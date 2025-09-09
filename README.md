# ⚡ SmartGrid-BigData-Analytics

This project applies **machine learning** and **data analytics** to **forecast power load in a smart grid** environment. It uses historical time-series data to train predictive models, enabling energy providers to anticipate demand more accurately and optimize grid performance.

---

## 📊 Project Features
- ✅ Train/test split with scikit-learn
- ✅ Regression model training 
- ✅ Error metrics: MAE, MSE, RMSE, R² Score
- ✅ Visualizations for actual vs predicted load

---

## 🗃️ Dataset

The dataset is a CSV file (`smart_grid_load.csv`) located in the `data/` folder, and includes:

- `Datetime`: timestamp (hourly)
- `Load`: electrical load value (float)

> Make sure the file has no missing `Datetime` values and is sorted chronologically.

---

## 🛠️ Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/SmartGrid-BigData-Analytics.git
cd SmartGrid-BigData-Analytics

# 2. Set up virtual environment
python -m venv .venv
.venv\Scripts\activate   # For Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the forecasting script
python src/load_forecasting.py



