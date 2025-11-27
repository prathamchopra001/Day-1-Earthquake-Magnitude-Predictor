
# 🌍 Earthquake Magnitude Predictor

A machine learning project that predicts earthquake magnitude with **uncertainty quantification** using  **Gaussian Process Regression** .

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

* 🔄 **Real-time data** from USGS Earthquake API
* 🎯 **Gaussian Process Regression** for probabilistic predictions
* 📊 **Confidence intervals** for uncertainty quantification
* 🗺️ **Interactive map** with earthquake visualization
* 📈 **Historical analysis** of seismic activity
* ☁️ **Deployable** to Streamlit Cloud

## 🖥️ Screenshots

The application provides:

* Interactive world map showing recent earthquakes
* Location selection for predictions
* Magnitude prediction with 95% confidence intervals
* Probability distribution visualization
* Regional seismic context

## 📁 Project Structure

```
earthquake-magnitude-predictor/
├── config.yaml              # Configuration settings
├── requirements.txt         # Python dependencies
├── setup.py                 # Quick setup script
│
├── data/
│   └── earthquake.db        # SQLite database
│
├── models/
│   ├── gp_model.pkl         # Trained GP model
│   ├── scaler.pkl           # Feature scaler
│   └── evaluation/          # Evaluation results
│
├── src/
│   ├── data/
│   │   ├── api_client.py    # USGS API client
│   │   ├── database.py      # SQLite operations
│   │   └── etl_pipeline.py  # ETL orchestration
│   │
│   ├── features/
│   │   ├── temporal_features.py
│   │   ├── spatial_features.py
│   │   ├── rolling_features.py
│   │   └── feature_pipeline.py
│   │
│   ├── model/
│   │   ├── kernel.py        # GP kernel design
│   │   ├── train.py         # Model training
│   │   ├── predict.py       # Inference
│   │   └── evaluate.py      # Evaluation metrics
│   │
│   └── utils/
│       ├── logger.py
│       └── helpers.py
│
├── app/
│   ├── streamlit_app.py     # Main Streamlit app
│   └── components/          # UI components
│
├── scripts/
│   ├── run_etl.py
│   ├── run_features.py
│   ├── train_model.py
│   └── deploy.sh
│
└── notebooks/               # Jupyter notebooks
```

## 🚀 Quick Start

### Option 1: Automated Setup

```bash
# Clone repository
git clone https://github.com/yourusername/earthquake-magnitude-predictor.git
cd earthquake-magnitude-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run automated setup (creates demo data & trains model)
python setup.py --demo

# Launch app
streamlit run app/streamlit_app.py
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Fetch earthquake data
python scripts/run_etl.py --days 30 --min-mag 2.5

# Step 2: Compute features
python scripts/run_features.py

# Step 3: Train model
python scripts/train_model.py --kernel simple

# Step 4: Run app
streamlit run app/streamlit_app.py
```

## 📖 Usage

### Command Line Scripts

```bash
# ETL: Fetch earthquake data
python scripts/run_etl.py --days 30 --min-mag 2.5 --limit 5000

# Features: Compute rolling statistics
python scripts/run_features.py --min-mag 3.0

# Train: Train GP model
python scripts/train_model.py --kernel simple --max-samples 2000

# Available kernels: simple, composite, advanced
```

### Python API

```python
# Make predictions
from src.model.predict import EarthquakePredictor

predictor = EarthquakePredictor()
predictor.load()

result = predictor.predict(
    latitude=-17.9,
    longitude=-178.4,
    depth=100.0
)

print(f"Magnitude: {result['magnitude']:.2f} ± {result['std']:.2f}")
print(f"95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
```

## 🧠 Model Details

### Gaussian Process Regression

The model uses GPR for probabilistic magnitude prediction:

* **Why GPR?** Inherent uncertainty quantification without additional calibration
* **Kernel** : RBF + WhiteKernel (configurable)
* **Training** : Marginal likelihood optimization

### Features (18 total)

| Category    | Features                                |
| ----------- | --------------------------------------- |
| Temporal    | Hour (cyclical), Day of week (cyclical) |
| Spatial     | Latitude, Longitude, Depth (normalized) |
| Rolling 7d  | Event count, Mean/Max/Std magnitude     |
| Rolling 30d | Event count, Mean/Max/Std magnitude     |
| Derived     | Days since M4+ event, Seismic density   |
| Quality     | Station count, Gap, Distance, RMS       |

### Evaluation Metrics

| Metric      | Description                  |
| ----------- | ---------------------------- |
| MAE         | Mean Absolute Error          |
| RMSE        | Root Mean Square Error       |
| R²         | Coefficient of determination |
| Calibration | CI coverage accuracy         |

## ☁️ Deployment

### Deploy to Streamlit Cloud

1. **Push to GitHub** :

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Streamlit Cloud** :

* Go to [share.streamlit.io](https://share.streamlit.io/)
* Click "New app"
* Select your repository
* Set main file: `app/streamlit_app.py`
* Click "Deploy!"

### Important Notes

* Ensure `models/gp_model.pkl` and `models/scaler.pkl` are committed
* The app will use `setup.py --demo` if database is missing
* Free tier has resource limits (~1GB RAM)

## 📊 Data Source

[USGS Earthquake Hazards Program](https://earthquake.usgs.gov/fdsnws/event/1/)

* Global earthquake data
* Updated in near real-time
* Free public API

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
# Data settings
data:
  days_to_fetch: 30
  min_magnitude_for_training: 2.5

# Feature engineering
features:
  rolling_windows: [7, 30]
  spatial_radius_km: 200

# Model settings
model:
  test_size: 0.2
  kernel:
    rbf_length_scale: 1.0
```

## 📝 License

MIT License - see [LICENSE](https://claude.ai/chat/LICENSE) for details.

## 🙏 Acknowledgments

* USGS Earthquake Hazards Program for data
* Scikit-learn for GP implementation
* Streamlit for the web framework
