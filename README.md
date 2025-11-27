# 🌍 Earthquake Magnitude Predictor

An AI-powered earthquake magnitude prediction system using **Gaussian Process Regression** with uncertainty quantification. Features real-time USGS data, interactive maps, and local LLM integration via Ollama.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Features

| Feature                          | Description                                                 |
| -------------------------------- | ----------------------------------------------------------- |
| 🎯**Magnitude Prediction** | Gaussian Process Regression with uncertainty quantification |
| 📊**Confidence Intervals** | 95% CI showing possible magnitude range                     |
| 🗺️**Interactive Map**    | Click anywhere to get predictions, view recent earthquakes  |
| 🌐**Real-Time Data**       | Live USGS Earthquake API integration                        |
| 🤖**AI Summaries**         | Local LLM via Ollama generates plain-English analysis       |
| 🌙**Dark Theme**           | Modern dark UI design                                       |
| 📍**Location Search**      | Geocoding via OpenStreetMap Nominatim                       |

---

## 📸 Screenshot

<img src="app\assets\image.png" alt="Earthquake Predictor">

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/earthquake-magnitude-predictor.git
cd earthquake-magnitude-predictor

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Data & Train Model

```bash
# Fetch real earthquake data from USGS (last 30 days)
python scripts/run_etl.py --days 30 --min-mag 2.5

# Generate features
python scripts/run_features.py

# Train the model
python scripts/train_model.py --kernel composite --max-samples 2000
```

### 3. Run the App

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

---

## 🤖 Ollama Integration (Optional)

The app uses a local LLM to generate plain-English earthquake analysis.

### Setup Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a small model (recommended: llama3.2 ~2GB)
ollama pull llama3.2

# Or smaller alternatives
ollama pull phi3         # 2.3GB
ollama pull gemma2:2b    # 1.6GB

# Start Ollama server
ollama serve
```

The app will automatically detect Ollama and show:

- 🟢 **Connected** - AI summaries enabled
- 🔴 **Offline** - Falls back to template-based summaries

---

## 📁 Project Structure

```
earthquake-magnitude-predictor/
├── app/
│   ├── streamlit_app.py          # Main Streamlit application
│   ├── components/
│   │   ├── map_component.py      # Map visualization
│   │   └── visualizations.py     # Charts and gauges
│   └── assets/                   # Static files
│
├── src/
│   ├── data/
│   │   ├── api_client.py         # USGS API client
│   │   ├── database.py           # SQLite database handler
│   │   └── etl_pipeline.py       # ETL orchestration
│   │
│   ├── features/
│   │   ├── feature_pipeline.py   # Feature engineering orchestration
│   │   ├── temporal_features.py  # Time-based features
│   │   ├── spatial_features.py   # Location-based features
│   │   └── rolling_features.py   # Rolling window statistics
│   │
│   ├── model/
│   │   ├── train.py              # Model training
│   │   ├── predict.py            # Prediction with uncertainty
│   │   ├── evaluate.py           # Model evaluation metrics
│   │   ├── kernel.py             # GP kernel definitions
│   │   └── data_prep.py          # Data preparation
│   │
│   └── utils/
│       ├── helpers.py            # Utility functions
│       ├── logger.py             # Logging configuration
│       └── progress.py           # Progress bar utilities
│
├── scripts/
│   ├── run_etl.py                # Run ETL pipeline
│   ├── run_features.py           # Generate features
│   ├── train_model.py            # Train model
│   └── deploy.sh                 # Deployment script
│
├── models/
│   ├── gp_model.pkl              # Trained GP model
│   ├── scaler.pkl                # Feature scaler
│   ├── feature_config.json       # Feature configuration
│   └── evaluation/               # Evaluation results
│
├── data/
│   ├── earthquake.db             # SQLite database
│   ├── raw/                      # Raw data
│   └── processed/                # Processed features
│
├── .streamlit/
│   └── config.toml               # Streamlit theme (dark mode)
│
├── config.yaml                   # Project configuration
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

---

## 🔧 Configuration

### config.yaml

```yaml
data:
  usgs_api_url: "https://earthquake.usgs.gov/fdsnws/event/1/query"
  min_magnitude: 2.5
  days_back: 30
  
model:
  kernel: "composite"
  max_samples: 2000
  confidence_level: 0.95
  
app:
  theme: "dark"
  map_zoom: 3
```

### .streamlit/config.toml

```toml
[theme]
base = "dark"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
primaryColor = "#667eea"
```

---

## 📊 Model Details

### Gaussian Process Regression

The model uses a composite kernel combining:

| Kernel                      | Purpose                            |
| --------------------------- | ---------------------------------- |
| **RBF**               | Captures smooth spatial variations |
| **Matérn**           | Models rough spatial patterns      |
| **RationalQuadratic** | Handles multi-scale patterns       |

### Features (18 total)

| Category           | Features                                                 |
| ------------------ | -------------------------------------------------------- |
| **Spatial**  | Latitude, Longitude, Depth, Distance to plate boundary   |
| **Temporal** | Hour, Day of week, Month (cyclical encoding)             |
| **Rolling**  | 7-day count, 30-day count, Mean magnitude, Max magnitude |
| **Derived**  | Depth category, Region cluster, Seismic zone             |

### Evaluation Metrics

| Metric                | Description                         |
| --------------------- | ----------------------------------- |
| **R²**         | Variance explained (target: > 0.7)  |
| **MAE**         | Mean absolute error (target: < 0.5) |
| **RMSE**        | Root mean squared error             |
| **Calibration** | CI coverage accuracy                |

---

## 🌐 Data Source

**USGS Earthquake Hazards Program**
https://earthquake.usgs.gov/fdsnws/event/1/

Real-time earthquake data updated every minute. The ETL pipeline:

1. Fetches events from USGS API
2. Stores in SQLite database
3. Generates engineered features
4. Caches for fast predictions

---

## 📦 Dependencies

```
requests>=2.31.0          # API calls
pandas>=2.0.0             # Data handling
numpy>=1.24.0             # Numerical computing
scikit-learn>=1.3.0       # Machine learning
plotly>=5.18.0            # Visualizations
streamlit>=1.29.0         # Web app
streamlit-folium>=0.15.0  # Interactive maps
folium>=0.15.0            # Map rendering
pyyaml>=6.0               # Config files
tqdm>=4.66.0              # Progress bars
```

---

## 🚢 Deployment

### Streamlit Cloud

1. Push to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with:
   - Main file: `app/streamlit_app.py`
   - Python: 3.9+

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py"]
```

```bash
docker build -t earthquake-predictor .
docker run -p 8501:8501 earthquake-predictor
```

---

## ⚠️ Disclaimer

This tool provides **statistical estimates** based on historical patterns.

**It cannot predict:**

- When an earthquake will occur
- The exact magnitude of future earthquakes
- Whether an earthquake will happen at all

**Do not use for:**

- Emergency planning
- Evacuation decisions
- Safety-critical applications

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **USGS** - Earthquake data API
- **Anthropic** - AI assistance
- **Streamlit** - Web framework
- **Ollama** - Local LLM inference

---
