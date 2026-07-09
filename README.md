<<<<<<< HEAD
# Air Quality ML Pipeline - Egypt

Complete machine learning pipeline for predicting Air Quality Index (AQI), identifying responsible pollutant gases, and classifying health status across Egyptian governorates.

## Features

- **3 Predictive Models**:
  - AQI Regression Model (XGBoost)
  - Gas Classification Model (XGBoost)
  - Health Status Classification Model (XGBoost)

- **Complete ML Pipeline**:
  - Exploratory Data Analysis (EDA)
  - Data Cleaning & Validation
  - Feature Engineering (10 derived features)
  - Model Training with 70/20/10 splits
  - Comprehensive Evaluation Metrics
  - Model Deployment & Monitoring
  - Real-time Dashboard

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Prepare your data**:
   - Place `clean_air_dataset.csv` in the root directory
   - Place `Egypt Governorates for API.xlsx` in the root directory

## Usage

### Option 1: Run Complete Pipeline

Run the entire pipeline from EDA to deployment:

```bash
python src/main_pipeline.py
```

Skip EDA if already done:
```bash
python src/main_pipeline.py --skip-eda
```

Skip data cleaning if already done:
```bash
python src/main_pipeline.py --skip-cleaning
```

### Option 2: Run Individual Components

#### 1. Exploratory Data Analysis
```python
from src.eda_module import EDAModule

eda = EDAModule("clean_air_dataset.csv", "eda_outputs")
eda.save_all_outputs()
```

#### 2. Data Cleaning
```python
from src.data_cleaner import DataCleaner

cleaner = DataCleaner("clean_air_dataset.csv")
cleaned_df = cleaner.clean_all("cleaned_air_dataset.csv")
```

#### 3. Train Models
```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer(random_state=42)
data_splits = trainer.create_data_splits(X, y_aqi, y_gas, y_health)

trainer.train_aqi_model(data_splits['X_train'], data_splits['y_aqi_train'],
                       data_splits['X_val'], data_splits['y_aqi_val'])
trainer.train_gas_model(data_splits['X_train'], data_splits['y_gas_train'],
                       data_splits['X_val'], data_splits['y_gas_val'])
trainer.train_health_model(data_splits['X_train'], data_splits['y_health_train'],
                          data_splits['X_val'], data_splits['y_health_val'])
```

#### 4. Evaluate Models
```python
from src.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(models, encoders, "evaluation_outputs")
results = evaluator.evaluate_all_splits(data_splits)
evaluator.save_evaluation_results()
```

#### 5. Deploy Models
```python
from src.deployment_manager import DeploymentManager

deployment = DeploymentManager(models, encoders, scaler, "models")
deployment.deploy_all(evaluation_results, feature_list)
```

### Run Dashboard

Launch the interactive web dashboard:

```bash
streamlit run src/dashboard.py
```

The dashboard will:
- Fetch live data from OpenWeatherMap API
- Make real-time predictions for all governorates
- Display AQI, responsible gas, and health status
- Show pollutant concentrations and weather conditions
- Auto-refresh every 5 minutes (optional)

### Make Predictions

Use the inference script for batch predictions:

```bash
python models/inference.py
```

Or use it programmatically:

```python
from models.inference import AirQualityPredictor

predictor = AirQualityPredictor("models")
predictions = predictor.predict(your_data)
```

## Project Structure

```
.
├── src/
│   ├── governorate_mapper.py    # Governorate name standardization
│   ├── eda_module.py             # Exploratory data analysis
│   ├── data_cleaner.py           # Data cleaning & validation
│   ├── preprocessor.py           # Scaling & encoding
│   ├── feature_engineer.py       # Feature engineering
│   ├── model_trainer.py          # Model training
│   ├── model_evaluator.py        # Model evaluation
│   ├── deployment_manager.py     # Model deployment
│   ├── monitor.py                # Performance monitoring
│   ├── dashboard.py              # Web dashboard
│   └── main_pipeline.py          # Complete pipeline
│
├── models/                       # Trained models & artifacts
│   ├── aqi_model.pkl
│   ├── gas_model.pkl
│   ├── health_model.pkl
│   ├── gov_encoder.pkl
│   ├── gas_encoder.pkl
│   ├── health_encoder.pkl
│   ├── scaler.pkl
│   ├── model_metadata.json
│   └── inference.py
│
├── eda_outputs/                  # EDA visualizations & reports
├── evaluation_outputs/           # Model evaluation results
├── monitoring_logs/              # Performance monitoring logs
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Model Performance Targets

- **AQI Model**: MAE < 10.0, R² > 0.85
- **Gas Model**: Accuracy > 80%
- **Health Model**: Accuracy > 85%

## Engineered Features

The pipeline creates 10 derived features:

1. **pm_ratio**: PM2.5 / (PM10 + 1) - Fine to coarse particle ratio
2. **no2_o3**: NO2 * O3 - Photochemical smog indicator
3. **hour_sin**: sin(2π * hour / 24) - Cyclical hour encoding
4. **hour_cos**: cos(2π * hour / 24) - Cyclical hour encoding
5. **month_sin**: sin(2π * month / 12) - Cyclical month encoding
6. **month_cos**: cos(2π * month / 12) - Cyclical month encoding
7. **total_pollutants**: Sum of all pollutant concentrations
8. **temp_humidity**: Temperature * Humidity interaction
9. **wind_vector_x**: Wind speed * cos(wind direction)
10. **wind_vector_y**: Wind speed * sin(wind direction)

## Monitoring

The pipeline includes performance monitoring:

```python
from src.monitor import Monitor

monitor = Monitor("monitoring_logs")
monitor.set_baseline_metrics(evaluation_results)

# Log predictions
monitor.log_prediction(input_data, predictions)

# Check for performance degradation
monitor.check_performance_degradation(current_metrics)

# Detect data drift
monitor.detect_data_drift(production_data, training_data)

# Generate reports
report = monitor.generate_performance_report(period='30d')
```

## API Configuration

The dashboard uses OpenWeatherMap API. Update the API key in `src/dashboard.py`:

```python
API_KEY = "your_api_key_here"
```

Get a free API key at: https://openweathermap.org/api

## Troubleshooting

### Issue: Models showing 0% accuracy

**Solution**: This was the original problem! The new pipeline fixes this by:
- Proper data preprocessing and encoding
- Correct feature engineering
- Proper train/validation/test splits
- Handling class imbalance

### Issue: Import errors

**Solution**: Make sure you're running from the project root directory and all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: API errors in dashboard

**Solution**: 
- Check your API key is valid
- Ensure you have internet connection
- API has rate limits (60 calls/minute for free tier)

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please open an issue in the repository.

