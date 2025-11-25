# NVIDIA Stock Price Prediction - Time Series Forecasting

A comprehensive machine learning project comparing LSTM, RNN, ARIMA, and Prophet models for predicting NVIDIA stock closing prices.

## 🎯 Project Objective

Predict NVIDIA stock closing prices using multiple time series forecasting models with a target MAPE < 5%.

## 📊 Dataset

- **Source**: Stooq (web scraping)
- **Ticker**: NVDA.US
- **Duration**: 3 months (August 2025 - November 2025)
- **Features**: 28 engineered features including technical indicators

## 🛠️ Technologies Used

- **Python 3.10.11**
- **Libraries**: 
  - Data Processing: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`, `plotly`
  - Machine Learning: `scikit-learn`, `tensorflow`, `keras`
  - Time Series: `statsmodels`, `prophet`
  - Web Scraping: `requests`, `BeautifulSoup`

## 📈 Models Implemented

1. **LSTM (Long Short-Term Memory)**
2. **RNN (Recurrent Neural Network)**
3. **ARIMA (AutoRegressive Integrated Moving Average)**
4. **Prophet (Meta/Facebook's Time Series Model)**

## 🔍 Feature Engineering

Created 28 features including:
- **Moving Averages**: SMA (5, 10 days), EMA (5, 10 days)
- **Technical Indicators**: RSI, MACD, Bollinger Bands
- **Volatility Measures**: 5-day and 10-day standard deviation
- **Price Features**: Daily return, high-low range, candle body
- **Candlestick Analysis**: Upper/lower shadows
- **Lag Features**: Previous day close and volume
- **Rolling Statistics**: 10-day max/min

## 📊 Results

### Model Performance (Test Set)

| Model | MAPE (%) | RMSE | MAE | MSE |
|-------|----------|------|-----|-----|
| **RNN** 🏆 | **2.29** | 4.33 | 4.16 | 18.72 |
| LSTM | 2.73 | 5.09 | 4.94 | 25.95 |
| Prophet | 3.35 | 6.30 | 6.04 | 39.71 |
| ARIMA | 4.98 | 9.29 | 9.01 | 86.30 |

✅ **All models achieved MAPE < 5% target!**

### Best Model: RNN
- Achieved **2.29% MAPE** on test set
- Lowest error across all metrics
- Best at capturing short-term price patterns

## 📁 Project Structure
```
├── nvidia_cleaned.csv                    # Cleaned raw data
├── nvidia_with_features.csv              # Data with engineered features
├── notebook_work.ipynb                   # Main Jupyter notebook
├── models/
│   ├── lstm_model.h5                     # Saved LSTM model
│   ├── rnn_model.h5                      # Saved RNN model
│   ├── arima_model.pkl                   # Saved ARIMA model
│   └── prophet_model.pkl                 # Saved Prophet model
├── visualizations/
│   ├── 1_price_with_ma.png              # Price with moving averages
│   ├── 2_volume.png                      # Trading volume
│   ├── 3_candlestick_features.png        # Candlestick analysis
│   ├── 4_technical_indicators.png        # RSI and MACD
│   ├── 5_bollinger_bands.png            # Bollinger bands
│   ├── 6_volatility.png                  # Volatility analysis
│   ├── 7_correlation_heatmap.png         # Feature correlations
│   ├── 8_price_distribution.png          # Price distribution
│   ├── model_comparison_metrics.png      # Model comparison
│   ├── train_vs_predicted_all_models.png # Training predictions
│   └── test_vs_predicted_all_models.png  # Test predictions
└── README.md                              # This file
```

## 📝 Methodology

1. **Data Collection**: Web scraping from Stooq
2. **Data Cleaning**: Handle missing values, sort chronologically
3. **Feature Engineering**: Create 28 technical and statistical features
4. **EDA**: Visualize trends, patterns, and correlations
5. **Train-Test Split**: 80-20 sequential split
6. **Model Training**: Train LSTM, RNN, ARIMA, and Prophet
7. **Evaluation**: Compare using MAPE, RMSE, MAE, MSE
8. **Visualization**: Create comprehensive comparison charts

## 📊 Key Insights

- Deep learning models (LSTM, RNN) outperformed traditional methods
- RNN showed best generalization on unseen data
- Technical indicators significantly improved prediction accuracy
- All models successfully captured the overall price trend

## 👨‍💻 Author

**Emaad Rehman**
- Data Analyst at Publicis Groupe
- LinkedIn: [[Emaad Ur Rehman](linkedin.com/in/emaad-ur-rehman)]
- Email: [emaadrehman3010@gmail.com]

## 🙏 Acknowledgments

- Stooq for providing stock data
- TensorFlow and Keras teams
- Prophet by Meta
- Statsmodels community
