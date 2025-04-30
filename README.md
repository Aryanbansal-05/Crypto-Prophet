# 💹 Crypto Prophet

**Crypto Prophet** is a deep learning-based cryptocurrency price predictor built using Long Short-Term Memory (LSTM) neural networks. It provides future price predictions for cryptocurrencies by analyzing historical time-series data.

## 🚀 Features

- 📈 Predicts future prices for cryptocurrencies like Bitcoin (BTC), Ethereum(ETH), Dogecoin (DOGE) etc.
- 🧠 Built with LSTM — a type of Recurrent Neural Network (RNN) suitable for time-series forecasting
- 🗃️ Trained on real historical market data (e.g., from Yahoo Finance)
- 📊 Visualizes actual vs. predicted prices
- 💾 Saves and loads trained models for reuse

## 🛠️ Tech Stack

- Python 🐍
- TensorFlow / Keras
- NumPy & Pandas
- Matplotlib / Seaborn
- Scikit-learn
- Yahoo Finance API (e.g., `yfinance`)

## 📁 Project Structure


## 🧪 How It Works

1. **Data Collection**: Historical price data is collected using the `yfinance` library.
2. **Preprocessing**: Data is normalized and shaped into sequences suitable for LSTM input.
3. **Model Training**: An LSTM model is trained on the processed dataset to learn temporal patterns.
4. **Prediction**: The model predicts future prices, which are compared with actual values for evaluation.

## ⚙️ Usage

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/crypto-prophet.git
cd crypto-prophet
pip install -r requirements.txt
python train.py
python predict.py
