from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import os
from sklearn.metrics import mean_squared_error, r2_score

# Set Matplotlib to non-interactive backend
matplotlib.use('Agg')

app = Flask(__name__)

# Helper: Mapping User Input to Tickers
ticker_map = {
    "btc": "BTC-USD",
    "bitcoin": "BTC-USD",
    "eth": "ETH-USD",
    "ethereum": "ETH-USD",
    "doge": "DOGE-USD",
    "dogecoin": "DOGE-USD",
    "xrp": "XRP-USD",
    "sol" : "SOL-USD",
}

# Helper: Mapping Tickers to Model files
model_map = {
    "BTC-USD": "models/BTC.keras",
    "ETH-USD": "models/ETH.keras",
    "DOGE-USD": "models/DOGE.keras",
    "XRP-USD": "models/XRP.keras",
    "SOL-USD": "models/SOL.keras",
}

# Helper: Convert Plot to HTML
def plot_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    buf.close()
    return f"data:image/png;base64,{data}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock = request.form.get("stock")
        no_of_days = int(request.form.get("no_of_days"))
        return redirect(url_for("predict", stock=stock, no_of_days=no_of_days))
    return render_template("index.html")

@app.route("/predict")
def predict():
    user_input = request.args.get("stock", "btc")
    no_of_days = int(request.args.get("no_of_days", 10))
    
    # Map user input to stock ticker
    stock = ticker_map.get(user_input.lower(), user_input.upper())
    
    # Load correct model
    model_path = model_map.get(stock)
    if model_path is None or not os.path.exists(model_path):
        return render_template("result.html", error="No model available for the selected stock.")

    model = load_model(model_path)

    # Fetch Stock Data
    end = datetime.now()
    start = datetime(end.year - 5, end.month, end.day)  # last 5 years
    stock_data = yf.download(stock, start, end)
    
    if stock_data.empty:
        return render_template("result.html", error="Invalid stock ticker or no data available.")

    # Data Preparation
    close_prices = stock_data[['Close']]
    splitting_len = int(len(close_prices) * 0.9)
    x_test = close_prices[splitting_len:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test)

    x_data = []
    y_data = []
    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Predictions
    predictions = model.predict(x_data)
    inv_predictions = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Plotting Data
    plotting_data = pd.DataFrame({
        'Original Test Data': inv_y_test.flatten(),
        'Predicted Test Data': inv_predictions.flatten()
    }, index=x_test.index[100:])

    # Plot 1: Closing Prices
    fig1 = plt.figure(figsize=(15,6))
    plt.plot(stock_data['Close'], label="Close Price", color="blue")
    plt.title(f"{stock} Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    original_plot = plot_to_html(fig1)

    # Plot 2: Original vs Predicted
    fig2 = plt.figure(figsize=(15,6))
    plt.plot(plotting_data['Original Test Data'], label="Original Test Data")
    plt.plot(plotting_data['Predicted Test Data'], label="Predicted Test Data", linestyle="--")
    plt.title(f"{stock} Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    predicted_plot = plot_to_html(fig2)

    # Plot 3: Future Predictions
    last_100 = close_prices.tail(100)
    last_100_scaled = scaler.transform(last_100)

    future_predictions = []
    last_100_scaled = last_100_scaled.reshape(1, -1, 1)

    for _ in range(no_of_days):
        next_day = model.predict(last_100_scaled)
        future_predictions.append(scaler.inverse_transform(next_day)[0][0])
        last_100_scaled = np.append(last_100_scaled[:, 1:, :], next_day.reshape(1, 1, 1), axis=1)

    future_predictions = np.array(future_predictions)

    fig3 = plt.figure(figsize=(15,6))
    plt.plot(range(1, no_of_days + 1), future_predictions, marker='o', color='purple', label="Predicted Future Prices")
    plt.title(f"{stock} Future Price Predictions ({no_of_days} days)")
    plt.xlabel("Days Ahead")
    plt.ylabel("Predicted Price (USD)")
    plt.grid(alpha=0.3)
    plt.legend()
    future_plot = plot_to_html(fig3)

    # Formatting to 4 decimal places
    future_predictions = [round(x, 4) for x in future_predictions]
    # Calculate Statistics
    mse = mean_squared_error(inv_y_test, inv_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(inv_y_test, inv_predictions)

    # Format nicely
    mse = round(mse, 2)
    rmse = round(rmse, 2)
    r2 = round(r2*100, 2)

    return render_template(
        "result.html",
        stock=stock,
        original_plot=original_plot,
        predicted_plot=predicted_plot,
        future_plot=future_plot,
        enumerate=enumerate,
        future_predictions=future_predictions,
        mse=mse,
        rmse=rmse,
        r2_score=r2
    )

if __name__ == "__main__":
    app.run(debug=True)
