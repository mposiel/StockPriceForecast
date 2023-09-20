import tensorflow as tf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
from dotenv import load_dotenv

# Define API key and symbol

load_dotenv()

symbol = 'IBM'
function = 'TIME_SERIES_DAILY'
apikey = os.getenv("API_KEY")

# Function to fetch data from Alpha Vantage
def fetch_data(symbol, function, apikey):
    response = requests.get(
        f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize=full&apikey={apikey}'
    )
    return response.json()

# Fetch data from Alpha Vantage
data = fetch_data(symbol, function, apikey)

# Extract daily closing prices
daily_prices = []

for record_date, record_data in data['Time Series (Daily)'].items():
    daily_prices.append(float(record_data["4. close"]))

# Reverse prices so they start from the earliest
daily_prices.reverse()

# Normalize data for model training
min_price = min(daily_prices)
max_price = max(daily_prices)

normalized_prices = [(price - min_price) / (max_price - min_price) for price in daily_prices]

# Prepare data for training
window_size = 50
x, y = [], []

for i in range(len(normalized_prices) - window_size):
    x.append(normalized_prices[i:i+window_size])
    y.append(normalized_prices[i+window_size])

x, y = np.array(x), np.array(y)

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(len(x) * split_ratio)
x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Shuffle the training data to prevent overfitting
x_train, y_train = shuffle(x_train, y_train)

# Build an LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=50, activation='relu', return_sequences=True, input_shape=(window_size, 1)))
model.add(tf.keras.layers.LSTM(units=50, activation='relu'))
model.add(tf.keras.layers.Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Save the model
model.save('IBM_model')

# Load the model
model = tf.keras.models.load_model('IBM_model')

# Make predictions on test data
predictions = model.predict(x_test)

# Inverse transform predictions and actual values to their original scale
predictions = [(price * (max_price - min_price)) + min_price for price in predictions]
y_test = [(price * (max_price - min_price)) + min_price for price in y_test]

# Calculate Mean Squared Error
squared_errors = [(pred - actual) ** 2 for pred, actual in zip(predictions, y_test)]
mse = sum(squared_errors) / len(squared_errors)
print(f"Mean Squared Error: {mse}")

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Fetch additional data for prediction
testdata = fetch_data(symbol, function, apikey)

# Extract daily closing prices for the test data
daily_prices = []

for record_date, record_data in testdata['Time Series (Daily)'].items():
    daily_prices.append(float(record_data["4. close"]))

# Use the latest 50 data points for prediction
daily_prices = daily_prices[:50]

# Normalize the new data
normal = [(price - min_price) / (max_price - min_price) for price in daily_prices]

# Reverse the order
normal.reverse()

# Convert to numpy array
xx = np.array([normal[0:window_size]])

# Make predictions for the new data
prediction = model.predict(xx)

# Inverse transform the prediction to the original scale
result = [(price * (max_price - min_price)) + min_price for price in prediction]

print(result)
