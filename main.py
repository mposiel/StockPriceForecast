import tensorflow as tf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Define your API key and symbol
# symbol = input("Stock name:")

symbol = 'IBM'
function = 'TIME_SERIES_DAILY'
apikey = 'T83EQT6ZCK3LV1X8'

# Fetch data from Alpha Vantage
data = requests.get(
    f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize=full&apikey={apikey}').json()

# store the prices in a daily_prices list

daily_prices = []

data_list = list(data)
for record in data[data_list[1]]:
    # print(f'Date: {record} Closing Price: {data[data_list[1]][record]["4. close"]}')
    daily_prices.append(float(data[data_list[1]][record]["4. close"]))

# print(daily_prices)

# reverse prices so it starts from the earliest
daily_prices.reverse()


# Normalize data for model training
min_price = min(daily_prices)
max_price = max(daily_prices)

print(f'max price: {max_price}, min price: {min_price}')

normalized_prices = []

for price in daily_prices:
    normalized_prices.append((float(price) - min_price) / (max_price - min_price))

# print(normalized_prices)

# Prepare data for training
window_size = 50  # Adjust this window size as needed
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


# shuffle the training data to prevent overfitting
x_train, y_train = shuffle(x_train, y_train)


print(f'total:{len(x)}')
print(f'len of train:{len(x_train)}, len of test:{len(x_test)}')



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


# Make predictions
predictions = model.predict(x_test)

# Inverse transform predictions and actual values to their original scale
predictions = [(price * (max_price - min_price)) + min_price for price in predictions]
y_test = [(price * (max_price - min_price)) + min_price for price in y_test]

# Calculate evaluation metrics without scikit-learn
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





