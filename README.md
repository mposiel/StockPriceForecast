# Stock Price Prediction using LSTM

This project utilizes a Long Short-Term Memory (LSTM) neural network model to predict stock prices. It trains on historical daily closing price data from Alpha Vantage and makes predictions on test data, visualizing the results.

## Prerequisites

- Python 3.x
- Dependencies 
  - TensorFlow
  - Pandas
  - NumPy
  - Requests
  - Matplotlib
  - Scikit-Learn
  - `dotenv` for handling API keys and environment variables

Ensure you have an Alpha Vantage API key stored in a `.env` file.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/mposiel/stockpriceforecast.git
2. Install required Python packages.
3. Create a .env file in the project directory with your API key:
    ```bash
    API_KEY=your_api_key_here

## Usage
1. Run main.py to train the LSTM model and make predictions:

2. View the Mean Squared Error (MSE) and a price comparison plot.

3. To predict for a different stock symbol, modify the symbol variable and run the script again.

### Feel free to customize and improve this project for your needs.

To cover yourself legally and make it clear that the project is not intended for financial advice, you can add a disclaimer at the beginning of your README file. Here's an example of how you could word it:

## Disclaimer:

This project is for educational and research purposes only. The stock price predictions generated by the Long Short-Term Memory (LSTM) neural network model are based on historical data and should not be considered as financial advice. Investing in the stock market involves risk, and past performance is not indicative of future results. Always conduct your own research and consult with a qualified financial advisor before making any investment decisions.
