# StockPrediction
Stock Prediction Bot: This repository contains code for a Google Sheet-based stock prediction bot. It retrieves data, performs feature engineering, preprocessing, and uses machine learning (Random Forest). Leverages Python libraries for analysis, with a well-documented codebase. Ideal for stock analysis and ML enthusiasts. V1 has the added functionality of telling you by how much the price will increase or decrease. 

#V2

Added the updated and more advanced V2.

### Installation

1. Clone this repository to your local machine:

git clone https://github.com/your-username/stock-prediction-bot.git

2. Navigate to the project directory:

cd stock-prediction-bot

3. Install the required libraries:

pip install -r requirements.txt


### Usage

1. Ensure you have a Google Sheet with historical stock data. Note the name of the sheet.

2. Open the `stock_prediction_bot.py` file and replace the following placeholders with your own information:

- `YOUR_GOOGLE_SHEET_NAME`
- `PATH_TO_SERVICE_ACCOUNT_KEY`

3. Run the script:

python stock_prediction_bot.py


4. The bot will retrieve the data, preprocess it, select relevant features, train the model, and provide predictions and recommendations based on the last day's data.

## Contributing

Contributions to this stock prediction bot are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
