# Google Sheet Stock Prediction Bot - Final Analysis plus price increase

import subprocess
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

# Install required libraries
required_libraries = ['pandas', 'scikit-learn', 'pandas_ta', 'gspread', 'oauth2client']
for library in required_libraries:
    install(library)

# Step 1: Google Sheet Integration
def read_google_sheet(sheet_name):
    # Set up credentials and access the Google Sheet
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('C:/Users/jilan/OneDrive/Desktop/Stock Prediction/stock-prediction-386402-75a0ee997a02.json', scope)
    client = gspread.authorize(credentials)
    
    try:
        sheet = client.open(sheet_name).sheet1
    except gspread.SpreadsheetNotFound:
        print(f"Google Sheet '{sheet_name}' not found.")
        return None

    # Read the data from the Google Sheet
    data = pd.DataFrame(sheet.get_all_records())

    return data

# Step 2: Data Collection
def retrieve_historical_data(sheet_name):
    data = read_google_sheet(sheet_name)

    if data is None:
        return None

    return data

# Step 3: Feature Engineering and Technical Indicators
def calculate_technical_indicators(data):
    # Calculate technical indicators
    data['sma'] = data['Close'].rolling(window=10).mean()
    data['ema'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['rsi'] = ta.rsi(data['Close'], length=14)
    
    # Calculate Bollinger Bands
    rolling_mean = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['upper_band'] = rolling_mean + 2 * rolling_std
    data['lower_band'] = rolling_mean - 2 * rolling_std

    return data


# Step 4: Data Preprocessing
def preprocess_data(data):
    # Handle missing data
    data = data.dropna()

    if data.empty:
        raise ValueError("No valid data available for preprocessing.")

    # Exclude the 'Ticker' column from scaling if it exists
    if 'Ticker' in data.columns:
        data_to_scale = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        # Scale/Normalize the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_to_scale)
        # Create a new DataFrame with the scaled features
        data_processed = pd.DataFrame(scaled_data, columns=data_to_scale.columns)
        # Combine the scaled features with the remaining columns
        data_processed = pd.concat([data[['Date', 'Ticker']], data_processed], axis=1)
    else:
        data_to_scale = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        # Scale/Normalize the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_to_scale)
        # Create a new DataFrame
        data_processed = pd.DataFrame(scaled_data, columns=data_to_scale.columns)

    return data_processed

def perform_feature_selection(data):
    # Split the data into features (X) and target variable (y)
    X = data.iloc[:, 2:] # Exclude 'Date' and 'Ticker' columns
    y = data['Close']
    
    # Perform feature selection using Random Forest
    model = RandomForestRegressor(random_state=42)
    rfe = RFE(estimator=model, n_features_to_select=5)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_].tolist()
    X = X[selected_features]
    
    return X, y

def train_and_evaluate_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, mae, r2, y_pred[-1]

def main():
    # Step 1: Google Sheet Integration
    sheet_name = "Stock Sheet" # Replace with the actual sheet name
    data = retrieve_historical_data(sheet_name)
    if data is None:
        print("Data retrieval failed. Exiting the program.")
        return
    
    # Step 2: Data Preprocessing
    data_processed = preprocess_data(data)

    # Step 3: Perform Feature Selection
    X, y = perform_feature_selection(data_processed)

    # Step 4: Train and Evaluate the Model
    mse, mae, r2, last_predicted_price = train_and_evaluate_model(X, y)

    # Print the evaluation metrics
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R^2 Score:", r2)

    # Predict if the stock will increase or decrease
    if last_predicted_price > data_processed['Close'].iloc[-1]:
        print("The stock price is predicted to increase.")
        price_difference = last_predicted_price - data_processed['Close'].iloc[-1]
        print("Price Increase:", price_difference)
    else:
        print("The stock price is predicted to decrease.")
        price_difference = data_processed['Close'].iloc[-1] - last_predicted_price
        print("Price Decrease:", price_difference)

    # Determine whether to buy or sell the stock
    if last_predicted_price > data_processed['Close'].iloc[-1]:
        print("Recommendation: Buy the stock.")
    else:
        print("Recommendation: Sell the stock.")

    # Calculate the number of days till the predicted price change
    today = datetime.now().date()
    predicted_date = today + timedelta(days=10)  # Replace with the appropriate number of days
    print("Predicted date of price change:", predicted_date)

if __name__ == "__main__":
    main()
