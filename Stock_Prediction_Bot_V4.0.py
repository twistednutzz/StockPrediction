###### New CODE 2.0 ######

import subprocess
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import time


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


def install_required_libraries():
    # Install required libraries
    required_libraries = ['pandas', 'mpl_toolkits.mplot3d', 'pyqt5']  # Add any other required libraries
    for library in required_libraries:
        install(library)


def retrieve_historical_data(folder_path):
    # Get a list of all .txt files in the folder
    files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]

    # Read and concatenate the data from all the files
    data = pd.DataFrame()  # Create an empty DataFrame to store the data

    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            file_data = pd.read_csv(file_path)
            if not file_data.empty:
                data = pd.concat([data, file_data])
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file}")
        except pd.errors.ParserError:
            print(f"Error parsing file: {file}")

    return data


def preprocess_data(data):
    # Convert 'Date' column to datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Calculate the number of days since a reference date
    reference_date = data['Date'].min()
    data['Days'] = (data['Date'] - reference_date).dt.days

    # Drop the original 'Date' column
    data = data.drop('Date', axis=1)

    # Perform any other necessary data preprocessing steps here

    return data


def train_and_evaluate_model(X, y):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = RandomForestRegressor()
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    training_time = end_time - start_time

    return mse, mae, r2, model, training_time


def predict_stock_price(model, data, stock_ticker, prediction_date):
    # Retrieve the data for the specified stock ticker and prediction date
    stock_data = data[(data['Stock Ticker'] == stock_ticker) & (data['Date'] == prediction_date)]

    # Extract the relevant features for prediction
    X_pred = stock_data[['Days', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt']]

    # Make the prediction using the trained model
    predicted_price = model.predict(X_pred)

    return predicted_price[0]


def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    # Determine the number of columns and rows for subplots
    ncolumns = min(nGraphShown, len(df.columns))
    nrows = nGraphShown // nGraphPerRow

    # Create the subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncolumns, figsize=(20, 10))

    # Flatten the axes if necessary
    if nrows == 1:
        axes = [axes]

    # Iterate over each column and plot the distribution
    for i, column in enumerate(df.columns[:nGraphShown]):
        ax = axes[i // nGraphPerRow][i % nGraphPerRow]
        sns.histplot(df[column], ax=ax)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")

    # Remove any unused subplots
    for j in range(nGraphShown, len(df.columns)):
        ax = axes[j // nGraphPerRow][j % nGraphPerRow]
        ax.remove()

    # Adjust the layout and display the plots
    plt.tight_layout()
    plt.show()


def plotCorrelationMatrix(df, graphWidth):
    # Compute the correlation matrix
    corr = df.corr()

    # Create a figure and axis with the specified width
    fig, ax = plt.subplots(figsize=(graphWidth, graphWidth))

    # Generate a heatmap of the correlation matrix
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)

    # Set the title and display the plot
    ax.set_title("Correlation Matrix")
    plt.show()


def plotScatterMatrix(df, plotSize, textSize):
    # Create a scatter matrix of the dataframe
    scatter_matrix = pd.plotting.scatter_matrix(df, figsize=(plotSize, plotSize))

    # Adjust the text size of the plot
    for ax in scatter_matrix.ravel():
        ax.title.set_fontsize(textSize)
        ax.xaxis.label.set_fontsize(textSize)
        ax.yaxis.label.set_fontsize(textSize)
        ax.tick_params(axis='both', labelsize=textSize)

    # Display the plot
    plt.tight_layout()
    plt.show()


def main():
    # Step 1: Retrieve Historical Data
    folder_path = 'Path_To_.txt Files'  # Replace with the path to the folder containing the .txt files
    print(f"Retrieving historical data from: {folder_path}")
    data = retrieve_historical_data(folder_path)
    print("Historical data retrieved successfully.")

    # Step 2: Data Preprocessing
    print("Preprocessing data...")
    data_processed = preprocess_data(data)
    print("Data preprocessing completed.")

    # Step 3: Train and Evaluate the Model
    print("Training and evaluating the model...")
    X = data_processed[['Days', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt']]  # Replace with the relevant feature columns
    y = data_processed['Close']  # Replace with the column containing the target variable

    mse, mae, r2, model, training_time = train_and_evaluate_model(X, y)

    # Print the evaluation metrics and training time
    print("Evaluation metrics:")
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R^2 Score:", r2)
    print("Training Time:", training_time, "seconds")

    # Step 4: User Input Processing
    print("Please enter the following information:")
    stock_ticker = input("Enter the stock ticker: ")
    prediction_date = input("Enter the date of prediction (YYYY-MM-DD): ")

    # Step 5: Predict Stock Price
    print("Predicting stock price...")
    predicted_price = predict_stock_price(model, data_processed, stock_ticker, prediction_date)
    print("Stock price prediction completed.")

    # Step 6: Output Generation
    print("Generating output...")
    current_price = data_processed[
        (data_processed['Stock Ticker'] == stock_ticker) & (data_processed['Date'] == prediction_date)]['Close'].values[0]

    percentage_change = ((predicted_price - current_price) / current_price) * 100
    dollar_change = predicted_price - current_price

    print(f"Percentage Change: {percentage_change:.2f}%")
    print(f"Dollar Change: ${dollar_change:.2f}")
    print(f"Current Share Price: ${current_price:.2f}")
    print(f"Predicted Share Price on {prediction_date}: ${predicted_price:.2f}")

    # Additional Code Snippets
    print("Plotting distributions...")
    plotPerColumnDistribution(data_processed, nGraphShown=10, nGraphPerRow=5)
    print("Plotting correlation matrix...")
    plotCorrelationMatrix(data_processed, graphWidth=10)
    print("Plotting scatter matrix...")
    plotScatterMatrix(data_processed, plotSize=20, textSize=10)


# Call the main function
if __name__ == "__main__":
    install_required_libraries()
    main()
