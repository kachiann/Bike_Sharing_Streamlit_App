# 🚲 Bike Sharing Analysis and Prediction App

## Overview

This Streamlit app analyzes bike sharing data to uncover usage patterns and predict demand. It provides insights into bike rental trends and allows users to make predictions based on various factors.

## Features

- 📊 Data Exploration: View and explore the bike sharing dataset
- 📈 Usage Patterns: Analyze bike usage patterns based on various factors
- 🔮 Prediction: Predict bike rental demand using a Gradient Boosting model

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/kachiann/bike-sharing-analysis.git
   cd bike-sharing-analysis
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
## Usage

Run the Streamlit app:
```bash
streamlit run bike_sharing_app.py
```

The app will open in your default web browser.

## Data

The app uses the Bike Sharing Dataset from the [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) Machine Learning Repository. Ensure you have the `hour.csv` file in the same directory as the app.

## App Sections

### Home
- Overview of the app and its features

### Data Exploration
- View summary statistics of bike rentals
- Explore the dataset structure and basic information

### Usage Patterns
- Visualize hourly, daily, and monthly usage patterns
- Analyze the impact of weather on bike rentals

### Prediction
- Input various factors to predict bike rental demand
- View feature importance in the prediction model

## Model

The app uses a Gradient Boosting Regressor to predict bike rental demand. The model is trained on historical data and considers factors such as:

- Season
- Month
- Hour
- Holiday
- Weekday
- Working day
- Weather situation
- Temperature
- Humidity
- Wind speed

   
