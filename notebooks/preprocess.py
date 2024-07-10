# Data preprocessing function used in the eda.ipynb

def preprocess_hourly_dataset(hourly_dataset):
    """
    Preprocess the hourly dataset by renaming columns, adding target variable, and checking for missing values.

    Parameters:
    - hourly_dataset (pandas.DataFrame): The input hourly dataset containing features and targets.

    Returns:
    - pandas.DataFrame: The preprocessed hourly dataset.
    - pandas.Series: A summary of missing values in the dataset.
    """
    # Accessing the features and target
    features = hourly_dataset.data.features
    targets = hourly_dataset.data.targets
    
    # Renaming column names
    hourly_df = features.rename(columns={'weathersit':'weather_situation',
                                           'yr':'year',
                                           'mnth':'month',
                                           'hr':'hour',
                                           'hum':'humidity',
                                           'temp': 'temperature'})
    
    # Adding 'count' column from targets
    hourly_df['count'] = targets['cnt']
    
    # Checking for missing values
    missing_values_summary = hourly_df.isnull().sum()
    
    return hourly_df, missing_values_summary

