import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hourly_trends(data):
    # Extract hour and rental count data
    hourly_data = data.groupby('hour')['count'].mean()
    # Plot the hourly trends
    plt.figure(figsize=(10, 6))
    hourly_data.plot(kind='line', color='blue', marker='o')
    plt.title('Avg. Hourly Trends in Bike Rentals')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Total Rental Count')
    plt.xticks(range(24))
    plt.grid(True)
    plt.show()


def plot_hourly_wkd(data, weekday_names):
    # Define a dictionary to map numeric weekday codes to day names
    weekday_names = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    plt.figure(figsize=(14, 6))
    sns.pointplot(data=data, x='hour', y='count', hue='weekday')
    plt.title('Hourly Bike Rental Analysis: Weekdays vs. Weekends', fontsize=16)
    plt.xlabel('Hour', fontsize=14)
    plt.ylabel('Average Count of Rentals', fontsize=14)
    plt.legend(title='Day', title_fontsize='14', fontsize='12', loc='upper right')

    # Set the legend labels using the weekday names dictionary
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles, [weekday_names[int(label)] for label in labels])
    plt.grid(True)
    plt.show()


def plot_temperature_vs_rentals(data):
    plt.figure(figsize=(8, 6))
    plt.scatter(data['temperature'], data['count'], color='green', alpha=0.5)
    plt.title('Bike Rentals vs. Temperature')
    plt.xlabel('Temperature (C)')
    plt.ylabel('Number of Rentals')
    plt.show()

def plot_seasonal_rentals(data):
    # Season data conversion
    season_num = {1:'winter', 2:'spring', 3:'summer', 4:'fall'}
    data['season'] = data['season'].map(season_num)
    season_counts = data.groupby('season')['count'].mean()
    plt.figure(figsize=(8, 6))
    season_counts.plot(kind='bar', color='orange', alpha=0.7)
    plt.title('Average Bike Rentals by Season')
    plt.xlabel('Season')
    plt.ylabel('Average Number of Rentals')
    plt.xticks(rotation=0)
    plt.show()    

def plot_weather_counts(data):
    # Weather data conversion
    weather_num = {1:'Clear',2:'Mist',3:'Light Snow',4:'Heavy Rain'}
    data['weather_situation'] = data['weather_situation'].map(weather_num)
    weather_counts = data.groupby('weather_situation')['count'].mean()
    plt.figure(figsize=(8, 6))
    weather_counts.sort_values(ascending=False).plot(kind='bar', color='orange', alpha=0.7) 
    plt.title('Average Bike Rentals by Weather')
    plt.xlabel('Weather')
    plt.ylabel('Average Number of Rentals')
    plt.xticks(rotation=0)
    plt.show()  

def plot_hourly_diff_weather(data):
    plt.figure(figsize=(14, 6))
    sns.pointplot(data=data, x='hour', y='count', hue='weather_situation')
    plt.title('Hourly Bike Rental Analysis: Different Weathers', fontsize=16)
    plt.xlabel('Hour', fontsize=14)
    plt.ylabel('Average Count of Rentals', fontsize=14)
    plt.legend(title='Weather', title_fontsize='14', fontsize='12', loc='upper right')
    plt.grid(True)
    plt.show()

def plot_daily_trends(data):
    # Convert 'datetime' column to datetime datatype
    data['datetime'] = pd.to_datetime(data['dteday'])
    # Extract the date from the datetime column
    data['date'] = data['datetime'].dt.date
    # Aggregate the data by date and count the number of trips
    daily_trips = data.groupby('date')['count'].mean()
    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(daily_trips.index, daily_trips.values, linestyle='-')
    plt.title('Number of Trips per Day')
    plt.xlabel('Date')
    plt.ylabel('Avg. Count')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def correlation_analysis(data):
    # Exclude non-numeric columns from the correlation matrix computation
    numeric_data = data.select_dtypes(include=['number'])
    # Compute the correlation matrix
    correlation_matrix = numeric_data.corr()
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()