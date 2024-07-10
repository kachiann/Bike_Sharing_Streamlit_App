# Data Visualization functions used in the eda.ipynb
# plotting.py

from visualization import plot_hourly_trends
from visualization import plot_hourly_wkd
from visualization import plot_temperature_vs_rentals
from visualization import plot_seasonal_rentals
from visualization import plot_weather_counts
from visualization import plot_hourly_diff_weather
from visualization import plot_daily_trends
from visualization import correlation_analysis



def visualize_data(data):
    """
    Visualize various aspects of bike rental data.

    Parameters:
    - data (pandas.DataFrame): The input dataframe containing bike rental data.

    Returns:
    - None
    """
    
    print("Starting visualize_data function...")
    # Define a dictionary to map numeric weekday codes to day names
    weekday_names = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    
    # Call each plotting function and add debug messages
    print("Plotting hourly trends...")
    plot_hourly_trends(data)
    
    print("Plotting hourly weekday analysis...")
    plot_hourly_wkd(data, weekday_names)
    
    print("Plotting temperature vs. rentals...")
    plot_temperature_vs_rentals(data)
    
    print("Plotting seasonal rentals...")
    plot_seasonal_rentals(data)
    
    print("Plotting weather counts...")
    plot_weather_counts(data)
    
    print("Plotting hourly difference by weather...")
    plot_hourly_diff_weather(data)

    print("Plotting hourly trends per day...")
    plot_daily_trends(data)
    
    print("Performing correlation analysis...")
    correlation_analysis(data)
    
    
    print("visualize_data function completed.") 

