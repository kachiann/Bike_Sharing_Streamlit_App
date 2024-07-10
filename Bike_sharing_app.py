import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set page config
st.set_page_config(page_title="Bike Sharing Analysis", page_icon="🚲", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox {
        color: #1E90FF;
    }
    .stMetric {
        background-color: #e6f3ff;
        border-radius: 5px;
        padding: 10px;
    }
    h1 {
        color: #2E8B57;
    }
    h2 {
        color: #4682B4;
    }
</style>
""", unsafe_allow_html=True)

# Load the bike sharing dataset
@st.cache_data
def load_data():
    data = pd.read_csv('hour.csv')
    return data

data = load_data()

# Map season numbers to season names
season_num = {1: 'winter', 2: 'spring', 3: 'summer', 4: 'fall'}
data['season'] = data['season'].map(season_num)

# Map weekday numbers to weekday names
weekday_names = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
data['weekday'] = data['weekday'].map(weekday_names)

# Calculate summary statistics
total_casual = data['casual'].sum()
total_registered = data['registered'].sum()
total_users = data['cnt'].sum()

# Streamlit app
def main():
    st.title('🚲 Bike Sharing Analysis')
    
    menu = ["Home", "Data Exploration", "Usage Patterns", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.header("Welcome to the Bike Sharing Analysis App")
        st.write("""
        This app analyzes bike sharing data to uncover usage patterns and predict demand.
        Use the menu on the left to navigate between pages:
        - 📊 Data Exploration: View and explore the dataset
        - 📈 Usage Patterns: Analyze bike usage patterns
        - 🔮 Prediction: Predict bike rental demand
        """)
        
    elif choice == "Data Exploration":
        st.subheader("Daily Users")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Casual Users", f"{total_casual:,}", delta="Daily Users")
        with col2:
            st.metric("Total Registered Users", f"{total_registered:,}", delta="Frequent Users")
        with col3:
            st.metric("Total Users", f"{total_users:,}", delta="All Users")

        st.header("📊 Bike Sharing Dataset")
        st.dataframe(data.head())
        
        st.subheader("Dataset Information")
        st.write(f"Shape of the dataset: {data.shape}")
        st.write(data.describe())
        
    elif choice == "Usage Patterns":
        st.header("📈 Bike Usage Patterns")
        
        st.subheader("Hourly Usage Pattern")
        hourly_usage = data.groupby('hr')['cnt'].mean()
        st.line_chart(hourly_usage)
        
        st.subheader("Daily Usage Pattern")
        daily_usage = data.groupby('weekday')['cnt'].mean()
        st.bar_chart(daily_usage)
        
        st.subheader("Monthly Usage Pattern")
        monthly_usage = data.groupby('mnth')['cnt'].mean()
        st.line_chart(monthly_usage)
        
        st.subheader("Weather Impact on Usage")
        weather_impact = data.groupby('weathersit')['cnt'].mean()
        st.bar_chart(weather_impact)
        
    elif choice == "Prediction":
        st.header("🔮 Predict Bike Rental Demand")
    
        # Prepare data for prediction
        feature_cols = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
        X = data[feature_cols]
        X = pd.get_dummies(X, columns=['season', 'weekday'], drop_first=True)
        y = data['cnt']
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
        y_pred = model.predict(X_test)
    
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-squared Score: {r2:.2f}")
    
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
        st.pyplot(fig)
    
        st.subheader("Make a Prediction")
        # Create input fields for user to enter values
        user_input = {}
    
        date_input_method = st.radio("Choose input method for date:", ('Season', 'Month'))
    
        if date_input_method == 'Season':
            user_input['season'] = st.selectbox("Select season", list(season_num.values()))
            user_input['mnth'] = 6  # default to middle of the year
        else:
            user_input['mnth'] = st.number_input("Enter month (1-12)", min_value=1, max_value=12, value=6)
            # Infer season from month
            month_to_season = {1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring', 
                           6: 'summer', 7: 'summer', 8: 'summer', 9: 'fall', 10: 'fall', 11: 'fall', 12: 'winter'}
            user_input['season'] = month_to_season[user_input['mnth']]
    
        user_input['hr'] = st.number_input("Enter hour (0-23)", min_value=0, max_value=23, value=12)
        user_input['holiday'] = st.number_input("Enter holiday (0: No, 1: Yes)", min_value=0, max_value=1, value=0)
        user_input['weekday'] = st.selectbox("Select weekday", list(weekday_names.values()))
        user_input['workingday'] = st.number_input("Enter working day (0: No, 1: Yes)", min_value=0, max_value=1, value=1)
        user_input['weathersit'] = st.number_input("Enter weather situation (1: Clear, 2: Mist, 3: Light Snow/Rain, 4: Heavy Snow/Rain)", min_value=1, max_value=4, value=1)
        # Temperature input in Celsius
        temp_celsius = st.number_input("Enter temperature (°C)", min_value=-20.0, max_value=50.0, value=20.0)
    
        # Convert Celsius to normalized temperature
        # Assuming the original data range was -8°C to 39°C (you may need to adjust these based on your actual data)
        temp_min, temp_max = -8, 39
        user_input['temp'] = (temp_celsius - temp_min) / (temp_max - temp_min)
    
        # For simplicity, we'll use the same normalization for feels-like temperature
        atemp_celsius = st.number_input("Enter feels-like temperature (°C)", min_value=-20.0, max_value=50.0, value=20.0)
        user_input['atemp'] = (atemp_celsius - temp_min) / (temp_max - temp_min)
        #user_input['temp'] = st.number_input("Enter temperature (normalized)", min_value=0.0, max_value=1.0, value=0.5)
        #user_input['atemp'] = st.number_input("Enter feeling temperature (normalized)", min_value=0.0, max_value=1.0, value=0.5)
        user_input['hum'] = st.number_input("Enter humidity (normalized)", min_value=0.0, max_value=1.0, value=0.5)
        user_input['windspeed'] = st.number_input("Enter wind speed (normalized)", min_value=0.0, max_value=1.0, value=0.2)
    
        # Convert user input to DataFrame
        user_df = pd.DataFrame(user_input, index=[0])
    
        # Ensure correct order and dummy variable creation
        user_df = pd.get_dummies(user_df, columns=['season', 'weekday'], drop_first=True)
    
        # Ensure all columns from training data are present in user input
        for col in X.columns:
            if col not in user_df.columns:
                user_df[col] = 0
    
        # Ensure columns are in the same order as training data
        user_df = user_df[X.columns]
    
        # Make prediction based on user input
        prediction = model.predict(user_df)
    
        st.success(f"Predicted number of bike rentals: {prediction[0]:.0f}")

if __name__ == '__main__':
    main()