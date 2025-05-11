# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df = pd.read_csv('sales_data.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)

# Visualize Sales Trends
plt.figure(figsize=(12,6))
plt.plot(df['Sales'], label='Sales')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Optional: Moving Average
df['Sales_MA'] = df['Sales'].rolling(window=7).mean()
df[['Sales', 'Sales_MA']].plot(figsize=(12,6), title='Sales with Moving Average')
plt.show()

# ARIMA Model (Assuming data is stationary or preprocessed)
model = ARIMA(df['Sales'], order=(5,1,0))  # Adjust p,d,q as needed
model_fit = model.fit()

# Forecast Next 30 Days
forecast = model_fit.forecast(steps=30)
forecast_dates = pd.date_range(start=df.index[-1]+pd.Timedelta(days=1), periods=30)
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast.values})
forecast_df.set_index('Date', inplace=True)

# Plot Forecast vs Historical
plt.figure(figsize=(12,6))
plt.plot(df['Sales'], label='Historical Sales')
plt.plot(forecast_df['Forecast'], label='Forecasted Sales', color='red')
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# Accuracy (train-test split can also be done for real evaluation)
predicted = model_fit.predict(start=0, end=len(df)-1, typ='levels')
rmse = np.sqrt(mean_squared_error(df['Sales'], predicted))
mape = mean_absolute_percentage_error(df['Sales'], predicted)
print(f'RMSE: {rmse:.2f}, MAPE: {mape:.2%}')
