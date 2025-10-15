# EX.NO.10        IMPLEMENTATION OF SARIMA MODEL
### Date: 15.10.2025

### AIM:
 To implement SARIMA model using python.
 ### ALGORITHM:
 1.Explore the dataset
 
 2.Check for stationarity of time series
 
 3.Determine SARIMA models parameters p, q
 
 4.Fit the SARIMA model
 
 5.Make time series predictions and Auto-fit the SARIMA model
 
 6.Evaluate model prediction
 
### PROGRAM:

Import necessary library:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
```

Load and clean data:
```
df = pd.read_csv('India_GDP.csv', header=1)

data = df[['Year', 'GDP in (Billion) $']].copy()
data.columns = ['Year', 'GDP']

data = data.dropna()
data['Year'] = data['Year'].astype(int)
data['GDP'] = data['GDP'].astype(float)
data = data.sort_values('Year')

print("Cleaned data:")
print(data.head())
```

Plot GDP Trend:
```
plt.figure(figsize=(10,5))
plt.plot(data['Year'], data['GDP'], marker='o')
plt.title('India GDP Over Years')
plt.xlabel('Year')
plt.ylabel('GDP (Billion $)')
plt.grid(True)
plt.show()
```

Check stationarity:
```
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

check_stationarity(data['GDP'])
```

Plot ACF & PCF:
```
plot_acf(data['GDP'])
plt.show()
plot_pacf(data['GDP'])
plt.show()

```

Split data:
```
train_size = int(len(data) * 0.8)
train, test = data['GDP'][:train_size], data['GDP'][train_size:]
```

Fit SARIMA model:
```
model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,4))
result = model.fit(disp=False)
```

Make predictions & Evaluate RMSE:
```
pred = result.predict(start=len(train), end=len(train)+len(test)-1)

rmse = np.sqrt(mean_squared_error(test, pred))
print("RMSE:", rmse)
```

Plot predictions:
```
plt.figure(figsize=(10,5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, pred, color='red', label='Predicted')
plt.title('SARIMA Model - India GDP Forecast')
plt.xlabel('Index (Years)')
plt.ylabel('GDP (Billion $)')
plt.legend()
plt.grid(True)
plt.show()
```

### OUTPUT:

Original Data:

<img width="1358" height="691" alt="image" src="https://github.com/user-attachments/assets/abf7df17-eea8-4f42-92f5-7a808d3d4c06" />

Autocorrelation:

<img width="856" height="643" alt="image" src="https://github.com/user-attachments/assets/82af90a0-8de4-45bd-8e12-7d09f0c37406" />

Partial Autocorrelation:

<img width="865" height="632" alt="image" src="https://github.com/user-attachments/assets/2817d79c-023c-4ea9-b79f-a1a6b6172ae9" />

SARIMA Model:

<img width="1329" height="684" alt="image" src="https://github.com/user-attachments/assets/e00528bf-0394-40b1-bcfe-0aed77a3a4f3" />

RMSE Value:

<img width="530" height="43" alt="image" src="https://github.com/user-attachments/assets/dcba0d63-0ebf-4321-a0da-26683f2b0fd0" />

### RESULT:
Thus the program run successfully based on the ARIMA model using python.
