import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import MonthEnd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score

# getting S&P/TSX Composite index data from yahoo finance
tsx = yf.download("^GSPTSE", start="2010-01-01", end="2023-01-01")
tsx.sort_values(by=['Date'], inplace=True, ascending=False)
tsx.reset_index(inplace=True)

file_path_cpi_inflation = "src/CPI_and_inflation.csv"
cpi_inflation = pd.read_csv(file_path_cpi_inflation)

file_path_gdp = "src/GDP_growthRate.csv"
gdp = pd.read_csv(file_path_gdp)

file_path_unemployment = "src/Unemployment rates.csv"
unemployment = pd.read_csv(file_path_unemployment)



# Convert 'Date' columns to datetime
tsx['Date'] = pd.to_datetime(tsx['Date']).dt.strftime('%Y/%m/%d')
cpi_inflation['Date'] = pd.to_datetime(cpi_inflation['REF_DATE'], format='%Y-%m').dt.strftime('%Y/%m/01')
gdp['Date'] = pd.to_datetime(gdp['REF_DATE'], format='%Y-%m').dt.strftime('%Y/%m/01')
# unemployment['Date'] = pd.to_datetime(unemployment['REF_DATE']).dt.strftime('%Y/%m/%d')
unemployment['Date'] = pd.to_datetime(unemployment['REF_DATE'].astype(str) + '-01-01', format='%Y-%m-%d').dt.strftime('%Y/%m/%d')

# pick rows from a dataset based on specific date ranges
START_DATE = "2010/01/01"
END_DATE = "2023/01/01"

# Filter rows
cpi_filtered_data = cpi_inflation[(cpi_inflation['Date'] >= START_DATE) & (cpi_inflation['Date'] <= END_DATE)]
gdp_filtered_data = gdp[(gdp['Date'] >= START_DATE) & (gdp['Date'] <= END_DATE)]
unemployment_filtered_data = unemployment[(unemployment['Date'] >= START_DATE) & (unemployment['Date'] <= END_DATE)]


# select columns that we need.

# 1. cpi_filtered_data: keep date, alternative measures, value
cpi_filtered_data = cpi_filtered_data[['Date', 'Alternative measures', 'VALUE']]
condition = (
    cpi_filtered_data['Alternative measures'] ==
    "Consumer Price Index (CPI), all-items excluding eight of the most volatile components as defined by the Bank of Canada and excluding the effect of changes in indirect taxes"
)
cpi_filtered_data = cpi_filtered_data[condition]

cpi_filtered_data.rename(columns={'VALUE': 'CPI value'}, inplace=True)
cpi_filtered_data.drop(columns=['Alternative measures'], inplace=True)
cpi_filtered_data['CPI value'] = cpi_filtered_data['CPI value'].values.astype("float64")


# 2. gdp_filtered_data
gdp_filtered_data = gdp_filtered_data[['Date', 'North American Industry Classification System (NAICS)', 'VALUE']]
condition = (
    gdp_filtered_data['North American Industry Classification System (NAICS)'] == "All industries [T001]"
)
gdp_filtered_data = gdp_filtered_data[condition]

gdp_filtered_data.rename(columns={'VALUE': 'GDP growth rate'}, inplace=True)
gdp_filtered_data.drop(columns=['North American Industry Classification System (NAICS)'], inplace=True)

gdp_filtered_data['GDP growth rate'] = gdp_filtered_data['GDP growth rate'].values.astype("float64")



# 3. unemployment_filtered_data
unemployment_filtered_data = unemployment_filtered_data[['Date',
                                                         'Characteristics of the population aged 15 and over',
                                                         'Educational attainment',
                                                         'VALUE']]

condition1 = (
    unemployment_filtered_data['Characteristics of the population aged 15 and over'] == "Population, Canada"
)
condition2 = (
    unemployment_filtered_data['Educational attainment'] == "All levels of education"
)

combined_condition = condition1 & condition2
unemployment_filtered_data = unemployment_filtered_data[combined_condition]

unemployment_filtered_data.rename(columns={'VALUE': 'Unemployment rate'}, inplace=True)
unemployment_filtered_data.drop(columns=['Characteristics of the population aged 15 and over'], inplace=True)
unemployment_filtered_data.drop(columns=['Educational attainment'], inplace=True)
unemployment_filtered_data['Unemployment rate'] = unemployment_filtered_data['Unemployment rate'].values.astype("float64")



# Merge datasets
# Expand CPI data to daily, assuming the CPI value is valid for the entire month
expanded_cpi = []
for idx, row in cpi_filtered_data.iterrows():
    start_date = pd.to_datetime(row['Date'])
    end_date = start_date + MonthEnd(0)  # End of the month
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    for date in daily_dates:
        expanded_cpi.append({'Date': date.strftime('%Y/%m/%d'), 'CPI value': row['CPI value']})

cpi_expanded = pd.DataFrame(expanded_cpi)

# Expand GDP data to daily, assuming the GDP value is valid for the entire month
expanded_gdp = []
for idx, row in gdp_filtered_data.iterrows():
    start_date = pd.to_datetime(row['Date'])
    end_date = start_date + MonthEnd(0)  # End of the month
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    for date in daily_dates:
        expanded_gdp.append({'Date': date.strftime('%Y/%m/%d'), 'GDP growth rate': row['GDP growth rate']})

gdp_expanded = pd.DataFrame(expanded_gdp)

# Expand unemployment data to daily, assuming the same unemployment rate for the entire year
unemployment_expanded = []
for idx, row in unemployment_filtered_data.iterrows():
    start_date = pd.to_datetime(row['Date'])
    end_date = start_date + pd.DateOffset(years=1) - pd.DateOffset(days=1)  # End of the year
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    for date in daily_dates:
        unemployment_expanded.append({'Date': date.strftime('%Y/%m/%d'), 'Unemployment rate': row['Unemployment rate']})

unemployment_expanded = pd.DataFrame(unemployment_expanded)

tsx.columns = ['_'.join(filter(None, col)) if isinstance(col, tuple) else col for col in tsx.columns]

merged_data = pd.merge(tsx, cpi_expanded, on='Date', how='inner')
merged_data = pd.merge(merged_data, gdp_expanded, on='Date', how='inner')
merged_data = pd.merge(merged_data, unemployment_expanded, on='Date', how='inner')

# merged_data is the dataFrame


# Convert 'Date' to datetime for grouping by year
merged_data['Date'] = (pd.to_datetime(merged_data['Date']) - pd.Timestamp("2010-01-01")).dt.days

# Group data by year and calculate mean for each metric
grouped_data = merged_data.groupby('Date').mean()

# Plotting CPI, GDP growth rate, and Unemployment rate
fig, ax = plt.subplots(figsize=(12, 6))

grouped_data[['CPI value', 'GDP growth rate', 'Unemployment rate']].plot(kind='bar', ax=ax)

# Customize the plot
ax.set_title('Distribution of CPI, GDP Growth Rate, and Unemployment Rate Over Years', fontsize=16)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Values', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()

# Shuffle the data to ensure randomness
merged_data = merged_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into Train (70%) and Remaining (30%)
train_data, temp_data = train_test_split(merged_data, test_size=0.3, random_state=42)

# Split the remaining data into Validation (20%) and Test (10%)
validation_data, test_data = train_test_split(temp_data, test_size=0.3333, random_state=42)

# Check the sizes of each set
print(f"Train Data: {len(train_data)} rows")
print(f"Validation Data: {len(validation_data)} rows")
print(f"Test Data: {len(test_data)} rows")

# model training

# Example: Assuming 'merged_data' has multiple target columns
target_columns = ['CPI value', 'GDP growth rate', 'Unemployment rate'] 

# Features and targets
X_train = train_data.drop(columns=target_columns)
y_train = train_data[target_columns]

X_val = validation_data.drop(columns=target_columns)
y_val = validation_data[target_columns]

X_test = test_data.drop(columns=target_columns)
y_test = test_data[target_columns]

# Linear Regression for Multi-Output Regression
lr_model = LinearRegression()
lr_multi_output = MultiOutputRegressor(lr_model)
lr_multi_output.fit(X_train, y_train)



# Initialize SVR
svr_model = SVR()
# Use MultiOutputRegressor for multiple target columns
svr_multi_output = MultiOutputRegressor(svr_model)
# Fit the model
svr_multi_output.fit(X_train, y_train)



# Initialize the RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# Wrap it with MultiOutputRegressor to handle multiple targets
rf_multi_output = MultiOutputRegressor(rf_model)
# Fit the model on training data
rf_multi_output.fit(X_train, y_train)

# Validations
# Calculate metrics
lr_preds = lr_multi_output.predict(X_val)
lr_mae = mean_absolute_error(y_val, lr_preds)
lr_rmse = mean_squared_error(y_val, lr_preds)
lr_r2 = r2_score(y_val, lr_preds)

svr_preds = svr_multi_output.predict(X_val)
svr_mae = mean_absolute_error(y_val, svr_preds)
svr_rmse = mean_squared_error(y_val, svr_preds)
svr_r2 = r2_score(y_val, svr_preds)

rf_preds = rf_multi_output.predict(X_val)
rf_mae = mean_absolute_error(y_val, rf_preds)
rf_rmse = mean_squared_error(y_val, rf_preds)
rf_r2 = r2_score(y_val, rf_preds)

# Print Results
print(f"Linear Regression ::: MAE: {lr_mae}, RMSE: {lr_rmse}, R2: {lr_r2}")
print(f"SVR ::: MAE: {svr_mae}, RMSE: {svr_rmse}, R2: {svr_r2}")
print(f"Random Forest ::: MAE: {rf_mae}, RMSE: {rf_rmse}, R2: {rf_r2}")

# Cross-Validation
X = merged_data.drop(columns=['CPI value'])  # Drop the target column to get the features
y = merged_data['CPI value']  # Target variable (dependent)
lr_cv = cross_val_score(lr_model, X, y, cv=5, scoring='neg_mean_absolute_error')
svr_cv = cross_val_score(svr_model, X, y, cv=5, scoring='neg_mean_absolute_error')
rf_cv = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_absolute_error')

print(f"Linear Regression ::: CV MAE: {-lr_cv.mean()}")
print(f"SVR ::: CV MAE: {-svr_cv.mean()}")
print(f"Random Forest ::: CV MAE: {-rf_cv.mean()}")