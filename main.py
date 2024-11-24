import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import MonthEnd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# getting S&P/TSX Composite index data from yahoo finance
tsx = yf.download("^GSPTSE", start="2010-01-01", end="2023-01-01")
# print(tsx.head())
print(tsx.columns.values)
# list(tsx['Volume', '^GSPTSE'].tolist())
# print(tsx.info)
tsx.sort_values(by=['Date'], inplace=True, ascending=False)
tsx.reset_index(inplace=True)

file_path_cpi_inflation = "src/CPI_and_inflation.csv"
cpi_inflation = pd.read_csv(file_path_cpi_inflation)

file_path_gdp = "src/GDP_growthRate.csv"
gdp = pd.read_csv(file_path_gdp)

file_path_unemployment = "src/Unemployment rates.csv"
unemployment = pd.read_csv(file_path_unemployment)

# Preview the data
print(cpi_inflation.head())

print(gdp.head())

# print(unemployment.head())
print(unemployment)

# Convert 'Date' columns to datetime
# tsx = tsx.reset_index()
# cpi_inflation = cpi_inflation.reset_index()
# gdp = gdp.reset_index()
# unemployment = unemployment.reset_index()

tsx['Date'] = pd.to_datetime(tsx['Date']).dt.strftime('%Y/%m/%d')
cpi_inflation['Date'] = pd.to_datetime(cpi_inflation['REF_DATE'], format='%Y-%m').dt.strftime('%Y/%m/01')
gdp['Date'] = pd.to_datetime(gdp['REF_DATE'], format='%Y-%m').dt.strftime('%Y/%m/01')
# unemployment['Date'] = pd.to_datetime(unemployment['REF_DATE']).dt.strftime('%Y/%m/%d')
unemployment['Date'] = pd.to_datetime(unemployment['REF_DATE'].astype(str) + '-01-01', format='%Y-%m-%d').dt.strftime('%Y/%m/%d')
# print(tsx['Date'])
# print(cpi_inflation['Date'])
# print(gdp['Date'])
# print(unemployment['Date'])

print(unemployment)

# pick rows from a dataset based on specific date ranges

START_DATE = "2010/01/01"
END_DATE = "2023/01/01"

# Filter rows
cpi_filtered_data = cpi_inflation[(cpi_inflation['Date'] >= START_DATE) & (cpi_inflation['Date'] <= END_DATE)]
gdp_filtered_data = gdp[(gdp['Date'] >= START_DATE) & (gdp['Date'] <= END_DATE)]
unemployment_filtered_data = unemployment[(unemployment['Date'] >= START_DATE) & (unemployment['Date'] <= END_DATE)]


# Display the filtered dataset
# print(cpi_filtered_data)
# print(gdp_filtered_data)
print(unemployment_filtered_data)

# select columns that we need.

# 1. cpi_filtered_data: keep date, alternative measures, value
cpi_filtered_data = cpi_filtered_data[['Date', 'Alternative measures', 'VALUE']]
condition = (
    cpi_filtered_data['Alternative measures'] ==
    "Consumer Price Index (CPI), all-items excluding eight of the most volatile components as defined by the Bank of Canada and excluding the effect of changes in indirect taxes"
)
cpi_filtered_data = cpi_filtered_data[condition]
# print(cpi_filtered_data)

cpi_filtered_data.rename(columns={'VALUE': 'CPI value'}, inplace=True)
cpi_filtered_data.drop(columns=['Alternative measures'], inplace=True)

print(cpi_filtered_data)

# 2. gdp_filtered_data
gdp_filtered_data = gdp_filtered_data[['Date', 'North American Industry Classification System (NAICS)', 'VALUE']]
condition = (
    gdp_filtered_data['North American Industry Classification System (NAICS)'] == "All industries [T001]"
)
gdp_filtered_data = gdp_filtered_data[condition]
# print(gdp_filtered_data)

gdp_filtered_data.rename(columns={'VALUE': 'GDP growth rate'}, inplace=True)
gdp_filtered_data.drop(columns=['North American Industry Classification System (NAICS)'], inplace=True)

print(gdp_filtered_data)

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

print(unemployment_filtered_data)

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
# print(merged_data['Date'].unique())
# print(unemployment_filtered_data['Date'].unique())
merged_data = pd.merge(merged_data, unemployment_expanded, on='Date', how='inner')
print(merged_data)