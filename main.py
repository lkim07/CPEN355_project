import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# getting S&P/TSX Composite index data from yahoo finance
tsx = yf.download("^GSPTSE", start="2010-01-01", end="2024-01-01")
print(tsx)