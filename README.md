# Air-Quality-Trend-Analysis-Script
This script calculates long-term trends in air quality pollutants (NO2, PM2.5, PM10, O3) using hourly datasets.

"""
Air Quality Trend Analysis Script
Author: Simon William Mkasimongwa
Date: 19/11/2024

Description:
This script analyzes long-term trends in air quality pollutants (NO2, PM2.5, PM10, O3) using hourly datasets.
The workflow includes:
1. Aggregating hourly data into daily averages.
2. Calculating annual means and standard deviations.
3. Performing linear regression on annual means to identify trends.
4. Assessing the significance of trends using a t-test.

Inputs:
- A CSV file containing hourly air quality data with a 'Datetime' column.

Outputs:
- A CSV file summarizing annual trends, slopes, and significance.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

# Function to load data
def load_data(file_path):
    """
    Load the air quality data and preprocess it.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Processed data with hourly 'Datetime'.
    """
    data = pd.read_csv(file_path)
    data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d/%m/%Y %H:%M')
    return data

# Function to calculate daily averages
def calculate_daily_means(data, pollutants):
    """
    Calculate daily mean concentrations from hourly data.

    Args:
        data (pd.DataFrame): Hourly air quality data.
        pollutants (list): List of pollutant columns.

    Returns:
        pd.DataFrame: Daily averages with dates and years.
    """
    data['Date'] = data['Datetime'].dt.date
    daily_data = data.groupby('Date')[pollutants].mean().reset_index()
    daily_data['Year'] = pd.to_datetime(daily_data['Date']).dt.year
    return daily_data

# Function to calculate linear regression
def calculate_linear_regression(X, y):
    """
    Perform linear regression and calculate the slope and p-value.

    Args:
        X (np.ndarray): Independent variable (e.g., years).
        y (np.ndarray): Dependent variable (e.g., pollutant concentrations).

    Returns:
        float, float: Slope of the regression line, p-value of the slope.
    """
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    _, _, _, p_value, _ = stats.linregress(X.flatten(), y)
    return slope, p_value

# Main function
def analyze_air_quality(file_path, pollutants, output_file):
    """
    Analyze air quality trends and save results to a CSV file.

    Args:
        file_path (str): Path to the input CSV file.
        pollutants (list): List of pollutant columns to analyze.
        output_file (str): Path to save the results CSV file.

    Returns:
        pd.DataFrame: Results summary.
    """
    data = load_data(file_path)
    daily_data = calculate_daily_means(data, pollutants)
    annual_stats = daily_data.groupby('Year')[pollutants].agg(['mean', 'std'])

    results = []
    for pollutant in pollutants:
        annual_mean = daily_data.groupby('Year')[pollutant].mean().dropna()
        X = np.array(annual_mean.index).reshape(-1, 1)
        y = annual_mean.values

        if len(X) > 1:
            linear_slope, p_value = calculate_linear_regression(X, y)
        else:
            linear_slope, p_value = np.nan, np.nan

        results.append({
            'Pollutant': pollutant,
            'Mean': annual_stats[(pollutant, 'mean')].mean(),
            'Standard Deviation': annual_stats[(pollutant, 'std')].mean(),
            'Linear Regression Slope': linear_slope,
            'p-value': p_value,
            'Significant': 'Yes' if p_value < 0.05 and not np.isnan(p_value) else 'No'
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    return results_df

# Example usage
if __name__ == "__main__":
    # User-configurable inputs
    input_file = 'example_air_quality.csv'  # Replace with your file
    output_file = 'air_quality_trend_analysis.csv'
    pollutants = ['NO2', 'PM2.5', 'PM10', 'O3']

    # Run the analysis
    results = analyze_air_quality(input_file, pollutants, output_file)
    print("Analysis complete. Results saved to:", output_file)
    print(results)
