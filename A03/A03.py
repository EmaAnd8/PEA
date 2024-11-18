import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Function to calculate moments and other statistical measures
def calculate_statistics(data):
    # Mean
    mean = np.mean(data)
    
    # Moments: second, third, and fourth moments
    second_moment = np.mean(data**2)
    third_moment = np.mean(data**3)
    fourth_moment = np.mean(data**4)
    
    # Centered moments
    variance = np.var(data)
    third_centered_moment = np.mean((data - mean)**3)
    fourth_centered_moment = np.mean((data - mean)**4)
    
    # Standard deviation
    std_dev = np.std(data)
    
    # Coefficient of variation
    coef_variation = std_dev / mean
    
    # Skewness
    skewness = skew(data)
    
    # Excess kurtosis (Fisher's definition)
    excess_kurtosis = kurtosis(data)
    
    # Fourth standardized moment (raw kurtosis)
    fourth_standardized_moment = fourth_centered_moment / std_dev**4
    
    # Median
    median = np.median(data)
    
    # First and third quartile (25th and 75th percentiles)
    first_quartile = np.percentile(data, 25)
    third_quartile = np.percentile(data, 75)
    
    # 5th and 90th percentiles
    fifth_percentile = np.percentile(data, 5)
    ninetieth_percentile = np.percentile(data, 90)
    
    # Print statistics
    print(f"Mean: {mean:.4f}")
    print(f"2nd Moment: {second_moment:.4f}")
    print(f"3rd Moment: {third_moment:.4f}")
    print(f"4th Moment: {fourth_moment:.4f}")
    print(f"Variance: {variance:.4f}")
    print(f"3rd Centered Moment: {third_centered_moment:.4f}")
    print(f"4th Centered Moment: {fourth_centered_moment:.4f}")
    print(f"Skewness: {skewness:.4f}")
    print(f"Excess Kurtosis: {excess_kurtosis:.4f}")
    print(f"4th Standardized Moment: {fourth_standardized_moment:.4f}")
    print(f"Standard Deviation: {std_dev:.4f}")
    print(f"Coefficient of Variation: {coef_variation:.4f}")
    print(f"Median: {median:.4f}")
    print(f"First Quartile: {first_quartile:.4f}")
    print(f"Third Quartile: {third_quartile:.4f}")
    print(f"5th Percentile: {fifth_percentile:.4f}")
    print(f"90th Percentile: {ninetieth_percentile:.4f}")

# Function to calculate Pearson correlation coefficients for lags m = 1 to m = 100
def calculate_pearson_corr(data, max_lag=100):
    autocorr = [np.corrcoef(data[:-lag], data[lag:])[0, 1] for lag in range(1, max_lag+1)]
    
    # Plot the Pearson Correlation Coefficients
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_lag + 1), autocorr, marker='o', linestyle='-', color='b')
    plt.title("Pearson Correlation Coefficients for Lags (m=1 to 100)")
    plt.xlabel("Lag (m)")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.grid(True)
    plt.show()

# Modified function to plot the survival function on log-log scale
def plot_survival_function(data):
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data)+1) / float(len(sorted_data))
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_data, yvals, linestyle='-', marker=None, color='r')
    plt.title("Approximated CDF of the Distribution")
    plt.xlabel("Inter-arrival Time")
    plt.ylabel("CDF")
    plt.xscale("log")
  
    plt.grid(True)
    plt.show()

# Function to plot the approximated CDF
def plot_cdf(data, xlim=None, ylim=None):
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data)+1) / float(len(sorted_data))
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_data, yvals, linestyle='-', marker=None, color='r')
    plt.title("Approximated CDF of the Distribution")
    plt.xlabel("Inter-arrival Time")
    plt.ylabel("CDF")
    
    if xlim is not None:
        plt.xlim(xlim)     # Set x-axis limits
    
    if ylim is not None:
        plt.ylim(ylim)     # Set y-axis limits
    
    plt.grid(True)
    plt.show()


trace1 = np.loadtxt('Trace1.csv', delimiter=',')
trace2 = np.loadtxt('Trace2.csv', delimiter=',')
trace3 = np.loadtxt('Trace3.csv', delimiter=',')


print("Statistics for Trace1:")
calculate_statistics(trace1)
calculate_pearson_corr(trace1)
plot_cdf(trace1, xlim=(0, trace1.max()), ylim=(0, 1))


print("\nStatistics for Trace2:")
calculate_statistics(trace2)
calculate_pearson_corr(trace2)


plot_survival_function(trace2)


print("\nStatistics for Trace3:")
calculate_statistics(trace3)
calculate_pearson_corr(trace3)
plot_cdf(trace3, xlim=(0, trace3.max()), ylim=(0, 1))
