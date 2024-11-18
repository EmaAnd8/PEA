import numpy as np
import matplotlib.pyplot as plt

from scipy import stats, optimize
from scipy.special import gamma
import math

# Compute the first three moments 
def compute_moments(data):
    first_moment = np.mean(data)
    second_moment = np.mean(data ** 2)
    third_moment = np.mean(data ** 3)
    return first_moment, second_moment, third_moment

# Compute coefficient of variation
def compute_cv(sigma_dis,mean_dis):
    return sigma_dis / mean_dis

# Uniform distribution fitting
def fit_uniform(data):
    mean = np.mean(data)
    variance = np.var(data)
    a = mean - np.sqrt(3 * variance)
    b = mean + np.sqrt(3 * variance)
    print(f"Uniform distribution parameters: a={a}, b={b}")
    return a, b

# Exponential distribution fitting 
def fit_exponential(data):
    mean = np.mean(data)
    rate = 1 / mean
    print(f"Exponential distribution rate: λ={rate}")
    return rate

# Weibull distribution fitting using method of moments
def fit_weibull_mom(data):
    mu = np.mean(data)
    sigma = np.std(data)
    cv = sigma / mu

    # Define function to solve for shape parameter k
    def func(k):
        return (gamma(1 + 2 / k) / gamma(1 + 1 / k) ** 2) - (1 + cv ** 2)

    # Use a numerical solver to find the root
    k_initial_guess = 1.0
    try:
        k = optimize.brentq(func, a=0.1, b=10)
    except ValueError:
        k = k_initial_guess  # Use initial guess if root finding fails

    # Compute scale parameter λ
    lam = mu / gamma(1 + 1 / k)

    print(f"Weibull distribution parameters (method of moments): shape={k}, scale={lam}")
    return k, lam

# Pareto distribution fitting using method of moments with second moment
def fit_pareto(data):
    mean = np.mean(data)
    second_moment = np.mean(data ** 2)

    # Define function to solve for alpha
    def func(x):
        alpha, xm = x
        first_moment_computed = (alpha * xm) / (alpha - 1)
        second_moment_computed = (alpha * (xm ** 2)) / (alpha - 2)
        return (np.abs(first_moment_computed / mean - 1)) ** 2 + (np.abs(second_moment_computed / second_moment - 1)) ** 2

    result = optimize.minimize(func, np.array([3, 1]), bounds=((0.001, 100.0), (0.001, 100.0)), constraints=[{'type': 'ineq', 'fun': lambda x: x[0] - 2.01}])
    alpha, xm = result.x

    print(f"Pareto distribution parameters (method of moments): α={alpha}, m={xm}")
    return alpha, xm

# Erlang distribution fitting using direct expressions
def fit_erlang(data):
    mean = np.mean(data)
    variance = np.var(data)
    if variance == 0:
        print("Variance is zero; cannot fit Erlang distribution.")
        return 1, 1e-10
    k = np.round((mean ** 2) / (np.mean(data ** 2) - mean ** 2))  # Shape parameter
    lambda_erlang = k / mean
    
    print(f"Erlang distribution parameters: k={int(k)}, λ={lambda_erlang}")
    return int(k), lambda_erlang

# Hyper-Exponential distribution fitting using MLE
def fit_hyperexponential_mle(data):
    sorted_data = np.sort(data)
    N = data.shape[0]

    M1t = np.sum(sorted_data) / N
    M2t = np.sum(sorted_data ** 2) / N

    def fun(x):
        l1, l2, p1 = x
        p2 = 1 - p1
        return -np.sum(np.log(p1 * l1 * np.exp(-l1 * sorted_data) + p2 * l2 * np.exp(-l2 * sorted_data)))

    sx = optimize.minimize(fun, np.array([0.8 / M1t, 1.2 / M1t, 0.4]), bounds=((0.001, 100.0), (0.001, 100.0), (0.001, 0.999)),
                           constraints=[{'type': 'ineq', 'fun': lambda x: x[1] - x[0] - 0.001}])
    l1d = sx.x[0]
    l2d = sx.x[1]
    p1d = sx.x[2]

    print(f"Hyper-Exponential distribution parameters: p1={p1d}, l1={l1d}, l2={l2d}")
    return p1d, l1d, l2d

# Hypo-Exponential distribution fitting using MLE
def fit_hypoexponential_mle(data):
    N = data.shape[0]
    srt = data.copy()
    srt.sort(0)
    M1t = np.sum(srt) / N
    M2t = np.sum(srt ** 2) / N

    def fun(x):
        l1 = x[0]
        l2 = x[1]
        if l1 == l2:
            return -1000000 - l1 - l2
        else:
            return -np.sum(np.log(l1 * l2 / (l1 - l2) * (np.exp(-l2 * srt) - np.exp(-l1 * srt))))

    sx = optimize.minimize(fun, np.array([1 / (0.7 * M1t), 1 / (0.3 * M1t)]), bounds=((0.001, 100.0), (0.001, 100.0)), constraints=[{'type': 'ineq', 'fun': lambda x: x[1] - x[0] - 0.001}])
    l1d = sx.x[0]
    l2d = sx.x[1]

    print("l1, l2 = ", l1d, l2d)
    return l1d, l2d


# Function to compute Coefficient of Variation (CV)
def compute_cv(sigma, mean):
    return sigma / mean

# Function to fit distributions and compute their CDFs, then plot them
def fit_and_plot(data, title):
    N = data.shape[0]
    sorted_data = np.sort(data)
    probV = np.r_[1.:N + 1] / N
    t=(np.r_[1.:N+1]/500)# Define t to match x-axis values for fitted distributions

    fits = []
    labels = []

    # Uniform Distribution Fitting
    a, b = fit_uniform(data)
    Funif = (t - a) / (b - a)
    Funif = np.clip(Funif, 0, 1)  # Clip values to be in the range [0, 1]
    fits.append(Funif)
    labels.append("Uniform distribution fitting")

    # Exponential Distribution Fitting
    rate = fit_exponential(data)
    Fexp = 1 - np.exp(-rate * t)  # Direct calculation for the CDF of Exponential
    fits.append(Fexp)
    labels.append("Exponential distribution fitting")

    # Weibull Distribution Fitting
    k, lam = fit_weibull_mom(data)
    FWeib = 1 - np.exp(-(t / lam) ** k)  # Direct calculation for the CDF of Weibull
    fits.append(FWeib)
    labels.append("Weibull Distribution Fitting")

    # Pareto Distribution Fitting
    alpha, xm = fit_pareto(data)
    FPar = 1 - (xm / t) ** alpha  # Direct calculation for the CDF of Pareto
    FPar[t < xm] = 0  # CDF is 0 for t < xm
    fits.append(FPar)
    labels.append("Pareto Distribution Fitting")

    # Erlang Distribution Fitting
    k_erlang, lambda_erlang = fit_erlang(data)
    sigma_erl = np.sqrt(k_erlang / (lambda_erlang ** 2))
    mean_erl = k_erlang / lambda_erlang
    cv_erl = compute_cv(sigma_erl, mean_erl)
    if cv_erl <= 1:
        FErl = 1 - np.exp(-lambda_erlang * t) * sum([(lambda_erlang * t) ** n / math.factorial(n) for n in range(k_erlang)])
        fits.append(FErl)
        labels.append("Erlang Distribution Fitting")
    else:
        print("cv > 1, so we can't plot Erlang distribution")

    # Hyper-Exponential Distribution Fitting
    p, lambda1, lambda2 = fit_hyperexponential_mle(data)
    sigma_hyper = np.sqrt((((2 * p) / (lambda1 ** 2)) + ((2 * (1 - p)) / (lambda2 ** 2))) -
                          (((p / lambda1) + ((1 - p) / lambda2)) ** 2))
    mean_hyper = (p / lambda1) + ((1 - p) / lambda2)
    cv_hyper = compute_cv(sigma_hyper, mean_hyper)
    if cv_hyper > 1:
        Fhyper = p * (1 - np.exp(-lambda1 * t)) + (1 - p) * (1 - np.exp(-lambda2 * t))
        fits.append(Fhyper)
        labels.append("Hyper-Exponential Distribution Fitting")
    else:
        print("No Hyper-Exponential distribution is admissible due to CV < 1")

    # Hypo-Exponential Distribution Fitting
    try:
        lambda1_hypo, lambda2_hypo = fit_hypoexponential_mle(data)
        sigma_hypo = np.sqrt((1 / (lambda1_hypo ** 2)) + (1 / (lambda2_hypo ** 2)))
        mean_hypo = (1 / lambda1_hypo) + (1 / lambda2_hypo)
        cv_hypo = compute_cv(sigma_hypo, mean_hypo)
        if cv_hypo <= 1:
            diff = lambda2_hypo - lambda1_hypo
            Fhypo = 1 - (lambda2_hypo * np.exp(-lambda1_hypo * t) - lambda1_hypo * np.exp(-lambda2_hypo * t)) / diff
            fits.append(Fhypo)
            labels.append("Hypo-Exponential Distribution Fitting")
        else:
            print("No Hypo-Exponential distribution is admissible due to CV > 1")
    except RuntimeError as e:
        print("Hypo-Exponential fitting error:", e)

    # Plot the empirical and fitted CDFs
    plot_cdf(data,sorted_data, probV, fits, labels, title)

# Function to plot the empirical CDF and fitted CDFs
def plot_cdf(data,sorted_data, probV, fits, labels, title):
    N = data.shape[0]
    t = t=np.r_[1.:N+1]/500  # Define t to match x-axis values for fitted distributions
    
    # Plot the empirical CDF using dots for clarity
    plt.plot(sorted_data, probV, ".", label="FTrace")

    # Plot each fitted distribution with the given labels
    for fit, label in zip(fits, labels):
        plt.plot(t, fit, label=label)

    # Set titles and labels
    plt.title(title)
    plt.xlabel('Service Time')
    plt.ylabel('CDF')
    plt.xlim(-2, 50)  # Set x-axis limits as specified
    plt.legend()
    plt.grid(True)
    plt.show()


# Load data and process
trace1 = np.loadtxt('Trace1.csv', delimiter=",")
trace2 = np.loadtxt('Trace2.csv', delimiter=",")

print("Processing Trace 1")
first_moment1, second_moment1, third_moment1 = compute_moments(trace1)
print(f"Trace 1 Moments - First: {first_moment1}, Second: {second_moment1}, Third: {third_moment1}")
fit_and_plot(trace1, "Trace 1: CDF Comparison")

print("Processing Trace 2")
first_moment2, second_moment2, third_moment2 = compute_moments(trace2)
print(f"Trace 2 Moments - First: {first_moment2}, Second: {second_moment2}, Third: {third_moment2}")
fit_and_plot(trace2, "Trace 2: CDF Comparison")
