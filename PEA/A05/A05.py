import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import expon, pareto, gamma

# Function to generate samples and plot for each distribution
def generate_and_plot_distributions(n=10000):
    # 1. Generate samples from an Exponential distribution with rate λ = 0.25
    lambda_exp = 0.25
    exp_samples = [-math.log(random.random()) / lambda_exp for _ in range(n)]
    x_exp = np.linspace(0, 50, 500)
    exp_cdf = expon.cdf(x_exp, scale=1/lambda_exp)
    
    # Plot Exponential distribution
    plot_cdf(exp_samples, x_exp, exp_cdf, 'Exponential Distribution (λ=0.25)', 'Exponential')

    # 2. Generate samples from a Pareto distribution with a = 2.5, m = 3
    a_pareto = 2.5
    m_pareto = 3
    pareto_samples = [m_pareto * (random.random() ** (-1/a_pareto)) for _ in range(n)]
    x_pareto = np.linspace(m_pareto, 50, 500)
    pareto_cdf = pareto.cdf(x_pareto, b=a_pareto, scale=m_pareto)

    # Plot Pareto distribution
    plot_cdf(pareto_samples, x_pareto, pareto_cdf, 'Pareto Distribution (a=2.5, m=3)', 'Pareto')

    # 3. Generate samples from an Erlang distribution with k=8, λ=0.8
    k_erlang = 8
    lambda_erlang = 0.8
    erlang_samples = [sum(-math.log(random.random()) / lambda_erlang for _ in range(k_erlang)) for _ in range(n)]
    x_erlang = np.linspace(0, 50, 500)
    erlang_cdf = gamma.cdf(x_erlang, a=k_erlang, scale=1/lambda_erlang)
    
    # Plot Erlang distribution
    plot_cdf(erlang_samples, x_erlang, erlang_cdf, 'Erlang Distribution (k=8, λ=0.8)', 'Erlang')

    # 4. Generate samples from a Hypo-exponential distribution with rates λ1=0.25, λ2=0.4
    lambda1_hypo = 0.25
    lambda2_hypo = 0.4
    hypoexp_samples = [(-math.log(random.random()) / lambda1_hypo) + (-math.log(random.random()) / lambda2_hypo) for _ in range(n)]
    x_hypoexp = np.linspace(0, 50, 500)
    hypoexp_cdf = 1 - (lambda2_hypo * np.exp(-lambda1_hypo * x_hypoexp) - lambda1_hypo * np.exp(-lambda2_hypo * x_hypoexp)) / (lambda2_hypo - lambda1_hypo)
    
    # Plot Hypo-exponential distribution
    plot_cdf(hypoexp_samples, x_hypoexp, hypoexp_cdf, 'Hypo-Exponential Distribution (λ1=0.25, λ2=0.4)', 'Hypo-Exponential')

    # 5. Generate samples from a Hyper-exponential distribution with rates λ1=1, λ2=0.05, p1=0.75
    lambda1_hyper = 1
    lambda2_hyper = 0.05
    p1_hyper = 0.75
    hyperexp_samples = []
    for _ in range(n):
        if random.random() < p1_hyper:
            hyperexp_samples.append(-math.log(random.random()) / lambda1_hyper)
        else:
            hyperexp_samples.append(-math.log(random.random()) / lambda2_hyper)

    x_hyperexp = np.linspace(0, 50, 500)
    hyperexp_cdf = p1_hyper * expon.cdf(x_hyperexp, scale=1/lambda1_hyper) + (1 - p1_hyper) * expon.cdf(x_hyperexp, scale=1/lambda2_hyper)
    
    # Plot Hyper-exponential distribution
    plot_cdf(hyperexp_samples, x_hyperexp, hyperexp_cdf, 'Hyper-Exponential Distribution (λ1=1, λ2=0.05, p1=0.75)', 'Hyper-Exponential')

# Function to plot the empirical CDF and theoretical CDF
def plot_cdf(samples, x_values, theoretical_cdf, title, label):
    # Calculate empirical CDF
    sorted_samples = np.sort(samples)
    empirical_cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    
    # Plot the empirical CDF and theoretical CDF
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_samples, empirical_cdf, label=f'Empirical CDF of {label}', linestyle='--')
    plt.plot(x_values, theoretical_cdf, label='Theoretical CDF', color='red')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlim([0, 50])
    plt.ylim([0, 1])
    plt.show()

# Run the function to generate and plot distributions
generate_and_plot_distributions(n=10000)
