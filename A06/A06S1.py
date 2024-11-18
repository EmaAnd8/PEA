import numpy as np

# Constants
M = 5000  # Number of jobs per batch
delta_k = 10  # Batch increment
max_error = 0.02  # Maximum relative error
K_lower, K_upper = 10000, 13500  # Range for number of batches (K)

# Hyper-exponential distribution parameters
l1, l2 = 0.025, 0.1
p1 = 0.35
K0=50

# Weibull distribution parameters
k_weib = 0.333
lambda_weib = 2.5

def generate_hyper_exponential_data(size, lambda1, lambda2, p1):
    """Generates inter-arrival times based on a hyper-exponential distribution."""
    return np.array([-np.log(np.random.rand()) / (lambda1 if np.random.rand() < p1 else lambda2) for _ in range(size)])

def generate_weibull_data(size, shape, scale):
    """Generates service times based on a Weibull distribution."""
    return scale * np.random.weibull(shape, size)

def compute_confidence_intervals(K, sum_values, sum_squares):
    """Calculates the mean, variance, and 95% confidence intervals for given metrics."""
    mean = sum_values / K
    variance = (sum_squares / K) - mean**2
    conf_interval = 1.96 * np.sqrt(variance / K)
    lower_bound, upper_bound = mean - conf_interval, mean + conf_interval
    relative_error = 2 * (upper_bound - lower_bound) / (upper_bound + lower_bound)
    return lower_bound, upper_bound, relative_error

def process_batch(arrival_times, service_times):
    """Calculates performance metrics for a single batch of data."""
    A = np.cumsum(arrival_times)  # Arrival times cumulative sum
    C = np.zeros(M)
    C[0] = A[0] + service_times[0]
    for i in range(1, M):
        C[i] = max(C[i - 1], A[i]) + service_times[i]
    
    busy_time = np.sum(service_times)
    total_time = C[-1] - A[0]
    
    utilization = busy_time / total_time
    X = M / total_time
    avg_response_time = np.mean(C - A)
    avg_num_jobs = X * avg_response_time
    
    return utilization, X, avg_response_time, avg_num_jobs

def run_simulation():
    """Main function to run the simulation and compute confidence intervals for Scenario I."""
    K = K0  # Start from the lower bound of the batch range
    U1, U2, X1, X2, R1, R2, N1, N2 = (0,) * 8
    
    while True:
        for _ in range(delta_k):
            arrival_times = generate_hyper_exponential_data(M, l1, l2, p1)
            service_times = generate_weibull_data(M, k_weib, lambda_weib)
            Uk, Xk, Rk, Nk = process_batch(arrival_times, service_times)
            
            # Accumulate sums for each metric
            U1 += Uk; U2 += Uk**2
            X1 += Xk; X2 += Xk**2
            R1 += Rk; R2 += Rk**2
            N1 += Nk; N2 += Nk**2
        
        # Calculate confidence intervals and relative errors for each metric
        Ulow, Uup, errorU = compute_confidence_intervals(K, U1, U2)
        Xlow, Xup, errorX = compute_confidence_intervals(K, X1, X2)
        Rlow, Rup, errorR = compute_confidence_intervals(K, R1, R2)
        Nlow, Nup, errorN = compute_confidence_intervals(K, N1, N2)

        # Check if all metrics meet the relative error threshold
        if all(error < max_error for error in [errorU, errorX, errorR, errorN]):
            break
        
        K += delta_k  # Increment batch count by delta_k
    
    print("SCENARIO I")
    print("Utilization bounds:", Ulow, Uup)
    print("Throughput bounds:", Xlow, Xup)
    print("Average number of jobs bounds:", Nlow, Nup)
    print("Average response time bounds:", Rlow, Rup)
    print("Number of batches required:", K)

run_simulation()
