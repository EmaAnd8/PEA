import numpy as np

# Constants
M = 5000 
del_k = 10
max_error = 0.02  

K0 = 50 # Initial batch count for Scenario II

# Erlang distribution parameters (for arrival times)
k_erlang = 8
l_erlang = 1.25

# Uniform distribution parameters (for service times)
service_min, service_max = 1, 10

"""Generation of inter-arrival times based on an Erlang distribution."""
def generate_erlang_data(size, shape, rate):
   
    return np.sum(-np.log(np.random.rand(size, shape)) / rate, axis=1)
  
def generate_uniform_data(size, low, high):
    """Generates service times based on a Uniform distribution."""
    return np.random.uniform(low, high, size)

"""Calculates the mean, variance, and 95% confidence intervals for given metrics."""
def compute_confidence_intervals(K, sum_values, sum_squares):
  
    mean = sum_values / K
    variance = (sum_squares / K) - mean**2
    conf_interval = 1.96 * np.sqrt(variance / K)
    lower_bound, upper_bound = mean - conf_interval, mean + conf_interval
    relative_error = 2 * (upper_bound - lower_bound) / (upper_bound + lower_bound)
    return lower_bound, upper_bound, relative_error
    
"""metrics""" 
def process_batch(arrival_times, service_times):
 
    A = np.cumsum(arrival_times)  # Arrival times cumulative sum
    C = np.zeros(M)
    C[0] = A[0] + service_times[0]
    for i in range(1, M):
        C[i] = max(C[i - 1], A[i]) + service_times[i]
    
    busy_time = np.sum(service_times)
    total_time = C[-1] - A[0]
    
    utilization = busy_time / total_time
    throughput = M / total_time
    avg_response_time = np.mean(C - A)
    avg_num_jobs = throughput * avg_response_time
    
    return utilization, throughput, avg_response_time, avg_num_jobs

"""run the simulation"""
def run_simulation_scenario_ii():
 
    K = K0  # Start from the lower bound of the batch range
    k_range = K
    U1, U2, X1, X2, R1, R2, N1, N2 = (0,) * 8

    
    while True:
        x = generate_erlang_data(M, k_erlang, l_erlang)
        
        for _ in range(k_range):
            arrival_times = generate_erlang_data(M, k_erlang, l_erlang)
            service_times = generate_uniform_data(M, service_min, service_max)
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

        
        K += del_k  # Increment batch count by delta_k
        k_range = del_k
     
    
    print("SCENARIO II")
    print("Utilization bounds:", Ulow, Uup)
    print("Throughput bounds:", Xlow, Xup)
    print("Average number of jobs bounds:", Nlow, Nup)
    print("Average response time bounds:", Rlow, Rup)
    print("Number of batches required:", K)

run_simulation_scenario_ii()
