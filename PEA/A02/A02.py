import numpy as np

# Load inter-arrival and service times from CSV files
a_i = np.loadtxt("Logger1.csv", delimiter=";")  # Inter-arrival times
s_i = np.loadtxt("Logger2.csv", delimiter=";")  # Service times

# Function to calculate arrivals (cumulative sum of inter-arrival times)
def arrivals(alpha_a_i):
    return np.cumsum(alpha_a_i)

# Function to calculate completions based on service times
def completions(arrivals_tmp, s_i):
    completions_tmp = np.zeros(len(s_i))
    completions_tmp[0] = arrivals_tmp[0] + s_i[0]  # First job completes at arrival time + service time
    for i in range(1, len(s_i)):
        completions_tmp[i] = max(completions_tmp[i - 1], arrivals_tmp[i]) + s_i[i]  # Next job completes after the last one finishes
    return completions_tmp

# --- Alpha_1 computation for the maximum arrival rate ---
R_first_question = 20  # Desired response time in minutes
alpha_first_question = 1.0
R_tmp = 0  # Temporary response time
lambda_first_question = -1

AT_CT = len(a_i)  # Number of arrivals/completions

# Iteratively adjust alpha to find the maximum arrival rate (alpha_1)
while R_tmp < R_first_question:
    # Adjust inter-arrival times by alpha_1
    alpha_a_i = alpha_first_question * a_i
    
    # Calculate average inter-arrival time
    Abar_tmp = np.sum(alpha_a_i) / AT_CT
    
    # Arrival rate
    lambda_first_question = 1 / Abar_tmp
    
    # Compute arrivals and completions
    arrivals_tmp = arrivals(alpha_a_i)
    completions_tmp = completions(arrivals_tmp, s_i)
    
    # Response times (completion time - arrival time)
    rt = completions_tmp - arrivals_tmp
    R_tmp = np.mean(rt)
    
    # Decrease alpha_1 to approach the desired response time of 20 minutes
    alpha_first_question -= 0.0001
    
    # Break the loop if alpha becomes non-positive
    if alpha_first_question <= 0:
        print("Alpha_1 reached an unrealistic value. Exiting.")
        break

# Output the results for alpha_1
print(f"Alpha_1 (for max arrival rate) = {alpha_first_question:.4f}")
print(f"Max Arrival Rate = {lambda_first_question:.4f} jobs/min")

# --- Alpha_2 computation to generate an arrival rate of 1.2 jobs/min ---
lambda_target = 1.2  # Desired arrival rate in jobs/min
alpha_2 = 1.0
lambda_new_computed = 0

# Iteratively adjust alpha_2 to approach the desired arrival rate
while lambda_new_computed < lambda_target:
    # Adjust inter-arrival times using alpha_2
    alpha_a_i_2 = alpha_2 * a_i
    
    # Compute the new arrival rate
    lambda_new_computed = 1 / np.mean(alpha_a_i_2)
    
    # Decrease alpha_2 to approach the target arrival rate
    alpha_2 -= 0.0001
    
    # Break the loop if alpha_2 becomes non-positive
    if alpha_2 <= 0:
        print("Alpha_2 reached an unrealistic value. Exiting.")
        break

# Output the results for alpha_2
print(f"Alpha_2 to achieve {lambda_target} jobs/min = {alpha_2:.4f}")
print(f"New Computed Arrival Rate = {lambda_new_computed:.4f} jobs/min")

# Recompute arrivals with the new inter-arrival times (using alpha_2)
new_arrival_times = arrivals(alpha_a_i_2)

# --- Compute beta to reduce service time so that average response time is less than 15 minutes ---

beta = 0.4  # Start with an initial guess for beta
new_average_response_time = 100  # Initial high value to start the loop

while new_average_response_time > 15:
    # Decrease beta in small steps for precision
    beta -= 0.0001

    # Adjust the service times based on beta
    new_service_times = beta * s_i

    # Use the completions function to calculate the new completion times
    new_completion_times = completions(new_arrival_times, new_service_times)

    # Compute new response times
    new_response_times = new_completion_times - new_arrival_times
    new_average_response_time = np.mean(new_response_times)

# Output the results for beta
print(f"Beta to achieve the target response time = {beta:.4f}")
print(f"New Computed Average Response Time = {new_average_response_time:.4f} minutes")
