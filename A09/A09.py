import math

# Function to compute probabilities for M/M/c queues
def mmc_probabilities(arrival_rate, service_rate, num_servers, n_max):
    rho = arrival_rate / (num_servers * service_rate)
    p0 = mmc_p0(arrival_rate, service_rate, num_servers)
    probabilities = []

    # P(N=n) for n < c
    for n in range(num_servers):
        probabilities.append((p0 * (arrival_rate / service_rate)**n) / math.factorial(n))

    # P(N=n) for n >= c
    for n in range(num_servers, n_max + 1):
        probabilities.append((p0 * (arrival_rate / service_rate)**n) / 
                             (math.factorial(num_servers) * num_servers**(n - num_servers)))

    return probabilities

# Helper function to compute P0 for M/M/c queue
def mmc_p0(arrival_rate, service_rate, num_servers):
    rho = arrival_rate / (num_servers * service_rate)
    sum_terms = sum([(arrival_rate / service_rate)**n / math.factorial(n) for n in range(num_servers)])
    last_term = ((arrival_rate / service_rate)**num_servers) / \
                (math.factorial(num_servers) * (1 - rho))
    return 1 / (sum_terms + last_term)





# Scenario 1: M/M/1 Queue
def scenario_1(lambda_rate, mu_rate):
    rho = lambda_rate / mu_rate
    U=rho
    P_N2 = (1 - rho) * rho**2
    P_N_lt_5 = sum([(1 - rho) * rho**n for n in range(5)])
    Lq = rho**2 / (1 - rho)
    R= 1 / (mu_rate - lambda_rate)
    P_W_gt_2 = math.exp(-(mu_rate - lambda_rate) * 2)
    W_95 = -math.log(1 - ((95/100))) * R

    return U, P_N2, P_N_lt_5, Lq, R, P_W_gt_2, W_95

# Scenario 2: M/M/2 Queue
def scenario_2(lambda_rate, mu_rate, num_servers, n_max=10):
    rho = lambda_rate / (num_servers * mu_rate)
    U_bar=rho
    U=lambda_rate/mu_rate
    p_n_vector = mmc_probabilities(lambda_rate, mu_rate, num_servers, n_max)
    P_N2 = p_n_vector[2]
    P_N_lt_5 = sum(p_n_vector[:5])
    Lq = (p_n_vector[num_servers] * rho) / (1 - rho)**2
    R = (1 / mu_rate)/(1-rho**2)

    return U,U_bar, P_N2, P_N_lt_5, Lq, R

# Scenario 3: M/M/c Queue
def scenario_3(lambda_rate, mu_rate, num_servers, n_max=10):
    rho = lambda_rate / (num_servers * mu_rate)
    U_bar=rho
    U=lambda_rate/mu_rate
    p_n_vector = mmc_probabilities(lambda_rate, mu_rate, num_servers, n_max)
    P_N2 = p_n_vector[2]
    P_N_lt_5 = sum(p_n_vector[:5])
    Lq = (p_n_vector[num_servers] * rho) / (1 - rho)**2
    D=(1/mu_rate)
    R = (D)+((D/(num_servers*(1-rho)))/(1+((1-rho)*(math.factorial(num_servers)/((num_servers*rho)**num_servers)*sum((num_servers*rho)**k/math.factorial(k) for k in range(num_servers))))))

    return U,U_bar, P_N2, P_N_lt_5, Lq, R,num_servers

# Scenario 4: M/M/∞ Queue
def scenario_4(lambda_rate, mu_rate, n_max=10):
    U = lambda_rate / mu_rate
    P_N2 = (U**2 / math.factorial(2)) * math.exp(-U)
    P_N_lt_5 = sum([(U**n / math.factorial(n)) * math.exp(-U) for n in range(5)])
    R = 1 / mu_rate

    return U, P_N2, P_N_lt_5, R

# Example execution
def print_scenario_1_results(U, P_N2, P_N_lt_5, Lq, R, P_W_gt_2, W_95):
    print("\nScenario 1: M/M/1 Queue")
    print(f"Average Utilization: {U}")
    print(f"P(N=2): {P_N2}")
    print(f"P(N<5): {P_N_lt_5}")
    print(f"Average Queue Length (Lq): {Lq}")
    print(f"Average Response Time (R): {R}")
    print(f"P(W > 2): {P_W_gt_2}")
    print(f"95th Percentile of W: {W_95}")

def print_scenario_2_results(U,U_bar, P_N2, P_N_lt_5, Lq, R):
    print("\nScenario 2: M/M/2 Queue")
    print(f"Total Utilization: {U}")
    print(f"Average Utilization: {U_bar}")
    print(f"P(N=2): {P_N2}")
    print(f"P(N<5): {P_N_lt_5}")
    print(f"Average Queue Length (Lq): {Lq}")
    print(f"Average Response Time (R): {R}")

def print_scenario_3_results(U,U_bar, P_N2, P_N_lt_5, Lq, R,num_servers):
    print("\nScenario 3: M/M/c Queue")
    print(f"Total Utilization: {U}")
    print(f"Average Utilization: {U_bar}")
    print(f"P(N=2): {P_N2}")
    print(f"P(N<5): {P_N_lt_5}")
    print(f"Average Queue Length (Lq): {Lq}")
    print(f"Average Response Time (R): {R}")
    print(f"c:{num_servers}")

def print_scenario_4_results(U, P_N2, P_N_lt_5, R):
    print("\nScenario 4: M/M/∞ Queue")
    print(f"Total Utilization (U): {U}")
    print(f"P(N=2): {P_N2}")
    print(f"P(N<5): {P_N_lt_5}")
    print(f"Average Response Time (R): {R}")
   

# Execute and print results
print_scenario_1_results(*scenario_1(0.5, 0.625))
print_scenario_2_results(*scenario_2(1.0, 0.625, 2))
print_scenario_3_results(*scenario_3(4.0, 0.625, math.ceil(4.0 / 0.625)))
print_scenario_4_results(*scenario_4(10.0, 0.625))
