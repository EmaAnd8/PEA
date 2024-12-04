import math
from pint import UnitRegistry

# Initialize unit registry
ureg = UnitRegistry()


def mmc_p0(arrival_rate, service_rate, num_servers, max_customers):
    """
    Compute P0 for an M/M/c/K queue model.
    """
    rho = arrival_rate / (num_servers * service_rate)  # Traffic intensity
    
    # Compute the summation term: Σ from k = 0 to c-1
    summation = sum(
        (num_servers * rho)**k / math.factorial(k)
        for k in range(num_servers)
    )
    
    # Compute the term for c ≤ n ≤ K
    final_term = (
        ((num_servers * rho)**num_servers / math.factorial(num_servers))
        *((1 - rho**(max_customers - num_servers + 1)) / (1 - rho))
    )
    
    # Compute P0
    p0 = 1 / (summation + final_term)
    
    return p0


def mmc_probabilities(arrival_rate, service_rate, num_servers, max_customers):
    """
    Compute probabilities Pn for an M/M/c/K queue model.

    """
    rho = arrival_rate / (num_servers * service_rate)  # Traffic intensity
    p0 = mmc_p0(arrival_rate, service_rate, num_servers, max_customers)  # Compute P0
    
    probabilities = []  # Store probabilities Pn
    
    # Case 1: n < c
    for n in range(num_servers):
        pn = (p0 * (num_servers * rho)**n) / math.factorial(n)
        probabilities.append(pn)
    
    # Case 2: c <= n <= K
    for n in range(num_servers, max_customers+1):
        pn = ((p0 * (num_servers**num_servers) * (rho)**n) / (
            math.factorial(num_servers) 
        ))
        probabilities.append(pn)
    
    return probabilities



def is_stable(arrival_rate, mu_rate, num_servers):
    #we check if the system is stable
    rho = arrival_rate / (num_servers * mu_rate)
    return rho < 1

def find_min_servers_stable(lambda_rate, mu_rate, capacity, target_loss_prob):
 
    num_servers = 1  # Start with one server
    while True:
        if not is_stable(lambda_rate, mu_rate, num_servers):
            num_servers += 1
            continue
        p_n_vector = mmc_probabilities(lambda_rate, mu_rate, num_servers, capacity)
        p_loss = p_n_vector[capacity]  # Loss probability
        if p_loss < target_loss_prob:
            break
        num_servers += 1  # Increment servers until condition is met

    return num_servers        


# Scenario 1: M/M/1/K
def scenario_1(lambda_rate, mu_rate,K):
   
    rho = lambda_rate / mu_rate
    U = ((rho-(rho**(K+1)))/(1-(rho**(K+1))))
    Lq = (rho**2 / (1 - rho)) * ((1 - rho**(K + 1) * (K + 1 - K * rho)) / (1 - rho**(K + 1)))
    Pl = rho**K * (1 - rho) / (1 - rho**(K + 1))
    N = (rho / (1 - rho)) - (((K + 1) * (rho ** (K + 1))) / (1 - rho ** (K + 1)))
    Dr = lambda_rate * Pl
    R =  ((1 / mu_rate) * ( (1 - ((K + 1) *( rho**K ))+( K * rho**(K + 1))) / ((1 - rho) * (1 - (rho**K))))) 
    D=(1/mu_rate)
    Theta=(R-D)
  
    return rho,U, Lq, R, Pl, Dr,Theta,N

# Scenario 2: M/M/2/K
def scenario_2(lambda_rate, mu_rate, num_servers, K):
    p_n_vector = mmc_probabilities(lambda_rate, mu_rate, num_servers, K)
    p_loss = p_n_vector[K]
    rho = lambda_rate / (num_servers * mu_rate)
    U = sum(i * p_n_vector[i] for i in range(1, num_servers+1)) + num_servers * sum(p_n_vector[i] for i in range(num_servers + 1, K +1))
    U_bar =U/num_servers
    N = sum(n * p_n_vector[n] for n in range(K + 1))
    Dr = lambda_rate * p_n_vector[K]
    R = N / (lambda_rate * (1 - p_n_vector[K])) if (1 - p_n_vector[K]) > 0 else float('inf')
    D=(1/mu_rate)
    Theta=(R-D)
    
    return rho,U, U_bar, R, p_loss, Dr,Theta,N

# Scenario 3: M/M/c/K
def scenario_3(lambda_rate, mu_rate, num_servers, K):
    p_n_vector = mmc_probabilities(lambda_rate, mu_rate, num_servers, K)
    p_loss = p_n_vector[K]
    rho = lambda_rate / (num_servers * mu_rate)
    U = sum(i * p_n_vector[i] for i in range(1, num_servers + 1)) + num_servers * sum(p_n_vector[i] for i in range(num_servers + 1, K + 1))
    U_bar = U/num_servers
    N = sum(n * p_n_vector[n] for n in range(K + 1))
    Dr = lambda_rate * p_loss
    R = N / (lambda_rate * (1 - p_loss)) if (1 - p_loss) > 0 else float('inf')
    D=(1/mu_rate)
    Theta=(R-D)
    return rho,U, U_bar, R, p_loss, Dr,Theta,N

# Print results for Scenario 1
def print_scenario_1_results(rho,U, Lq, R, Pl, Dr,Theta,N):
   
    print("\nScenario 1: M/M/1/K Queue")
    print(f"Utilization: {U}")
    print(f"Average Queue Length (Lq): {Lq}")
    # Convert R to milliseconds
    R=R*ureg.seconds    
    R_ms = R.to(ureg.millisecond)
    print(f"Average Response Time : {R_ms}")
    print(f"Loss Probability (Pl): {Pl}")
    # Convert Dr to minutes
    Dr_min=Dr*60
   
    # Print the result
    print(f"Drop Rate : {Dr_min}")
      # Convert Theta to milliseconds
    Theta=Theta*ureg.seconds   
    Theta_ms = Theta.to(ureg.millisecond)
    # Print the result
    print(f"Average Time spent in the queue: {Theta_ms}")
    print(f"Average  number of jobs: {N}")

# Print results for Scenario 2
def print_scenario_2_results(rho,U, U_bar, R, Pl, Dr,Theta,N):
    print("\nScenario 2: M/M/2/K Queue")
    print(f"Total Utilization: {U}")
    print(f"Average Utilization: {U_bar}")
    print(f"Average number of Jobs : {N}")
    R=R*ureg.seconds
        # Convert R to milliseconds
    R_ms = R.to(ureg.millisecond)
    print(f"Average Response Time : {R_ms}")
    print(f"Loss Probability (Pl): {Pl}")
    # Convert Dr to minutes
   
    Dr_min=Dr*60
    # Print the result
    print(f"Drop Rate : {Dr_min}")
    # Convert Theta to milliseconds
    Theta=Theta*ureg.seconds 
    Theta_ms = Theta.to(ureg.millisecond)
    # Print the result
    print(f"Average Time spent in the queue: {Theta_ms}")


  

# Print results for Scenario 3
def print_scenario_3_results(rho,U, U_bar, R, Pl, Dr,Theta,N):
    print("\nScenario 3: M/M/c/K Queue")
    print(f"Total Utilization: {U}")
    print(f"Average Utilization: {U_bar}")
    print(f"Average number of Jobs : {N}")
    R=R*ureg.seconds
    # Convert R to milliseconds
    R_ms = R.to(ureg.millisecond)
    print(f"Average Response Time : {R_ms}")
    print(f"Loss Probability (Pl): {Pl}")
    # Convert Dr to minutes
   
   
    Dr_min=Dr*60
    # Print the result
    print(f"Drop Rate (Dr): {Dr_min}")
      # Convert Theta to milliseconds
    Theta=Theta*ureg.seconds  
    Theta_ms = Theta.to(ureg.millisecond)
    # Print the result
    print(f"Average Time spent in the queue: {Theta_ms}")
    print(f"c:{find_min_servers_stable(960 / 60, 1 / 0.2, 16, 0.01)}")



# Execute and print results for the three scenarios
print_scenario_1_results(*scenario_1(240 / 60, 1 / 0.2,16))  # Scenario 1: M/M/1/K
print_scenario_2_results(*scenario_2(360 / 60, 1 / 0.2, 2, 16))  # Scenario 2: M/M/2/K
print_scenario_3_results(*scenario_3(960 / 60, 1 / 0.2, find_min_servers_stable(960 / 60, 1 / 0.2, 16, 0.01), 16))  # Scenario 3: M/M/c/K (4 servers)
