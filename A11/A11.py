import math
import numpy as np






def is_stable(arrival_rate, mu_rate, num_servers):
    #we check if the system is stable
    rho = arrival_rate / (num_servers * mu_rate)
    return rho < 1

def find_min_servers_stable(lambda_rate, mu_rate):
 
    num_servers = 1  # Start with one server
    while True:
        if not is_stable(lambda_rate, mu_rate, num_servers):
            num_servers += 1
        else:
            break    
          
      
       

    return num_servers        


# Scenario 1: M/G/1 Queue
def scenario_1(lambda_rate, lambda_erlang,k):
    mean_erlang = k / lambda_erlang
    D=mean_erlang
    U=lambda_rate*D
    rho=lambda_rate*D
    Cv=1/(np.sqrt(k))
    m2=(D**2)*(1+Cv**2)
    R=D+(((lambda_rate)*m2)/(2*(1-rho)))
    N=rho+(((lambda_rate**2)*m2)/(2*(1-rho)))

    return U,R,N
    

# Scenario 2: G/G/c Queue
def scenario_2(l1,l2,p1,lambda_erlang,k):
    p2=1-p1
    mean_erlang = k / lambda_erlang
    D=mean_erlang
    mean_hyper = (p1 / l1) + (p2 / l2)
    lambda_rate=1/mean_hyper
    mu_rate=1/D
    c=find_min_servers_stable(lambda_rate,mu_rate)
    rho=(lambda_rate*D)/c
    U=rho
    
  
    # Compute E[X^2] (second moment of arrival time)
    second_moment_arrival = p1 * (2 / (l1**2)) + p2 * (2 /(l2**2))

    # Compute Variance of arrival time
    variance_arrival_time = second_moment_arrival - mean_hyper**2

   
    Ca = np.sqrt(variance_arrival_time) / mean_hyper
    Cv=1/(np.sqrt(k))
    R=D+((((Ca**2)+(Cv**2))/(2))*(((rho**c)*D)/(1-(rho**c))))
    N=lambda_rate*R

    return c,U,R,N



# Example execution
def print_scenario_1_results(U,R,N):
    print("\nScenario 1: G/G/1 Queue")
    print(f"Average Utilization: {U}")
    R_ms=R*1000
    print(f"Average (Exact) Response Time : {R_ms}")
    print(f"Average (Exact) Number of Jobs : {N}")
 

def print_scenario_2_results(c,U_bar,R,N):
    print("\nScenario 2: G/G/c Queue")
    print(f"The minimum number of servers c for which the considered system is stable:{c}")
    print(f"Average Utilization: {U_bar}")
    R_ms=R*1000
    print(f"Approximate average   Response Time (R): {R_ms}")
    print(f"Approximate average Number of Jobs (N): {N}")


   

# Execute and print results
print_scenario_1_results(*scenario_1(20, 100,4))
print_scenario_2_results(*scenario_2(40, 240,0.8,100,4))





