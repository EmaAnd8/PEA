import numpy as np



# Define the arrival rates in input
lambda_in_1 = 2.5
lambda_in_2 = 2.0

# Compute total system arrival rate
l_0 = lambda_in_1 + lambda_in_2

# Initialize a 4x4 transition probability matrix
P = np.zeros((4, 4))


# Assign transition probabilities
P[0, 1] = 0.7 
P[1, 2] = 0.25
P[2,1]=P[3,1]=1
 
P[1, 3] = 0.45 


#print(P)
# Input arrival rates as a vector
lambda_in = np.array([lambda_in_1, lambda_in_2])




# Compute the relative arrival rates (l_i)
l_i = lambda_in / l_0

# Assign the relative arrival rates to the vector `l`
l = np.array([l_i[0], l_i[1], 0, 0])

# Compute the visits vector
v = np.matmul(l, np.linalg.matrix_power(np.identity(4) - P, -1))

# Define service demands for each station
s = np.zeros((4))
s[0] = 2.0  
s[1] = 0.02  
s[2] = 0.1   
s[3] = 0.07

# Compute service demands 
d = np.multiply(v, s)

# Compute utilizations for each server
u = np.zeros((4))
u[0] = d[0] * l_0
u[1] = d[1] * l_0
u[2] = d[2] * l_0
u[3] = d[3] * l_0

print(u)

# Compute average system response time using Little's Law
n0 = u[0] 
n1 = u[1] / (1 - u[1]) 
n2 = u[2] / (1 - u[2]) 
n3 = u[3] / (1 - u[3]) 
N= n0 + n1 + n2 + n3

# Compute system saturation throughput
l_sat = 1 / np.max(d[1:])

X=l_0

# Compute average number of jobs in the system using Little's Law
R=N/X


# Print  the  results
def print_results(X, v, s, d, u, r, l_sat, N):
    
    # 1. The visits of the application server, the storage and the DBMS
    # 2. The throughput of the system (X)
    # 3. The average number of jobs in the system (N)
    # 4. The average system response time (R)
    # 5. The maximum arrival rate the system would be able to handle (l_sat)
    
    # According to the indexing:
    # [1] Self-check = index 0
    # [2] Application server = index 1
    # [3] Storage = index 2
    # [4] DBMS = index 3
    
    visits_app_server = v[1]
    visits_storage = v[2]
    visits_dbms = v[3]
    r_ms=r*1000
    print("=== Requested Results ===")
    print(f"Visits of the application server (V_app): {visits_app_server}")
    print(f"Visits of the storage (V_storage): {visits_storage}")
    print(f"Visits of the DBMS (V_dbms): {visits_dbms}")
    print(f"System Throughput (X): {X}")
    print(f"Average Number of Jobs in the System (N): {N}")
    print(f"Average Response Time (R): {r_ms}")
    print(f"Maximum Arrival Rate System Can Handle (l_sat): {l_sat}")
    print("=========================")

# Call the function to print results
print_results(X=X, v=v, s=s, d=d, u=u,r=R, l_sat=l_sat, N=N)
