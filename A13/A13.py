import numpy as np

def compute_visits(P, reference_station):
    """
    Compute the visit vector v using the formula:
    v = l * (I - P0)^(-1)
    where P0 is P with the reference column zeroed.
    """
    num_stations = P.shape[0]
    l = np.zeros(num_stations)
    l[reference_station - 1] = 1  # Set reference station arrivals to 1
    
    # Zero out the reference column in P to create P0
    P0 = P.copy()
    P0[:, reference_station - 1] = 0

    # Compute visit vector v
    I = np.eye(num_stations)
    v = np.matmul(l, np.linalg.inv(I - P0))
    return v

def compute():
    # System parameters
    N = 100  # Number of users
    S1 = 40.0  # Think time (seconds)
    S2 = 0.050  # Application Server service time (seconds)
    S3 = 0.002  # Storage Controller service time (seconds)
    S4 = 0.080  # DBMS service time (seconds)
    S5 = 0.080  # Disk 1 service time (seconds)
    S6 = 0.100  # Disk 2 service time (seconds)

    # Routing probabilities
    P = np.zeros((6, 6))
    P[0, 1] = 1.0    
    P[1, 2] = 0.35   
    P[1, 3] = 0.6   
    P[1, 0] = 0.05   
    P[2, 4] = 0.65  
    P[2, 5] = 0.35  
    P[3, 1] = 1.0    
    P[4, 1] = 0.9    
    P[5, 1] = 0.9    
    P[4, 5] = 0.1    
    P[5, 4] = 0.1    

    # Compute visit ratios (v)
    v = compute_visits(P, reference_station=1)

    # Service demands (D = S * v)
    S = [0, S2, S3, S4, S5, S6]
    D = np.array([S[i] * v[i] for i in range(len(v))])

    # Initialize queue lengths
    Q = np.zeros(len(D))

    # Perform Mean Value Analysis iterations
    for n in range(1, N + 1):
        # Residence times (R_k = D_k * (1 + Q_k))
        R = D * (1 + Q)
        # System throughput (X)
        X = n / (S1 + np.sum(R))
        # Update queue lengths (Q_k = X * R_k)
        Q = X * R

    # System response time
    R_total = np.sum(R)

    R_ms=R_total * 1000  # Convert to milliseconds

    # Utilization (U_k = X * D_k)
    U = X * D

    # Convert demands to milliseconds
    D_ms = D * 1000

    # Print the requested results
    print("Results:")
    print(f"1. Demand of Disk 1: {D_ms[4]} ms, Demand of Disk 2: {D_ms[5]} ms")
    print(f"2. Throughput of the system (X): {X} requests/second")
    print(f"3. Average system response time (R): {R_ms} ms")
    print(f"4. Utilization:")
    print(f"   - Application Server: {U[1]}")
    print(f"   - DBMS: {U[3]}")
    print(f"   - Disk 1: {U[4]}")
    print(f"   - Disk 2: {U[5]}")
    print(f"5. Throughput of Disk 1: {X * v[4]} requests/second, Throughput of Disk 2: {X * v[5]} requests/second")

def run():
    compute()

if __name__ == '__main__':
    run()
