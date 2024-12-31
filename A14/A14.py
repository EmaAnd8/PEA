import numpy as np

# Service Times (min)
Sa = np.array([8, 10])  # Service times for Class A
Sb = np.array([3, 2])   # Service times for Class B
Sc = np.array([4,7])   # Service times for Class C

# Lambdas (parts / min)
lka = np.array([1.5, 0]) / 60  # Arrival rates for Class A
lkb = np.array([2.5, 0]) / 60  # Arrival rates for Class B
lkc = np.array([2, 0]) / 60  # Arrival rates for Class C

la = np.sum(lka)
lb = np.sum(lkb)
lc = np.sum(lkc)


# Routing probability matrices for each class
P_A = np.array([[0, 1],
                 [0.1, 0]])

P_B = np.array([[0, 1],
                 [0.08, 0]])

P_C = np.array([[0, 1],
                 [0.12, 0]])

Xa = la
Xb = lb
Xc = lc

X = Xa + Xb + Xc  # Total Throughput  

vka = np.matmul(lka / la, np.linalg.inv(np.eye(2) - P_A))  # Visit ratios for Class A
vkb = np.matmul(lkb / lb, np.linalg.inv(np.eye(2) - P_B))  # Visit ratios for Class B
vkc = np.matmul(lkc / lc, np.linalg.inv(np.eye(2) - P_C))  # Visit ratios for Class C

Dka = vka * Sa  # Service demands for Class A
Dkb = vkb * Sb  # Service demands for Class B
Dkc = vkc * Sc  # Service demands for Class C

# 1. The utilization of the two stations U
Uka = la * Dka
Ukb = lb * Dkb
Ukc = lc * Dkc

Uk = np.array([Uka[0] + Ukb[0] + Ukc[0], Uka[1] + Ukb[1] + Ukc[1]])

Nka = Uka / (1 - Uk)
Nkb = Ukb / (1 - Uk)
Nkc = Ukc / (1 - Uk)

# 2. The average number of jobs in the system for each type of product (class c - Nc)
Na = np.sum(Nka)
Nb = np.sum(Nkb)
Nc = np.sum(Nkc)

# 3. The average system response time per product type (class c - Rc)
Ra = Na / la 
Rb = Nb / lb 
Rc = Nc / lc 

# 4. The class-independent average system response time (R)
l_tot = la + lb + lc
weights = np.array([la, lb, lc]) / l_tot
R_classes = np.array([Ra, Rb, Rc])

R = np.sum(R_classes * weights)

# Total number of jobs in the system (N)
N = Na + Nb + Nc




print(f"Utilization Production: {Uk[0]}")
print(f"Utilization Packaging: {Uk[1]}")
print(f"Average Number of Class A in the system: {Na}")
print(f"Average Number of Class B in the system: {Nb}")
print(f"Average Number of Class C in the system: {Nc}")
print(f"Average System Response Time Class A: {Ra}")
print(f"Average System Response Time Class B: {Rb}")
print(f"Average System Response Time Class C: {Rc}")
print(f"The class-independent  Jobs : {N}")
print(f"The class-independent average system response time: {R}")
