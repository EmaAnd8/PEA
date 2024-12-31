import numpy as np





# System Response Time (B) values for each class
Ra = 535.1999478710144  # Response Time for Class A
Rb =540.6889546546512 # Response Time for Class B
Rc = 243.0486943000801 # Response Time for Class C

# Separate throughput by class (A, B, C)
Xa = 0.036683789274190616

Xb = 0.005528028485321022 # Throughput for Class B
Xc = 0.3333 # Throughput for Class C

X_total = Xa + Xb+ Xc

# Compute class-independent average system response time (R)
R = ((Xa / X_total) * Ra) + (((Xb) / X_total) * Rb) + (((Xc)/ X_total) * Rc)


print(f"Class-independent Average System Response Time (R): {R:.2f}")
