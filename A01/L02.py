import numpy as np
import matplotlib.pyplot as plt

# Load the arrival times from Logger1 and departure times from Logger2
arrival_times = np.loadtxt("Logger1.csv", delimiter=";")
departure_times = np.loadtxt("Logger2.csv", delimiter=";")

A_T= len(arrival_times)
C_T=len(departure_times)



# Ensure that the array is not empty
if arrival_times.size > 0:
    first_element = arrival_times[0]      # First element
    last_element = departure_times[-1]      # Last element
    T=last_element-first_element
   
else:
    print("The data array is empty.")

  # Compute busy time
idle_time = first_element
for i in range(0, len(arrival_times)-1):
     if (arrival_times[i+1] > departure_times[i]):
            idle_time += arrival_times[i+1] - departure_times[i]
B= last_element - idle_time




lambda_=A_T/T



X = len(departure_times) / T



U=B/T


S=U/X


r_i=0
#compute r_i
for i in range(0,len(departure_times)):
     r_i=r_i+(departure_times[i]-arrival_times[i])



R=r_i/C_T
N=X*R

print("T =", T)
print("X =", X)
print("B =", B)
print("U =", U)
print("S =", S)
print("R =", R)
print("N =", N)
print("lambda =", lambda_)





import numpy as np
import matplotlib.pyplot as plt

# Assuming arrival_times and departure_times are already loaded
# Create an event list for arrivals (+1) and departures (-1)
arrival_events = np.column_stack((arrival_times, np.ones_like(arrival_times)))  # Arrival adds 1 car
departure_events = np.column_stack((departure_times, -1 * np.ones_like(departure_times)))  # Departure removes 1 car

# Combine and sort events by time
events = np.vstack((arrival_events, departure_events))
events = events[events[:, 0].argsort()]  # Sort by event times

# Initialize variables to track the number of cars in the system and their durations
cars_in_system = 0
prev_time = 0
T = events[-1, 0] - events[0, 0]  # Total observation period
max_cars = 25
Y_m = np.zeros(max_cars + 1)  # Array to store total time with m cars

# Track the time spent with each number of cars in the system
for event in events:
    current_time = event[0]
    # Calculate the duration since the last event
    duration = current_time - prev_time
    # Record the time spent with the current number of cars
    if 0 <= cars_in_system <= max_cars:
        Y_m[cars_in_system] += duration
    # Update the number of cars in the system
    cars_in_system += int(event[1])
    prev_time = current_time

# Calculate the probability p(N = m)
p_N_equals_m = Y_m / T

# Plot the probability distribution of the number of cars in the system
plt.figure(figsize=(10, 6))
plt.bar(range(max_cars + 1), p_N_equals_m, color='blue', edgecolor='black')
plt.xlabel('Number of Cars in the Road Segment')
plt.ylabel('Probability')
plt.title('Probability Distribution of Number of Cars in the Road Segment (0-25 cars)')
plt.xticks(range(0, max_cars + 1))
plt.grid(True)
plt.show()






#we have already computed r_i



# Assuming arrival_times and departure_times are already loaded
# Initialize the cumulative response time variable
cumulative_r_i = 0

# Initialize an empty list to store cumulative response times
cumulative_response_times = []

# Compute cumulative response times for each car
for i in range(len(departure_times)):
    r_i = departure_times[i] - arrival_times[i]
    cumulative_r_i += r_i  # Add current response time to the cumulative sum
    cumulative_response_times.append(cumulative_r_i)

# Convert the list to a numpy array for easy processing
cumulative_response_times = np.array(cumulative_response_times)

# Filter cumulative response times between 1 and 40 minutes
cumulative_response_times_filtered = cumulative_response_times[(cumulative_response_times >= 1) & (cumulative_response_times <= 40)]

# Sort the filtered cumulative response times
sorted_cumulative_response_times = np.sort(cumulative_response_times_filtered)

# Calculate the cumulative distribution values
cdf_r = np.arange(1, len(sorted_cumulative_response_times) + 1) / len(sorted_cumulative_response_times)

# Plot the CDF with a granularity of 0.1 minutes
plt.figure(figsize=(10, 6))
plt.plot(sorted_cumulative_response_times, cdf_r, marker='o', linestyle='-', color='blue', linewidth=0.5, markersize=1, label='Response Time')


# Set the y-axis ticks with a granularity of 0.1
plt.xticks(np.arange(1,40 , 1))  # Set ticks from 0 to 1, step by 0.1

plt.xlabel(' Time (minutes)')
plt.ylabel('Cumulative Probability')
plt.title('Response Time Distribution (between 1 and 40 minutes, with a granularity of  1 min.)')
plt.grid(True)
plt.legend()
plt.show()





# Assuming service_times is already computed
service_times = departure_times - arrival_times

# Filter service times between 0.1 and 5 minutes
service_times_filtered = service_times[(service_times >= 0.1) & (service_times <= 5)]

# Sort the service times
sorted_service_times = np.sort(service_times_filtered)

# Calculate the cumulative distribution values
cdf = np.arange(1, len(sorted_service_times) + 1) / len(sorted_service_times)

# Plot the CDF with a granularity of 0.1 minutes
plt.figure(figsize=(10, 6))
plt.plot(sorted_service_times, cdf, marker='o', linestyle='-', color='blue', linewidth=0.5, markersize=1, label='Service Time')


# Set the y-axis ticks with a granularity of 0.1
plt.xticks(np.arange(0.1, 5, 0.1))  # Set ticks from 0 to 1, step by 0.1

plt.xlabel(' Time (minutes)')
plt.ylabel('Cumulative Probability')
plt.title('Service Time Distribution (between 0.1 and 5 minutes, with a granularity of 0.1 min.)')
plt.grid(True)
plt.legend()
plt.show()
