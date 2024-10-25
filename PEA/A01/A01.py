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

A_bar=1/lambda_



X = len(departure_times) / T



U=B/T


S=U/X


cumulative_r_i=0
#compute r_i
for i in range(0,len(departure_times)):
     cumulative_r_i=cumulative_r_i+(departure_times[i]-arrival_times[i])



R=cumulative_r_i/C_T
N=X*R

print("T =", T)
print("X =", X)
print("B =", B)
print("U =", U)
print("S =", S)
print("R =", R)
print("N =", N)
print("Abar=",A_bar)
print("lambda =", lambda_)






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


# Initialize an empty list to store cumulative response times
cumulative_response_times = []

# Compute cumulative response times for each car
for i in range(len(departure_times)):
    r_i = departure_times[i] - arrival_times[i]
    cumulative_response_times.append(r_i)

# Convert the list to a numpy array for easy processing
cumulative_response_times = np.array(cumulative_response_times)

# Filter cumulative response times between 1 and 40 minutes
cumulative_response_times_filtered = cumulative_response_times[(cumulative_response_times >= 1) & (cumulative_response_times <= 40)]

# Sort the filtered cumulative response times
sorted_cumulative_response_times = np.sort(cumulative_response_times_filtered)


#y axis
y_axis = [0] * 41
#x axis
response_time = [0] * 41


for i in range(0, 41):
    response_time[i] = i
    y_axis[i] = sum(cumulative_response_times< i) / A_T

plt.plot(response_time, y_axis, label='Response Time', color='b')
plt.title('Response Time Distribution (between 1 and 40 minutes, with a granularity of 1 min.)')
plt.grid(True)
plt.legend()
plt.show()

'''
# Calculate the cumulative distribution values
cdf_r = np.arange(1, len(sorted_cumulative_response_times) + 1) / len(sorted_cumulative_response_times)

# Plot the CDF with a granularity of 0.1 minutes
plt.figure(figsize=(10, 6))
plt.plot(sorted_cumulative_response_times, cdf_r, marker='o', linestyle='-', color='blue', linewidth=0.5, markersize=1, label='Response Time')


# Set the y-axis ticks with a granularity of 0.1
plt.xticks(np.arange(1,41 , 1))  # Set ticks from 0 to 1, step by 0.1

plt.xlabel(' Time (minutes)')
plt.ylabel('Cumulative Probability')
plt.title('Response Time Distribution (between 1 and 40 minutes, with a granularity of  1 min.)')
plt.grid(True)
plt.legend()
plt.show()

'''



# Assuming service_times is already computed
service_times = [departure_times[0] - arrival_times[0]] + [departure_times[i] - np.maximum(departure_times[i - 1], arrival_times[i]) for i in range(1, len(arrival_times))]

service_times=np.array(service_times)
# Filter service times between 0.1 and 5 minutes
service_times_filtered = service_times[(service_times >= 0.1) & (service_times <= 5)]

# Sort the service times
sorted_service_times = np.sort(service_times_filtered)
'''
# Calculate the cumulative distribution values
cdf = np.arange(1, len(sorted_service_times) + 1) / len(sorted_service_times)

# Plot the CDF with a granularity of 0.1 minutes
plt.figure(figsize=(10, 6))
plt.plot(sorted_service_times, cdf, marker='o', linestyle='-', color='blue', linewidth=0.5, markersize=1, label='Service Time')


# Set the y-axis ticks with a granularity of 0.1
plt.xticks(np.arange(0.1, 5.1, 0.1))  # Set ticks from 0 to 1, step by 0.1

plt.xlabel(' Time (minutes)')
plt.ylabel('Cumulative Probability')
plt.title('Service Time Distribution (between 0.1 and 5 minutes, with a granularity of 0.1 min.)')
plt.grid(True)
plt.legend()
plt.show()
'''


#Time range
time_range = (0.1, 5)
#Granularity
granularity = 0.1


#y axis
y_axis = [0] * 51
#x axis
service_time = [0] * 51


for i in range(0, 51):
    t = i / 10
    service_time[i] = t
    y_axis[i] = sum(service_times < t) / A_T

plt.plot(service_time, y_axis, label='Service Time', color='b')
plt.title('Service Time Distribution (between 0.1 and 5 minutes, with a granularity of 0.1 minutes).')
plt.grid(True)
plt.legend()
plt.show()


