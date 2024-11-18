import numpy as np
import pandas as pd
from scipy.linalg import null_space

# Step 1: Import the data from a CSV file this time using pandas
file_path = "Data.csv"  # Replace with the actual file path

songs_data_df = pd.read_csv(file_path, delimiter=";", header=None)



songs_data_n = songs_data_df.to_numpy()
songs_data = np.array(songs_data_n)
# Define the CTMC rate matrix Q with 20 states (Play and Extended for each song)
n_states = 20
Q = np.zeros((n_states, n_states))

# Populate the rate matrix Q according to the transition rules for each song
for i in range(10):
    play_state = 2 * i        # Index for Play state of song i+1
    extended_state = 2 * i + 1  # Index for Extended state of song i+1
    
    # Extract song data
    length, extension_prob, skip_prob, extension_len, skip_extended, royalty_fee = songs_data[i]
    
    # Convert probabilities to rates assuming exponential durations with the provided mean times
    base_rate = 1 / length  # Base rate for transitioning from Play state
    extension_rate = extension_prob / 100 * base_rate
    skip_rate = skip_prob / 100 * base_rate
    continue_rate = (1 - extension_prob / 100 - skip_prob / 100) * base_rate  # Assumed probability

    # For the extended state
    extended_base_rate = 1 / extension_len if extension_len > 0 else 0
    extended_continue_rate = (1 - skip_extended / 100) * extended_base_rate
    extended_skip_rate = skip_extended / 100 * extended_base_rate

    # Populate rates for Play state
    Q[play_state, extended_state] = extension_rate

    if i < 9:
        Q[play_state, play_state + 2] = continue_rate
    else:
        Q[play_state, 0] = continue_rate  # Wrap to first song if at last song

    # Skip transition for Play state, only if within bounds
    if play_state + 4 < n_states:
        Q[play_state, play_state + 4] = skip_rate

    # Set the diagonal rate for Play state
    Q[play_state, play_state] = -(extension_rate + continue_rate + skip_rate)

    # Populate rates for Extended state
    if i < 9:
        Q[extended_state, play_state + 2] = extended_continue_rate
    else:
        Q[extended_state, 0] = extended_continue_rate  # Wrap to first song if at last song

    # Skip transition for Extended state, only if within bounds
    if extended_state + 4 < n_states:
        Q[extended_state, extended_state + 3] = extended_skip_rate

    # Set the diagonal rate for Extended state
    Q[extended_state, extended_state] = -(extended_continue_rate + extended_skip_rate)

# Step 2: Solve for steady-state distribution π such that πQ = 0 and sum(π) = 1
# Use the null space method to find the steady-state distributions
null_space_Q = null_space(Q.T)

# The null space provides the steady-state vector; normalize it so that it sums to 1
steady_state = null_space_Q[:, 0] / np.sum(null_space_Q[:, 0])


# Step 3: Calculate Metrics

#  Probabilities for specified songs
song_probabilities = {f'Song {i+1}': steady_state[2*i] + steady_state[2*i + 1] for i in [0, 1, 4, 8, 9]}

#  Average royalty cost per concert
royalty_fees = songs_data[:, 5]
average_cost = np.dot(steady_state[::2] + steady_state[1::2], royalty_fees)





# Define a matrix xi0
xi0 = np.zeros((n_states, n_states)) 
# Set the entry for transition from state 10 to state 1 to 1
xi0[18,0]=1
xi0[19, 0] = 1 # Using zero-based indexing for states

# Calculate the number of shows per second 
number_show_per_seconds = (((Q*xi0) @ np.ones(20)) @  steady_state)
num_show_per_day = (number_show_per_seconds) * 60 * 60 * 24

# Convert to minutes 
average_duration_minutes = 1 /(number_show_per_seconds * 60)



# Print Results 
print("Probability of hearing each song (Songs 1, 2, 5, 9, 10):")
for song, probability in song_probabilities.items():
    print(f"  {song}: {probability:.4f}")  # Four decimal places for probability

print(f"Average royalty cost per concert: €{average_cost:.2f}")
print(f"Average duration of concert in minutes: {average_duration_minutes:.2f}")
print(f"Number of shows that can be played per day: {num_show_per_day:.2f}")

