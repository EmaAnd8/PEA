import numpy as np
from collections import deque
import pandas as pd

# Step 1: Import the data from a CSV file this time using pandas
file_path = "Data.csv"  # Replace with the actual file path

data = pd.read_csv(file_path, delimiter=";", header=None)



songs_data_n = data.to_numpy()
songs_data = np.array(songs_data_n)
# Define the CTMC rate matrix Q with 20


song_lengths = data[0].values
extension_probs = data[1].values/100
skip_probs = data[2].values/100
extension_lengths = data[3].values
skip_extended_probs = data[4].values/100
royalties = data[5].values
royalties_doubled = np.zeros(2 * len(royalties))
for i in range(len(royalties)):
    royalties_doubled[i] = royalties[i]
    royalties_doubled[i + 1] = royalties[i]

Q = np.array([ [-1/song_lengths[0], 1/song_lengths[0]*extension_probs[0], 1/song_lengths[0]*(1-extension_probs[0]-skip_probs[0]), 0, 1/song_lengths[0]*skip_probs[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1/extension_lengths[0], 1/extension_lengths[0]*(1-skip_extended_probs[0]), 0, 1/extension_lengths[0]*skip_extended_probs[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1/song_lengths[1], 1/song_lengths[1]*extension_probs[1], 1/song_lengths[1]*(1-extension_probs[1]-skip_probs[1]), 0, 1/song_lengths[1]*skip_probs[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1/extension_lengths[1], 1/extension_lengths[1]*(1-skip_extended_probs[1]), 0, 1/extension_lengths[1]*skip_extended_probs[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1/song_lengths[2], 1/song_lengths[2]*extension_probs[2], 1/song_lengths[2]*(1-extension_probs[2]-skip_probs[2]), 0, 1/song_lengths[2]*skip_probs[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1/extension_lengths[2], 1/extension_lengths[2]*(1-skip_extended_probs[2]), 0, 1/extension_lengths[2]*skip_extended_probs[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1/song_lengths[3], 1/song_lengths[3]*extension_probs[3], 1/song_lengths[3]*(1-extension_probs[3]-skip_probs[3]), 0, 1/song_lengths[3]*skip_probs[3], 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1/extension_lengths[3], 1/extension_lengths[3]*(1-skip_extended_probs[3]), 0, 1/extension_lengths[3]*skip_extended_probs[3], 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -1/song_lengths[4], 1/song_lengths[4]*extension_probs[4], 1/song_lengths[4]*(1-extension_probs[4]-skip_probs[4]), 0, 1/song_lengths[4]*skip_probs[4], 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -1/extension_lengths[4], 1/extension_lengths[4]*(1-skip_extended_probs[4]), 0, 1/extension_lengths[4]*skip_extended_probs[4], 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/song_lengths[5], 1/song_lengths[5]*extension_probs[5], 1/song_lengths[5]*(1-extension_probs[5]-skip_probs[5]), 0, 1/song_lengths[5]*skip_probs[5], 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/extension_lengths[5], 1/extension_lengths[5]*(1-skip_extended_probs[5]), 0, 1/extension_lengths[5]*skip_extended_probs[5], 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/song_lengths[6], 1/song_lengths[6]*extension_probs[6], 1/song_lengths[6]*(1-extension_probs[6]-skip_probs[6]), 0, 1/song_lengths[6]*skip_probs[6], 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/extension_lengths[6], 1/extension_lengths[6]*(1-skip_extended_probs[6]), 0, 1/extension_lengths[6]*skip_extended_probs[6], 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/song_lengths[7], 1/song_lengths[7]*extension_probs[7], 1/song_lengths[7]*(1-extension_probs[7]-skip_probs[7]), 0, 1/song_lengths[7]*skip_probs[7], 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/extension_lengths[7], 1/extension_lengths[7]*(1-skip_extended_probs[7]), 0, 1/extension_lengths[7]*skip_extended_probs[7], 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/song_lengths[8], 1/song_lengths[8]*extension_probs[8], 1/song_lengths[8]*(1-extension_probs[8]-skip_probs[8]), 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/extension_lengths[8], 1/extension_lengths[8]*(1-skip_extended_probs[8]), 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/song_lengths[9], 1/song_lengths[9]*extension_probs[9]],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1/extension_lengths[9]]
    ])

# for i in range(n):
#     # Riga associata a song_lengths[i]
#     row_song = [0] * (2 * n)  # Crea una riga di zeri
#     row_song[2 * i] = -1 / song_lengths[i]

#     # Prima canzone: non può essere saltata
#     if i == 0:
#         if 2 * i + 1 < 2 * n:
#             row_song[2 * i + 1] = 1 / song_lengths[i] * extension_probs[i]
#         if 2 * i + 2 < 2 * n:
#             row_song[2 * i + 2] = 1 / song_lengths[i] * (1 - extension_probs[i])
#     # Ultima canzone: non può essere saltata
#     elif i == n - 1:
#         if 2 * i + 1 < 2 * n:
#             row_song[2 * i + 1] = 1 / song_lengths[i] * extension_probs[i]
#         if 2 * i + 2 < 2 * n:
#             row_song[2 * i + 2] = 1 / song_lengths[i] * (1 - extension_probs[i])
#     # Canzoni intermedie
#     else:
#         if 2 * i + 1 < 2 * n:
#             row_song[2 * i + 1] = 1 / song_lengths[i] * extension_probs[i]
#         if 2 * i + 2 < 2 * n:
#             row_song[2 * i + 2] = 1 / song_lengths[i] * (1 - extension_probs[i] - skip_probs[i])
#         if 2 * i + 4 < 2 * n:
#             row_song[2 * i + 4] = 1 / song_lengths[i] * skip_probs[i]
#     Q.append(row_song)

#     # Riga associata a extension_lengths[i]
#     row_extension = [0] * (2 * n)  # Crea una riga di zeri
#     row_extension[2 * i + 1] = -1 / extension_lengths[i]

#     # Prima canzone: non può essere saltata dopo estensione
#     if i == 0:
#         if 2 * i + 2 < 2 * n:
#             row_extension[2 * i + 2] = 1 / extension_lengths[i] * (1 - skip_extended_probs[i])
#     # Ultima canzone: non può essere saltata dopo estensione
#     elif i == n - 1:
#         if 2 * i + 2 < 2 * n:
#             row_extension[2 * i + 2] = 1 / extension_lengths[i] * (1 - skip_extended_probs[i])
#     # Canzoni intermedie
#     else:
#         if 2 * i + 2 < 2 * n:
#             row_extension[2 * i + 2] = 1 / extension_lengths[i] * (1 - skip_extended_probs[i])
#         if 2 * i + 4 < 2 * n:
#             row_extension[2 * i + 4] = 1 / extension_lengths[i] * skip_extended_probs[i]
#     Q.append(row_extension)

# # Converti in matrice numpy se necessario
# import numpy as np
# Q = np.array(Q)
Q_copy = Q.copy()

Q[:,0] = np.ones(20)
u = np.zeros(20)
u[0] = 1

pi = np.linalg.solve(Q.T, u)

print("Probability of a patron entering the concert and earing song 1:", pi[0]+pi[1])
print("Probability of a patron entering the concert and earing song 2:", pi[2]+pi[3])
print("Probability of a patron entering the concert and earing song 5:", pi[8]+pi[9])
print("Probability of a patron entering the concert and earing song 9:", pi[16]+pi[17])
print("Probability of a patron entering the concert and earing song 10:", pi[18]+pi[19])
#  Average royalty cost per concert
royalty_fees = songs_data[:, 5]
average_cost = np.dot(pi[::2] + pi[1::2], royalty_fees)





# Define a matrix xi0
xi0 = np.zeros((20, 20)) 
# Set the entry for transition from state 10 to state 1 to 1
xi0[18,0]=1
xi0[19, 0] = 1 # Using zero-based indexing for states
print(Q)
# Calculate the number of shows per second 
number_show_per_seconds = (((Q*xi0) @ np.ones(20)) @  pi)
num_show_per_day = (number_show_per_seconds) * 60 * 60 * 24

# Convert to minutes 
average_duration_minutes = 1 /(number_show_per_seconds * 60)
print("Number of shows per day:", num_show_per_day)
print("Average show duration:", average_duration_minutes)