import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define the distributions
def hyper_exponential(p, lambdas):
    if np.random.rand() < p[0]:
        return np.random.exponential(1 / lambdas[0])
    else:
        return np.random.exponential(1 / lambdas[1])

def erlang(k, lambda_):
    return np.sum([np.random.exponential(1 / lambda_) for _ in range(k)])

def hyper_erlang(p, ks, lambdas):
    if np.random.rand() < p[0]:
        return np.sum([np.random.exponential(1 / lambdas[0]) for _ in range(ks[0])])
    else:
        return np.random.exponential(1 / lambdas[1])

# Define parameters
p_gui = [0.8, 0.2]
lambdas_gui = [0.4, 0.1]
lambda_cash = 0.4
k_electronic = 4
lambda_electronic = 2
p_print = [0.95, 0.05]
ks_print = [2, 1]
l_print = [10, 0.1]

# Define ticket prices and probabilities
ticket_prices = [2.5, 4.0, 6.0]
ticket_probs = [0.9, 0.06, 0.04]

# Simulation parameters
simulation_time = 20 * 60  # 20 hours in minutes
elapsed_time = 0
total_cash_collected = 0
state_counts = {"User Input": 0, "Cash Transaction": 0, "Electronic Transaction": 0, "Printing": 0}
time_spent = {"User Input": 0, "Cash Transaction": 0, "Electronic Transaction": 0, "Printing": 0}
complete_runs = 0  # To count the number of complete runs

while elapsed_time < simulation_time:
    # Increment initial state count
    state_counts["User Input"] += 1

    # GUI time
    gui_time = hyper_exponential(p_gui, lambdas_gui)
    time_spent["User Input"] += gui_time
    elapsed_time += gui_time
    
    # Check if the customer leaves
    if np.random.rand() < 0.2:
        continue
    
    complete_runs += 1  # Increment complete run count as we move past initial state

    # Payment method
    if np.random.rand() < 0.35:
        # Cash payment
        cash_time = np.random.exponential(1 / lambda_cash)
        state_counts["Cash Transaction"] += 1
        time_spent["Cash Transaction"] += cash_time
        elapsed_time += cash_time
        
        # Ticket printing
        print_time = hyper_erlang(p_print, ks_print, l_print)
        state_counts["Printing"] += 1
        time_spent["Printing"] += print_time
        elapsed_time += print_time
        
        # Cash collected
        ticket_price = np.random.choice(ticket_prices, p=ticket_probs)
        total_cash_collected += ticket_price
    else:
        # Electronic payment
        el_time = erlang(k_electronic, lambda_electronic)
        state_counts["Electronic Transaction"] += 1
        time_spent["Electronic Transaction"] += el_time
        elapsed_time += el_time
        
        # Ticket printing
        print_time = hyper_erlang(p_print, ks_print, l_print)
        state_counts["Printing"] += 1
        time_spent["Printing"] += print_time
        elapsed_time += print_time

# Compute probabilities
total_state_time = sum(time_spent.values())
probabilities = {state: time / total_state_time for state, time in time_spent.items()}

# Compute average transaction duration and cash collected
average_transaction_duration = total_state_time / complete_runs
average_cash_20_hours = total_cash_collected

print("Probabilities:")
for state, prob in probabilities.items():
    print(f"{state}: {prob:.4f}")

print(f"Average transaction duration: {average_transaction_duration:.4f} minutes")
print(f"Cash collected in 20 hours: €{average_cash_20_hours:.2f}")
print(f"Number of complete runs: {complete_runs}")
print(f"Number of times passed through the 'Waiting for User Input' state: {state_counts['User Input']}")

# Define state machine transitions 
transitions = {
    ("Waiting for User Input", "Waiting for User Input"): (0.2, "Leaves w/o Ticket"),
    ("Waiting for User Input", "Handling Cash Transaction"): (0.35 * (1 - 0.2), "Cash Payment (Exp)"),
    ("Waiting for User Input", "Handling Electronic Transaction"): (0.65 * (1 - 0.2), "Electronic Payment (Erlang k=4)"),
    ("Handling Cash Transaction", "Printing Ticket"): (1.0, "Print Ticket (Hyper-Erlang)"),
    ("Handling Electronic Transaction", "Printing Ticket"): (1.0, "Print Ticket (Hyper-Erlang)"),
    ("Printing Ticket", "Waiting for User Input"): (1.0, "Return to Input")
}

# Create the state machine diagram
G = nx.DiGraph()

# Add nodes (states)
state_labels = {
    "Waiting for User Input": f"Waiting\n({probabilities['User Input']:.2%})",
    "Handling Cash Transaction": f"Cash Transaction\n({probabilities['Cash Transaction']:.2%})",
    "Handling Electronic Transaction": f"Electronic Transaction\n({probabilities['Electronic Transaction']:.2%})",
    "Printing Ticket": f"Printing Ticket\n({probabilities['Printing']:.2%})"
}
for state in state_labels.keys():
    G.add_node(state)

# Add edges (transitions) with probabilities and distribution names as labels
for (start, end), (probability, dist) in transitions.items():
    G.add_edge(start, end, label=f"{dist}\n{probability * 100:.0f}%")

# Draw the state machine
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 10))
nx.draw(G, pos, labels=state_labels, node_size=5500, node_color="lightblue", font_size=6, font_weight="bold", arrows=True)

# Draw edge labels with distributions and probabilities
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black", font_size=8)
plt.title("State Machine Diagram for Ticketing System with Empirical Probabilities and Distributions")
plt.show()
