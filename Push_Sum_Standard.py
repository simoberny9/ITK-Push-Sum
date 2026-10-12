import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

class StandardPushSumNode:
    """
    Class representing a node in the Standard Push-Sum algorithm.
    """
    def __init__(self, node_id, initial_value, out_degree):
        self.id = node_id
        # States initialization (Eq. 1 in P5)
        self.x = initial_value
        self.y = 1.0 
        self.z = initial_value
        self.out_degree = out_degree # Includes self-loop
        
        # Buffer for incoming messages in the current iteration
        self.next_x = 0.0
        self.next_y = 0.0

    def get_transmission_message(self):
        # Prepare message for out-neighbors (Publisher callback equivalent)
        return {
            'src': self.id, 
            'x_msg': self.x / self.out_degree, 
            'y_msg': self.y / self.out_degree
        }

    def receive_message(self, msg):
        # Accumulate incoming mass (Subscriber callback equivalent)
        self.next_x += msg['x_msg']
        self.next_y += msg['y_msg']

    def update_state(self):
        # Update states synchronously at the end of the step
        self.x = self.next_x
        self.y = self.next_y
        self.z = self.x / self.y if self.y > 0 else 0.0
        # Reset buffers
        self.next_x = 0.0
        self.next_y = 0.0


def plot_graph(G, title):
    """Display one communication graph."""
    plt.figure(figsize=(7, 5))
    nx.draw_circular(
        G,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=700,
        font_weight="bold",
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_results(G, history_z, history_x, history_y, true_average, title, N, iterations):
    """Helper function to plot the results and graph in a 2x2 grid."""
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    time_steps = np.arange(iterations + 1)
    
    # 1. Top-Left: Network Graph
    # Use different random colors for nodes to enhance visibility
    #color_map = plt.colormaps.get_cmap('tab10')
    nx.draw_circular(G, ax=axs[0, 0], with_labels=True, node_color="lightblue", edge_color='gray', node_size=700, font_weight='bold')
    axs[0, 0].set_title("Final communication graph G(T)")

    # 2. Top-Right: Z Estimate
    for i in range(N):
        axs[0, 1].plot(time_steps, history_z[:, i], label=f'Node {i}')
    axs[0, 1].axhline(true_average, color='r', linestyle='--', label='Real Mean')
    axs[0, 1].set_title(r'Estimate $z_i(t)$')
    axs[0, 1].set_ylabel(r'$z_i(t)$')
    axs[0, 1].set_xlabel('Iterations k')
    axs[0, 1].legend(loc='best', fontsize=9)
    axs[0, 1].grid(True)

    # 3. Bottom-Left: X Mass
    for i in range(N):
        axs[1, 0].plot(time_steps, history_x[:, i], label=f'Node {i}')
    axs[1, 0].set_title(r'Local Mass $x_i(t)$')
    axs[1, 0].set_ylabel(r'$x_i(t)$')
    axs[1, 0].set_xlabel('Iterations k')
    axs[1, 0].legend(loc='best', fontsize=9)
    axs[1, 0].grid(True)

    # 4. Bottom-Right: Y Mass
    for i in range(N):
        axs[1, 1].plot(time_steps, history_y[:, i], label=f'Node {i}')
    axs[1, 1].set_title(r'Local Weight $y_i(t)$')
    axs[1, 1].set_ylabel(r'$y_i(t)$')
    axs[1, 1].set_xlabel('Iterations k')
    axs[1, 1].legend(loc='best', fontsize=9)
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Leave space for the main title
    plt.show()


def simulate_standard_push_sum(N=5, iterations=50, packet_loss_prob=0.3):
    # 1. Generate the initial strongly connected directed graph.
    EDGE_PROB = 0.2
    G = nx.erdos_renyi_graph(n=N, p=EDGE_PROB, directed=True)
    while not nx.is_strongly_connected(G): # Ensure the graph is strongly connected
        G = nx.erdos_renyi_graph(n=N, p=EDGE_PROB, directed=True)
    initial_graph = G.copy()

    # The graph is displayed only once before the simulation starts.
    plot_graph(initial_graph, "Initial communication graph G(0)")

    # Initialization
    true_values = np.random.uniform(0, 10, N)
    print(f"True values: {np.round(true_values, 2)}")
    true_average = np.mean(true_values)
    print(f"True average: {true_average:.3f}\n")
    
    nodes = []
    for i in range(N):
        # Out-degree = number of successors + 1 (for the self loop)
        out_deg = len(list(G.successors(i))) + 1
        nodes.append(StandardPushSumNode(i, true_values[i], out_deg))
    
    # History arrays (+1 to store initial state at k=0)
    history_z = np.zeros((iterations + 1, N))
    history_x = np.zeros((iterations + 1, N))
    history_y = np.zeros((iterations + 1, N))

    # Log initial state at k=0
    for i, node in enumerate(nodes):
        history_z[0, i] = node.z
        history_x[0, i] = node.x
        history_y[0, i] = node.y

    # Main Loop
    for k in range(iterations):
        # Use a new strongly connected directed graph at every iteration.
        # The self-loop is implicit and is not included in G.
        G = nx.erdos_renyi_graph(n=N, p=EDGE_PROB, directed=True)
        while not nx.is_strongly_connected(G):
            G = nx.erdos_renyi_graph(n=N, p=EDGE_PROB, directed=True)

        # Push-sum must split the mass according to the current graph.
        for node in nodes:
            node.out_degree = len(list(G.successors(node.id))) + 1

        messages = [node.get_transmission_message() for node in nodes]
        
        # Routing with packet loss on the current graph
        for sender in nodes:
            msg = messages[sender.id]
            out_neighbors = list(G.successors(sender.id)) + [sender.id] 
            
            for receiver_id in out_neighbors:
                receiver = nodes[receiver_id]
                if sender.id == receiver.id:
                    receiver.receive_message(msg) # Self-loop never drops
                else:
                    if random.random() > packet_loss_prob:
                        receiver.receive_message(msg)
                        
        # Update and Log
        for i, node in enumerate(nodes):
            node.update_state()
            history_z[k+1, i] = node.z
            history_x[k+1, i] = node.x
            history_y[k+1, i] = node.y

    title = f'Standard Push-Sum - Time-varying graph - Packet Loss: {packet_loss_prob*100}%'
    plot_results(G, history_z, history_x, history_y, true_average, title, N, iterations)

if __name__ == "__main__":
    simulate_standard_push_sum(N=5, iterations=40, packet_loss_prob=0.0)