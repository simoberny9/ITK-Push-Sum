import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class RobustPushSumNode:
    """
    Node state for the robustified push-sum algorithm from P6.

    The naming follows the standard push-sum implementation from P5:
    x is the numerator, y is the denominator, and z is their ratio.
    """

    def __init__(self, node_id, initial_value, node_ids, out_degree):
        self.id = node_id

        # Local algorithm states: Eqs. (14)-(15) in P6.
        self.x = initial_value
        self.y = 1.0
        self.z = initial_value
        self.out_degree = out_degree  # Includes the implicit self-loop.

        # Per-link running sums used to recover mass from lost packets.
        # A separate cumulative value is needed for every time-varying link.
        self.sigma = {node_id: 0.0 for node_id in node_ids}
        self.eta = {node_id: 0.0 for node_id in node_ids}

        # Last received running sums for every possible sender.
        # Keeping all node IDs supports a time-varying graph.
        self.rho_old = {node_id: 0.0 for node_id in node_ids}
        self.mu_old = {node_id: 0.0 for node_id in node_ids}
        self.rho_new = {node_id: 0.0 for node_id in node_ids}
        self.mu_new = {node_id: 0.0 for node_id in node_ids}

    def prepare_transmission(self, receiver_ids):
        """Advance and transmit one running sum for every active link."""
        x_share = self.x / self.out_degree
        y_share = self.y / self.out_degree
        messages = {}

        for receiver_id in receiver_ids:
            self.sigma[receiver_id] += x_share
            self.eta[receiver_id] += y_share
            messages[receiver_id] = {
                "src": self.id,
                "sigma_msg": self.sigma[receiver_id],
                "eta_msg": self.eta[receiver_id],
            }

        return messages

    def receive_message(self, msg):
        """Store the latest successfully received running sum."""
        src = msg["src"]
        self.rho_new[src] = msg["sigma_msg"]
        self.mu_new[src] = msg["eta_msg"]

    def update_state(self):
        """
        Recover the incoming x and y mass from running-sum differences.

        If a packet is lost, rho_new and mu_new remain at their previous
        values, so a later packet contains the accumulated missing mass.
        """
        self.x = sum(self.rho_new[i] - self.rho_old[i] for i in range(len(self.rho_new)))
        self.y = sum(self.mu_new[i] - self.mu_old[i] for i in range(len(self.mu_new)))
        self.z = self.x / self.y if self.y > 0 else 0.0

        # Shift memory for the next iteration
        for i in self.rho_old:
            self.rho_old[i] = self.rho_new[i]
            self.mu_old[i] = self.mu_new[i]


def plot_graph(graph, title):
    """Display one communication graph."""
    plt.figure(figsize=(7, 5))
    nx.draw_circular(graph,with_labels=True,node_color="lightblue",edge_color="gray",node_size=700,font_weight="bold",)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_results(graph,history_z,history_x,history_y,true_average,title,):
    """Plot the final graph and the three local state histories."""
    iterations = history_z.shape[0] - 1
    node_count = history_z.shape[1]
    time_steps = np.arange(iterations + 1)

    fig, axes = plt.subplots(2,2,figsize=(14, 10),constrained_layout=True,)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    nx.draw_circular(graph,ax=axes[0, 0],with_labels=True,node_color="lightblue",edge_color="gray",node_size=700,font_weight="bold")
    axes[0, 0].set_title("Final communication graph G(T)")

    plot_state = [
        (axes[0, 1], history_z, r"Estimate $z_i(t)$", r"$z_i(t)$"),
        (axes[1, 0], history_x, r"Local state $x_i(t)$", r"$x_i(t)$"),
        (axes[1, 1], history_y, r"Local state $y_i(t)$", r"$y_i(t)$"),
    ]

    for axis, history, plot_title, y_label in plot_state:
        for node_id in range(node_count):
            axis.plot(
                time_steps,
                history[:, node_id],
                linewidth=1.5,
                label=f"Node {node_id}",
            )
        if history is history_z:
            axis.axhline(true_average, color="r", linestyle="--", label="True average")
        axis.set_title(plot_title)
        axis.set_xlabel("Iterations k")
        axis.set_ylabel(y_label)
        axis.legend(loc="best", fontsize=8, ncol=2)
        axis.grid(True)

    plt.show()


def generate_strongly_connected_graph(node_count, edge_probability):
    """Generate a strongly connected directed graph without self-loops."""
    graph = nx.erdos_renyi_graph(n=node_count,p=edge_probability,directed=True,)
    while not nx.is_strongly_connected(graph):
        graph = nx.erdos_renyi_graph(n=node_count,p=edge_probability,directed=True,)
    return graph


def simulate_robust_push_sum(N=5, iterations=50, packet_loss_prob=0.3):
    """Simulate robustified push-sum over a time-varying directed graph."""
    edge_probability = 0.2
    initial_graph = generate_strongly_connected_graph(N, edge_probability)
    G = initial_graph.copy()
    node_ids = list(range(N))

    # Display the initial graph only once.
    plot_graph(initial_graph, "Initial communication graph G(0)")

    true_values = np.random.uniform(0, 10, N)
    true_average = np.mean(true_values)
    print(f"True values: {np.round(true_values, 2)}")
    print(f"True average: {true_average:.3f}\n")

    nodes = [
        RobustPushSumNode(
            node_id,
            true_values[node_id],
            node_ids,
            len(list(G.successors(node_id))) + 1,
        )
        for node_id in node_ids
    ]

    history_z = np.zeros((iterations + 1, N))
    history_x = np.zeros((iterations + 1, N))
    history_y = np.zeros((iterations + 1, N))
    for node_id, node in enumerate(nodes):
        history_z[0, node_id] = node.z
        history_x[0, node_id] = node.x
        history_y[0, node_id] = node.y

    # Simulation Loop
    for k in range(iterations):
        # Change the strongly connected directed graph at every iteration.
        G = generate_strongly_connected_graph(N, edge_probability)

        # Each node splits mass according to the current out-degree.
        for node in nodes:
            node.out_degree = len(list(G.successors(node.id))) + 1

        messages = []
        for node in nodes:
            receiver_ids = list(G.successors(node.id)) + [node.id]
            messages.append(node.prepare_transmission(receiver_ids))

        # Route cumulative messages on the current graph.
        for sender in nodes:
            sender_neighbours = list(G.successors(sender.id)) + [sender.id]

            for receiver_id in sender_neighbours:
                receiver = nodes[receiver_id]
                if (sender.id == receiver_id or random.random() > packet_loss_prob):
                    receiver.receive_message(messages[sender.id][receiver_id])

        # Update and log the state of each node.
        for node_id, node in enumerate(nodes):
            node.update_state()
            history_z[k + 1, node_id] = node.z
            history_x[k + 1, node_id] = node.x
            history_y[k + 1, node_id] = node.y

    final_average = np.mean(history_z[-1])
    final_error = abs(final_average - true_average)
    print(f"Expected average: {true_average:.6f}")
    print(f"Final average estimate: {final_average:.6f}")
    print(f"Absolute error: {final_error:.6e}")

    title = ("Robustified Push-Sum - Time-varying graph - "f"Packet Loss: {packet_loss_prob * 100}%")
    plot_results(G,history_z,history_x,history_y,true_average,title)


if __name__ == "__main__":
    simulate_robust_push_sum(N=5, iterations=80, packet_loss_prob=0.0)