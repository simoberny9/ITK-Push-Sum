# Push-Sum Consensus Simulations

This repository contains two Python simulations of the **Push-Sum distributed
averaging algorithm**. They were developed to study how a network of nodes can
compute the average of their initial values without a central coordinator.

## Files

- [`Push_Sum_Standard.py`](Push_Sum_Standard.py) implements the standard
  Push-Sum algorithm.
- [`Robustified_Push_Sum.py`](Robustified_Push_Sum.py) implements a robust
  version that can recover mass carried by packets lost on the network.

In both simulations:

1. Each node starts with a random value between 0 and 10.
2. A strongly connected directed graph is generated and regenerated at every
   iteration, representing a time-varying communication network.
3. Nodes exchange two quantities, `x` and `y`, and estimate the average as
   `z = x / y`.
4. The scripts print the true average and display plots of the network,
   estimates, and local states.

## Difference between the algorithms

The standard implementation sends the current fractions of `x` and `y`
directly to each neighbour. If a packet is lost, its mass is lost as well,
which can prevent the nodes from reaching the correct average.

The robustified implementation maintains cumulative per-link sums (`sigma` and
`eta`). Receivers compare each newly received cumulative value with the last
one they processed. Consequently, a later successfully delivered packet also
contains any mass from previous lost packets. This allows the algorithm to
preserve and recover the total mass under packet loss.

## Requirements

Python 3 and the following packages:

```bash
pip install numpy matplotlib networkx
```

## Running the simulations

From this directory, run either script:

```bash
python Push_Sum_Standard.py
python Robustified_Push_Sum.py
```

The default configurations use five nodes and a time-varying graph. To study
packet losses, change the `packet_loss_prob` argument in the `__main__` block,
for example:

```python
simulate_robust_push_sum(N=5, iterations=80, packet_loss_prob=0.3)
```

The robustified version is intended to be compared with the standard version
under the same network and packet-loss conditions, showing the impact of
communication failures on distributed average consensus.
