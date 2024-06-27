# Playing Catan with Policy Optimized Monte-Carlo Tree Search

**Authors**:

| Nathan Guzman | Christopher Nguyen |
| --- | --- |
| Stanford University | Stanford University |

## Abstract
This project aims to develop a reinforcement learning (RL) agent inspired by AlphaGo Zero, using self-play and Monte-Carlo Tree Search (MCTS) for the board game Catan. By extending the functionality of MCTS with policy optimization, we guide the simulation policy to make optimal actions rather than random sub-optimal ones. The agent learns to develop strategies through exploration and exploitation, optimizing performance against human competitors. We evaluate the policy optimized MCTS against random MCTS and a heuristic-based AI player in terms of win rate, point-giving buildings built, and average points per game. Results show limited performance improvement due to insufficient training time.

## Introduction
This project explores applying reinforcement learning (RL) to the strategic board game Settlers of Catan. By implementing MCTS enhanced with Proximal Policy Optimization (PPO) during rollout, we aim to develop an AI agent capable of making sophisticated moves, surpassing the random action selection in standard MCTS. Inspired by AlphaGo Zero, we use self-play, exploration, and exploitation to refine the agent's strategies. We evaluate the policy-optimized MCTS against baseline MCTS and heuristic players, aiming to address strategic decision-making challenges in complex environments.

This project was completed as the final project for our CS234 Reinforcement Learning class at Stanford University, which we took in Spring 2024.

<p align="center">
  <img src="https://i.ibb.co/dfNM4ng/Screenshot-2024-05-21-at-1-50-17-AM.jpg" alt="Catan UI" width="400">
  <br>
  <em>Catan UI</em>
</p>

## Background: Settlers of Catan
Catan is a strategic board game where players build settlements, cities, and roads, competing for resources. Players collect and trade resources like wood, brick, wheat, sheep, and ore, which are produced based on dice rolls corresponding to numbered tiles. The game involves multi-action turns based on resource availability, build locations, and proximity to victory. Our project aims to provide insights into developing RL agents for complex real-world scenarios.

### Game Rules
Each game features a unique hex tile layout with associated resources and dice numbers. The goal is to accumulate 10 or more victory points through construction and special achievements, requiring careful resource management, strategic placement, and negotiation.

### Build Roads, Settlements, Cities
- **Roads**: Cost brick and wood, max 15 per player.
- **Settlements**: Cost brick, wood, wheat, and sheep, max 5 per player.
- **Cities**: Built on settlements, cost 3 ore and 2 wheat, max 4 per player.

### Trading
Players can trade resources with the bank or other players. Ports allow resource exchanges at set ratios, e.g., 4:1.

### Development Cards
We focus on the Victory Point (VP) card, granting 2 victory points when drawn. Other development cards were not utilized in this project.

### Robber
A dice roll of 7 allows the current player to move the robber, stealing resources from players adjacent to the chosen tile.

### Special Conditions
The player with the longest road receives 2 points, which can be stolen by others.

<div align="center">
  <img src="https://i.ibb.co/Xt78rBS/s-l1600.jpg" alt="Catan UI" width="300">
  <img src="https://i.ibb.co/k97ppM2/Catan-Games-Catan-Studio-4-c27adfac-84e5-4b27-a90b-67a378f3c9f0-jpg.webp" alt="MCTS Algorithm Iteration" width="300">
</div>

<p align="center">
  <em>Road, City, and Settlement pieces</em> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<em>Resource and Development cards</em>
</p>

## Related Work
### AlphaGo Zero
AlphaGo Zero demonstrated the power of combining MCTS with deep learning, starting from random play without supervision. We adopt a similar approach, focusing on implementing a workable Catan game using self-play and MCTS.

### Monte Carlo Tree Search in Settlers of Catan
Szita et al. (2010) successfully adapted MCTS for Catan, showing strong performance compared to heuristic-based implementations. We followed a similar approach, focusing on rule changes and action availability.

## Approach
The core of our project is the implementation of Monte-Carlo Tree Search (MCTS) to develop an AI agent for Catan. Using a pre-implemented game environment, we added MCTS with UCB for decision-making during simulations.

### Monte-Carlo Tree Search
MCTS involves four phases: selection, expansion, simulation, and backpropagation. These phases are iterated to determine the best actions:

- **Selection**: Traversing the tree to select a promising node.
- **Expansion**: Adding a new node to the tree.
- **Simulation**: Running a simulation from the new node to a terminal state.
- **Backpropagation**: Updating the nodes based on the simulation results.

<p align="center">
  <img src="https://i.ibb.co/XCRH2W3/Phases-of-the-Monte-Carlo-tree-search-algorithm-A-search-tree-rooted-at-the-current-2.png" alt="MCTS" width="400">
  <br>
  <em>Phases of the Monte Carlo Tree Search algorithm</em>
</p>

## Experimental Results
### Experiment Setup
Experiments involved two or three players, comparing MCTS and heuristic-based agents over multiple games.

### Initial Experiment Results
Initial results showed the heuristic-based agent outperformed MCTS. Modifying the reward function improved MCTS performance.

### Modified Rewards Experiment
Adjusted rewards increased the MCTS win rate and average points per game.

### Three Player Experiment
Testing with three players provided insights into multi-player dynamics.

## Discussion
### Performance of MCTS
Reward alignment significantly improved MCTS performance, highlighting the importance of well-designed reward functions.

### Key Challenges in MCTS
Catan's complexity, stochasticity, and multi-agent environment posed significant challenges for MCTS implementation.

### Observations
Reward alignment and variability in performance were key observations, emphasizing the need for further tuning and training.

## Future Work
Future work includes performance tuning, optimizing the reward function, enabling all possible actions, and improving the custom Catan gym environment.

## Conclusion
We successfully implemented an MCTS agent for Catan. While performance was limited, the project demonstrated the potential of MCTS in complex multi-player games and provided valuable insights for future improvements.

## References
- Baier, H., & Cowling, P. I. (2018). Evolutionary MCTS for multi-action adversarial games. IEEE Conference on Computational Intelligence and Games (CIG), 1-8.
- Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017). Mastering the game of Go without human knowledge. Nature, 550(7676), 354-359.
- Swiechowski, M., Godlewski, K., Sawicki, B., & Mandziuk, J. (2021). Monte Carlo tree search: A review of recent modifications and applications. arXiv preprint arXiv:2103.04931.
- Szita, I., Chaslot, G., & Spronck, P. (2010). Monte-Carlo tree search in settlers of Catan. Advances in Computer Games, 21-32.
- Vombatkere, K. (2022). Catan-AI. Retrieved from https://github.com/kvombatkere/Catan-AI
