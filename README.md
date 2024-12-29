# 🤖 Computational Intelligence Project

## 📖 Overview
This project, part of the Computational Intelligence course at Aristotle University of Thessaloniki, focuses on solving complex problems using three different techniques:
- **Reinforcement Learning (RL)**
- **Fuzzy Logic**
- **Genetic Algorithms**

Each technique is applied to specific challenges, with detailed implementation and analysis. Special emphasis is given to the RL agent, which is designed to solve a maze problem.

---

## 🎯 Goals

- **Train an RL Agent** to navigate and solve mazes of varying sizes efficiently.
- **Apply Fuzzy Logic** to solve exercises related to set operations and Takagi-Sugeno fuzzy models.
- **Implement Genetic Algorithms** for optimization tasks, including analysis of mutation and crossover effects.

---

## ✨ Features

### 🧠 Reinforcement Learning
- **Maze Problem**: Train an agent to find a treasure in a maze while avoiding obstacles.
- **RL Environment**:
  - **State**: Represented as a 2D grid encoding walls, paths, player position, and treasure.
  - **Action Space**: Four possible moves (up, down, left, right).
  - **Reward Mechanism**: Positive reward for reaching the treasure, penalties for invalid moves.
- **Agent Architectures**:
  - Custom DQN-CNN
  - Dueling DQN-CNN
  - Inception DQN-CNN
- **Learning Techniques**:
  - Double DQN algorithm for stable learning.
  - Epsilon-greedy strategy for exploration-exploitation balance.
  - Experience replay for better training.
- **Key Results**:
  - RL agents successfully learned to navigate mazes with fixed starting positions.
  - Generalization to dynamic mazes remains a challenge, requiring further tuning.

### 🌫️ Fuzzy Logic
- Solved multiple exercises involving fuzzy set operations:
  - **Union, Intersection, and Complement**: Explored using various operators (min, max, product).
  - **Takagi-Sugeno Fuzzy Models**: Designed rules and membership functions to compute outputs based on inputs.
- **Results**:
  - Successfully implemented fuzzy operations and visualized membership functions.
  - Generated output surfaces for fuzzy systems.

### 🧬 Genetic Algorithms
- Implemented optimization tasks using genetic algorithms:
  - **Fitness Functions**: Maximizing binary sequences and minimizing specific cost functions.
  - **Key Components**:
    - Roulette-wheel selection.
    - Single-point crossover.
    - Bit mutation with varying rates.
  - **Analysis**:
    - Explored effects of population size, mutation rate, and crossover rate.
    - Demonstrated the balance between exploration (mutation) and exploitation (crossover).
- **Results**:
  - Identified optimal parameter ranges for faster convergence and better solutions.

---

## 🏆 Results
- **Reinforcement Learning**: Developed RL agents that learned maze navigation under specific conditions.
- **Fuzzy Logic**: Successfully solved and visualized complex fuzzy operations and systems.
- **Genetic Algorithms**: Showcased the trade-offs between mutation and crossover in optimization tasks.

---

## 🛠️ Techniques Utilized
- **Deep Reinforcement Learning**: Double DQN, epsilon-greedy strategy, and experience replay.
- **Fuzzy Logic**: Set operations, Takagi-Sugeno models, and membership functions.
- **Genetic Algorithms**: Evolutionary optimization techniques with parameter analysis.
- **Python Libraries**: PyTorch, NumPy, and Matplotlib for RL implementations and visualizations.

---

## 📂 Repository Contents
- **📄 Report**: Detailed analysis in [Computational Intelligence Report](./Computational_Intelligence.pdf).
- **💻 Code**: Python scripts for all three techniques (RL, Fuzzy Logic, Genetic Algorithms).
- **📊 Plots**: Visualizations of training results, fuzzy operations, and genetic algorithm performance.

---

## 🤝 Contributors
- [Panagiotis Karvounaris](https://github.com/karvounaris)

---

Thank you for exploring this project! 🌟 Feel free to raise issues or contribute to improve the repository. 🚀😊
