import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from maze_env_6x6 import GridEnvironment_1
from maze_agent_6x6 import DQNCNN
# from maze_agent_6x6 import DuelingDQNCNN
# from maze_agent_6x6 import InceptionDQNCNN

# Hyperparameters
EPISODES = 8000  # number of episodes to train
GAMMA = 0.99    # discount factor for future rewards
LEARNING_RATE = 0.005
BATCH_SIZE = 256
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.005
EPSILON_DECAY = 0.995
MIN_REPLAY_SIZE = 100
TARGET_UPDATE_FREQUENCY = 10
STEPS_PER_EPISODE = 300

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment and agent
env = GridEnvironment_1()
agent = DQNCNN().to(device)
target_agent = DQNCNN().to(device)
target_agent.load_state_dict(agent.state_dict())
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# Experience replay memory
memory = deque(maxlen=MEMORY_SIZE)

def to_tensor(states):
    states = np.array(states)  # Convert list of states to a numpy array

    # Check if the states are in 'batch' format or single state
    if states.ndim == 3:  # Assuming states are [batch, height, width]
        states = np.expand_dims(states, axis=1)  # Add channel dimension [batch, channel, height, width]
    elif states.ndim == 2:  # Single state [height, width]
        states = np.expand_dims(states, axis=0)  # Add batch dimension [1, height, width]
        states = np.expand_dims(states, axis=1)  # Add channel dimension [1, channel, height, width]

    return torch.tensor(states, dtype=torch.float32).to(device)  # Convert to tensor and send to device

def mean_reward_last_n_episodes(rewards, n=40):
    return np.mean(rewards[-n:]) if len(rewards) >= n else None

# List to store total rewards per episode
total_rewards = []
average_loss_per_episode = []

# Training loop
epsilon = EPSILON_START
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    loss_list = []
    average_loss = 0

    while not done and step_count < STEPS_PER_EPISODE:
        step_count += 1

        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice([0, 1, 2, 3])
        else:
            q_values = agent(to_tensor(state))
            action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Experience replay
        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            # Convert batched data to tensors
            states = to_tensor(list(states))
            next_states = to_tensor(list(next_states))
            actions = torch.tensor(actions).to(device)
            rewards = torch.tensor(rewards).to(device)
            dones = torch.tensor(dones).to(device)

            # Compute Q-values for current states
            current_q = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Double DQN update: Use the online network to select actions and the target network to evaluate them
            next_actions = agent(next_states).max(1)[1]  # Action selection by the online network
            next_q_values = target_agent(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)  # Q-value estimation by the target network
            expected_q = rewards + GAMMA * next_q_values * (1 - dones.float())

            # Loss and optimization steps
            loss = loss_fn(current_q, expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())

    if episode % TARGET_UPDATE_FREQUENCY == 0:
        target_agent.load_state_dict(agent.state_dict())

    # Calculate and store the average loss for this episode
    if len(loss_list) > 0:
        average_loss = sum(loss_list) / len(loss_list)
        average_loss_per_episode.append(average_loss)

    # Append the total reward for this episode
    if total_reward < -100:
        total_rewards.append(-100)
    else:    
        total_rewards.append(total_reward)
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Log the results
    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}, Average Loss: {average_loss}")

env.close()

# Plot the total rewards per episode
plt.figure(figsize=(10, 5))
plt.plot(total_rewards, label='Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.legend()
plt.show()

# Plot the average loss per episode after training
plt.figure(figsize=(10, 5))
plt.plot(average_loss_per_episode, label='Average Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Loss')
plt.title('Average Loss per Episode during Training')
plt.legend()
plt.show()

# Test the trained agent
def test_agent(env, agent):
    state = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    while not done:
        step_count += 1
        env.render()
        q_values = agent(to_tensor(state))
        action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward

        print(f"Step: {step_count}, Total Reward: {total_reward}")

    print(f"Test completed, Total Reward: {total_reward}")

# Run the test
test_agent(env, agent)

env = GridEnvironment_1()
test_agent(env, agent)
env = GridEnvironment_1()
test_agent(env, agent)
env = GridEnvironment_1()
test_agent(env, agent)

