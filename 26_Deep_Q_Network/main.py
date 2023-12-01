import numpy as np
import random
import gym


# Define the Q-network
class QNetwork:
    def __init__(self, state_size, action_size, hidden_size):
        self.weights1 = np.random.rand(state_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.rand(hidden_size, action_size)
        self.bias2 = np.zeros((1, action_size))

    def forward(self, state):
        hidden_layer = np.maximum(0, np.dot(state, self.weights1) + self.bias1)
        q_values = np.dot(hidden_layer, self.weights2) + self.bias2
        return q_values


# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add_experience(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)


# Deep Q Network Agent
class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        hidden_size,
        learning_rate,
        gamma,
        epsilon,
        epsilon_decay,
        min_epsilon,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size, hidden_size)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.replay_buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            q_values = self.q_network.forward(state)
            return np.argmax(q_values)

    def train(self, batch_size):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        batch = self.replay_buffer.sample_batch(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.vstack(states)
        next_states = np.vstack(next_states)

        current_q_values = self.q_network.forward(states)
        next_q_values = self.q_network.forward(next_states)

        targets = np.copy(current_q_values)
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(
                    next_q_values[i]
                )

        # Update the Q-network
        loss = np.mean(np.square(targets - current_q_values))
        gradient = 2 * (current_q_values - targets) / batch_size
        self.q_network.weights2 -= self.learning_rate * np.dot(
            self.q_network.hidden_layer.T, gradient
        )
        self.q_network.bias2 -= self.learning_rate * np.sum(
            gradient, axis=0, keepdims=True
        )
        hidden_gradient = np.dot(gradient, self.q_network.weights2.T)
        hidden_gradient[self.q_network.hidden_layer <= 0] = 0
        self.q_network.weights1 -= self.learning_rate * np.dot(
            states.T, hidden_gradient
        )
        self.q_network.bias1 -= self.learning_rate * np.sum(
            hidden_gradient, axis=0, keepdims=True
        )

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# Main training loop
def train_dqn(env, agent, episodes, max_steps, batch_size):
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])

            agent.replay_buffer.add_experience(
                (state, action, reward, next_state, done)
            )
            agent.train(batch_size)

            state = next_state

            if done:
                break

        print(f"Episode: {episode + 1}, Total Reward: {step + 1}")


# Define environment and agent parameters
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_size = 24
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
episodes = 500
max_steps = 200
batch_size = 32

# Create DQNAgent
agent = DQNAgent(
    state_size,
    action_size,
    hidden_size,
    learning_rate,
    gamma,
    epsilon,
    epsilon_decay,
    min_epsilon,
)

# Train the agent
train_dqn(env, agent, episodes, max_steps, batch_size)

# After training, you can use the trained agent to play the game
state = env.reset()
state = np.reshape(state, [1, agent.state_size])

for step in range(max_steps):
    action = agent.select_action(state)
    env.render()
    next_state, _, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, agent.state_size])
    state = next_state

    if done:
        break

env.close()
