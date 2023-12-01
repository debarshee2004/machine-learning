import numpy as np


class QLearning:
    def __init__(
        self,
        num_states,
        num_actions,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_prob=0.1,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        # Initialize Q-values to zeros
        self.q_values = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        # Exploration-exploitation trade-off
        if np.random.rand() < self.exploration_prob:
            # Explore: choose a random action
            return np.random.choice(self.num_actions)
        else:
            # Exploit: choose the action with the highest Q-value
            return np.argmax(self.q_values[state, :])

    def update_q_values(self, state, action, reward, next_state):
        # Q-value update using the Q-learning formula
        current_q_value = self.q_values[state, action]
        max_future_q_value = np.max(self.q_values[next_state, :])
        new_q_value = (
            1 - self.learning_rate
        ) * current_q_value + self.learning_rate * (
            reward + self.discount_factor * max_future_q_value
        )
        self.q_values[state, action] = new_q_value


# Example usage:
# Define the environment
num_states = 5
num_actions = 3
env = QLearning(num_states, num_actions)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state = np.random.randint(0, num_states)  # Initialize the state randomly

    # Run episode
    while True:
        action = env.choose_action(state)  # Choose an action
        # Simulate taking the chosen action and observe the next state and reward
        next_state = (state + action) % num_states
        reward = (
            1 if next_state == 0 else 0
        )  # Reward is 1 when the episode reaches the goal state (state 0)
        env.update_q_values(state, action, reward, next_state)  # Update Q-values

        state = next_state

        if state == 0:  # Goal state reached
            break

# Print the learned Q-values
print("Learned Q-values:")
print(env.q_values)
