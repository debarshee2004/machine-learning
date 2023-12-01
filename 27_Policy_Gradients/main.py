import numpy as np


class PolicyGradient:
    def __init__(self, state_dim, action_dim, learning_rate=0.01, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Initialize policy parameters
        self.theta = np.random.rand(state_dim, action_dim)

        # Store the trajectory
        self.states = []
        self.actions = []
        self.rewards = []

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0, keepdims=True)

    def select_action(self, state):
        logits = np.dot(state, self.theta)
        probabilities = self.softmax(logits)
        action = np.random.choice(self.action_dim, p=probabilities.ravel())
        return action

    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def compute_discounted_rewards(self):
        discounted_rewards = np.zeros_like(self.rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            discounted_rewards[t] = running_add
        # Normalize the discounted rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards

    def update_policy(self):
        discounted_rewards = self.compute_discounted_rewards()

        for t in range(len(self.states)):
            state = self.states[t]
            action = self.actions[t]
            discounted_reward = discounted_rewards[t]

            # Compute the gradient of the log probability with respect to the policy parameters
            logits = np.dot(state, self.theta)
            probabilities = self.softmax(logits)
            dsoftmax = probabilities.copy()
            dsoftmax[0, action] -= 1

            # Gradient of the policy parameters
            dtheta = np.dot(state.T, dsoftmax[np.newaxis, :]) * discounted_reward

            # Update the policy parameters
            self.theta += self.learning_rate * dtheta.squeeze()

        # Clear the trajectory
        self.states = []
        self.actions = []
        self.rewards = []


# Example usage
state_dim = 4
action_dim = 2
policy = PolicyGradient(state_dim, action_dim)

# Training loop
for episode in range(1000):
    # Assume you have an environment with reset(), step(), and get_state() functions
    state = env.reset()

    while True:
        # Select an action based on the current policy
        action = policy.select_action(state)

        # Take the selected action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)

        # Store the transition
        policy.store_transition(state, action, reward)

        if done:
            # Update the policy at the end of the episode
            policy.update_policy()
            break

        # Move to the next state
        state = next_state
