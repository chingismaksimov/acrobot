import numpy as np
import gym
import keras
import copy
import time
import random


class Agent():
    def __init__(self, discount_factor=0.99, learning_rate=0.02, min_learning_rate=0.02, learning_rate_reduction=0.999, exploration_rate=0.5, min_exploration_rate=0.01, exploration_rate_reduction=0.95, num_episodes=1, num_steps=400, batch_size=10, num_epochs=1, num_training_sessions=100, winning_reward=10):
        self.brain = self.make_model()
        self.memory = []
        self.env = gym.make('Acrobot-v1')
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.learning_rate_reduction = learning_rate_reduction
        self.learning_rates = np.zeros(num_training_sessions)
        self.exploration_rate = exploration_rate
        self.min_exloration_rate = min_exploration_rate
        self.exploration_rate_reduction = exploration_rate_reduction
        self.exploration_rates = np.zeros(num_training_sessions)
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_training_sessions = num_training_sessions
        self.min_brain_size = self.num_episodes * self.num_steps
        self.winning_reward = winning_reward
        self.rewards_during_session = 0
        self.tot_rewards_per_session = np.zeros(num_training_sessions)
        self.training_time = None

    def play(self):
        self.memory = []
        self.rewards_during_session = 0
        for episode in range(self.num_episodes):
            observation = self.env.reset()
            for step in range(self.num_steps):
                # self.env.render()
                initial_observation = observation.reshape(1, -1)
                initial_q_values = self.brain.predict(initial_observation).flatten()
                if random.random() < self.exploration_rate:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(initial_q_values)
                observation, reward, done, _ = self.env.step(action)
                if done:
                    q_values = self.brain.predict(observation.reshape(1, -1)).flatten()
                    target = copy.copy(initial_q_values)
                    target[action] = initial_q_values[action] + self.learning_rate * (self.winning_reward - initial_q_values[action])
                    self.memory.append((initial_observation, initial_q_values, observation, q_values, target, action, reward, done))
                    self.rewards_during_session += self.winning_reward
                    break
                else:
                    q_values = self.brain.predict(observation.reshape(1, -1)).flatten()
                    target = copy.copy(initial_q_values)
                    target[action] = initial_q_values[action] + self.learning_rate * (reward + self.discount_factor * np.max(q_values) - initial_q_values[action])
                    self.memory.append((initial_observation, initial_q_values, observation, q_values, target, action, reward, done))
                    self.rewards_during_session += reward

    def learn_from_memory(self):
        self.memory = np.asarray(self.memory)
        x = self.memory[:, 0]
        y = self.memory[:, 4]
        x = np.concatenate(x)
        y = np.concatenate(y).reshape(-1, 3)
        for epoch in range(self.num_epochs):
            self.brain.fit(x, y, batch_size=self.batch_size, shuffle=True)

    def train(self, run=None):
        for training_session in range(self.num_training_sessions):
            print("This is training session number %s" % (training_session))
            print("The current exploration rate is %s" % (self.exploration_rate))
            print("The current learning rate is %s" % (self.learning_rate))
            self.play()
            self.tot_rewards_per_session[training_session] = self.rewards_during_session
            self.learn_from_memory()
            self.exploration_rate = max(self.min_exloration_rate, self.exploration_rate * self.exploration_rate_reduction)
            self.exploration_rates[training_session] = self.exploration_rate
            if training_session <= 2000:
                self.learning_rate = max(self.min_learning_rate, self.learning_rate * self.learning_rate_reduction)
            else:
                self.learning_rate = 0.05
            self.learning_rates[training_session] = self.learning_rate
        self.brain.save('trained_agent.h5')

    def make_model(self):
        inputs = keras.layers.Input(shape=(6,))
        x = keras.layers.Dense(256, activation='linear')(inputs)
        x = keras.layers.LeakyReLU(alpha=0.3)(x)
        x = keras.layers.Dense(512, activation='linear')(x)
        x = keras.layers.LeakyReLU(alpha=0.3)(x)
        predictions = keras.layers.Dense(3, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='Adam', loss='mean_squared_error')
        return model


agent = Agent(discount_factor=0.99, learning_rate=0.5, min_learning_rate=0.1, learning_rate_reduction=0.998, exploration_rate=0.9, min_exploration_rate=0.01, exploration_rate_reduction=0.997, num_episodes=1, num_steps=400, batch_size=10, num_epochs=1, num_training_sessions=3000, winning_reward=50)


def main():
    start_time = time.time()
    agent.train()
    agent.training_time = time.time() - start_time

    print('Training times are: ', agent.training_time)


if __name__ == '__main__':
    main()
