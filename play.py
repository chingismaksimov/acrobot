import numpy as np
import gym
import keras
import copy
import os
from keras.models import load_model

model_name = "trained_agent.h5"
agent = load_model(model_name)

number_of_tests = 150

env = gym.make('Acrobot-v1')
while True:
    observation = env.reset()
    done = False
    for step in range(number_of_tests):
        env.render()
        action = np.argmax(agent.predict(observation.reshape(1, -1)).flatten())
        observation, reward, done, _ = env.step(action)
        if done:
            print("The episode finished in ", step + 1, " steps")
            break
        if step == number_of_tests - 1:
            print("The episode did not finish")
