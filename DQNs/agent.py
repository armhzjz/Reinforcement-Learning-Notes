# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# libraries used
import sys
import random
import gym
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

import matplotlib.pyplot as plt

# Set plotting options
#%matplotlib inline
#plt.style.use('ggplot')
#np.set_printoptions(precision=3, linewidth=120)



# let's create the environment

#env = gym.make('Acrobot-v1')
env = gym.make('CartPole-v1')
# we don't want to have a maximum number of steps
env._max_episode_steps = None
env.seed(1981)

# explorint the state (observation) space
print("State space: ", env.observation_space)
print(" - low:   ", env.observation_space.low)
print(" - high: ", env.observation_space.high)

# explorint the action space
print("Action space: ", env.action_space)



# DQN agent.
# This agen approximates the Q funtion by means of Neural Networks
# Memory replay and Fixed Q-targets are implemented in this agent
class DQN_agent:
    def __init__ (self, state_size, action_size,
                  gamma=0.99,
                  epsilon=1.0,
                  epsilon_min=0.01,
                  epsilon_decay=0.995,
                  learning_rate=0.001,
                  batch_size=64,
                  min_recollection_before_start_learning=1000,
                  render_mode=False,
                  load_weights=False,
                 running_mode='training'):
        # should the agent render see the environment in its changes made
        # by the agent 
        self.render_mode = render_mode
        # should we load existing weights?
        self.load_weights = load_weights
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.eps = epsilon
        self.eps_min = epsilon_min
        self.eps_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.min_recollection_before_start_learning = min_recollection_before_start_learning
        self.running_mode = running_mode
        
        # build the agent's models
        # target model to be build only if we are runnin in training mode
        if self.running_mode == 'training':
            self.model = self.build_neuralnet()
            self.target_model = self.build_neuralnet()
        else:
            self.model = self.build_neuralnet()
            self.load_w()
        
    def build_neuralnet(self):
        # the Q_values or Q function is approximated by the Neural Networks
        # that is the reason why the output of the neural net is linear with the same size as possible actions
        # The net receives as input a state of the environment and the output is the actions values (Q values of each action)
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def choose_action(self, state_s):
        # Choose action A from state S using the epsilon-greedy policy
        if (np.random.rand() <= self.eps) and (self.running_mode == 'training'):
            action_A = random.randrange(self.action_size)
            return action_A # this is a scalar!
        else:
            action_A = self.model.predict(state_s)
            return np.argmax(action_A[0]) # this is a scalar choosen out of an array!
    
    def store_experience(self, state, action, reward, next_state, done):
        # once the next tuple of the environment has been obtained, it has
        # to be "remembered".
        self.memory.append([state, action, reward, next_state, done])
        
    def replay(self):
        # The agent must learn now from the "remembered" experiences.
        # That is, it must learn now from a batch of stored state tuples
        
        if len(self.memory) < self.min_recollection_before_start_learning:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        update_in = np.zeros((self.batch_size, self.state_size))
        update_trget = np.zeros((self.batch_size, self.state_size))
        
        action, reward, done = [], [], []
        for i in range(self.batch_size):
            update_in[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            update_trget[i] = batch[i][3]
            done.append(batch[i][4])
            
        target_w = self.model.predict(update_in)
        target_w_minus = self.target_model.predict(update_trget)
        
        for i in range(self.batch_size):
            if done[i]:
                target_w[i][action[i]] = reward[i]
            else:
                target_w[i][action[i]] = reward[i] + self.gamma * np.amax(target_w_minus[i])
        
        self.model.fit(update_in, target_w, batch_size=self.batch_size, epochs=1, verbose=0)
        
        # update epsilon
        if self.eps >= self.eps_min:
            self.eps *= self.eps_decay
        
    def update_target_model_weights(self):
        # update the parameteres w of the target model
        # this is equivalent to updating w_minus to w in the DQ-Learning algorithm
        self.target_model.set_weights(self.model.get_weights())
        
    def save_w(self):
        self.model.save_weights('./save_model/acrobot_dqn.h5')
    
    def load_w(self):
        self.model.load_weights('./save_model/acrobot_dqn.h5')
        
        
        
if __name__ == "__main__":
    # set total number of episodes
    num_episodes = 5000
    # initial max average score
    max_avg_score = -np.inf
    
    # create our agent
    agent = DQN_agent(env.observation_space.shape[0], env.action_space.n, min_recollection_before_start_learning=100)
    
    # we want to monitor our scores
    # we want to make the nice graphs
    scores = []
    
    
    for episode in range(num_episodes):
        # initial score
        score = 0
        # set done to false
        done = False
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        while not done:
            if agent.render_mode:
                env.render()
            
            # choose action A from state S using policy epsilon-greedy
            action = agent.choose_action(state)
            # prepare next state
            next_state, reward, done,  _ = env.step(action)
            # Store experienced tuple (SARS') in replay memory
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            agent.store_experience(state, action, reward, next_state, done)
            # state_t+1 will be now state_t
            state = next_state
            agent.replay()
            
            #update my score
            score += reward
        
        #score = score if score <= 500 else score - 100
        scores.append(score)
        # update the fixed target (w minus)
        agent.save_w()
    
        if agent.running_mode == 'training':
            # if the episode is done, update the target model's weights before
            # starting the next episode
            agent.update_target_model_weights()
            
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score
            
            if episode % 100  == 0:
                print("\nEpisode {}/{} | Max. average score: {}".format(episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()
        env.close()