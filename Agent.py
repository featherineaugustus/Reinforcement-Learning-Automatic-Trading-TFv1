# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:14:51 2022

@author: WeiYanPEH
"""

import numpy as np
from collections import deque
import random
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()



###############################################################################
# Agent Class
###############################################################################
class Agent:
    def __init__(self, state_size, window_size, trend, skip, batch_size):
        
        # Initialize
        self.state_size = state_size            # Window size
        self.window_size = window_size          # Window size
        self.half_window = window_size // 2     # Half of window size
        self.trend = trend                      # Close train
        self.skip = skip                        # Skip = 1
        self.action_size = 3                    # Action = 3 
        self.batch_size = batch_size            # Batch size (training)
        self.memory = deque(maxlen = 1000)      # Memory
        self.inventory = []                     # Inventory (purchased)
        self.gamma = 0.95                       # Learning rate
        self.epsilon = 0.5                      # Probability to explore
        self.epsilon_min = 0.01                 # Minimum explore probability (0.01)
        self.epsilon_decay = 0.9999             # Decay ratio
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        
        # Initalize shape of training
        self.X = tf.placeholder(tf.float32, [None, self.state_size])   # Shape (,30)
        self.Y = tf.placeholder(tf.float32, [None, self.action_size])  # Shape (,3)
        # Action = {0: SIT, 1: BUY, 2: SELL}
        
        # Neural Network
        feed = tf.layers.dense(self.X, 256, activation = tf.nn.relu)
        self.logits = tf.layers.dense(feed, self.action_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.GradientDescentOptimizer(1e-5).minimize(
            self.cost
        )
        
        # Train
        self.sess.run(tf.global_variables_initializer())
        
        
    ###########################################################################
    # Act
    ###########################################################################
    # Choose what to do
    # Explore or Exploit
    def act(self, state):
        # Explore randomly
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        else:
            predict = self.sess.run(self.logits, feed_dict = {self.X: state})[0]
            return np.argmax(predict)
    
    
    ###########################################################################
    # Get State
    ###########################################################################
    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        
        # If enough windows, use all
        if d >= 0:
            block = self.trend[d : t + 1]
            
        # Else, fill the front with the first index
        else:
            block = -d * [self.trend[0]] + self.trend[0 : t + 1]
        
        # Get difference between i+1th and ith day
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
            
        return np.array([res])

    
    ###########################################################################
    # Replay
    ###########################################################################
    def replay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        
        # Retrain from memory, the last "batch_size" data
        for i in range(l - batch_size, l):
            mini_batch.append(self.memory[i])
            
        replay_size = len(mini_batch)
   
        X = np.empty((replay_size, self.state_size))
        Y = np.empty((replay_size, self.action_size))
        
        states = np.array([a[0][0] for a in mini_batch])
        new_states = np.array([a[3][0] for a in mini_batch])
        
        # Check Policy
        # Q is the logits
        Q = self.sess.run(self.logits, feed_dict = {self.X: states})
        Q_new = self.sess.run(self.logits, feed_dict = {self.X: new_states})
        
        for i in range(len(mini_batch)):
            state, action, reward, next_state, done = mini_batch[i]
            target = Q[i]
            target[action] = reward
            
            # Update target
            if not done:
                target[action] += self.gamma * np.amax(Q_new[i])
                
            X[i] = state
            Y[i] = target
            
        # Update Policy
        cost, _ = self.sess.run(
                                [self.cost, self.optimizer], 
                                feed_dict = {self.X: X, self.Y: Y}
                                )
        #print('Cost:', cost)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return cost     
                


    ###########################################################################
    # Train
    ###########################################################################
    def train(self, iterations, checkpoint, initial_money):
        for i in range(iterations):
            total_profit = 0                # Total profit throughout the run
            inventory = []                  # Bought what
            state = self.get_state(0)       # The past 29 days of price change
            current_money = initial_money  # Total amount of cash
            
            for t in range(0, len(self.trend) - 1, self.skip):
                action = self.act(state)
                next_state = self.get_state(t + 1)
                
                #print('Time:', t)
                #print('Action:', action)
                #print('Next State:', next_state)
                
                if ((action == 1) and                              # If model choose to buy
                    (current_money >= self.trend[t]) and           # If have enough to buy
                    (t < (len(self.trend) - self.half_window))     # Cannot buy if only left 15 days
                    ):
                    inventory.append(self.trend[t])                # Save bought price
                    current_money -= self.trend[t]                 # Spent some money
                    
                elif ((action == 2) and                            # If model choose to sell
                      (len(inventory) > 0)                         # If there is stock to sell
                      ):
                    bought_price = inventory.pop(0)                # Take the oldest stock
                    total_profit += (self.trend[t] - bought_price) # Update profit
                    current_money += self.trend[t]                 # Update total money
                
                # Compute
                invest = ((current_money - initial_money) / initial_money)
                
                self.memory.append((state,
                                    action, 
                                    invest, 
                                    next_state, 
                                    current_money < initial_money))
                state = next_state
                batch_size = min(self.batch_size, len(self.memory))
                
                # Train the model
                cost = self.replay(batch_size)
                
            inventory_worth_bought = np.sum(inventory)
            inventory_worth_current = len(inventory)*self.trend[t]
            
            net_worth = initial_money + inventory_worth_current
                
            #print('\nBalance $%.3f' % (initial_money))
            #print('Holdings: %d, Current Worth: $%.3f' % (len(inventory), inventory_worth_current))
            #print('Holdings: %d, Purchase Cost: $%.3f' % (len(inventory), inventory_worth_bought))
            #print('Networth: $%.3f' % (net_worth))
                
            #if (i+1) % checkpoint == 0:
            print('Epoch: %d, Cost: %.3f, Profit: $%.1f, Total Money: $%.1f, Inventory: %d (%.1f)'
                  %(i + 1, cost, total_profit, current_money, 
                    len(inventory), inventory_worth_current))
                
        # Now, save the graph
        saver = tf.train.Saver()
        saver.save(self.sess, 'Checkpoints/my_test_model')
            
            
    ###########################################################################
    # Buy
    ###########################################################################
    def buy(self, initial_money, trend_test):
        current_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        state = self.get_state(0)
        
        for t in range(0, len(trend_test) - 1, self.skip):
            action = self.act(state)
            next_state = self.get_state(t + 1)
            
            # BUY
            if ((action == 1) and                              # If model choose to buy
                (current_money >= trend_test[t]) and           # If have enough to buy
                (t < (len(trend_test) - self.half_window))     # Cannot buy if only left 15 days
                ):
                inventory.append(trend_test[t])                # Save bought price
                current_money -= trend_test[t]                 # Spent some money
                states_buy.append(t)
                
                print('Day %d: BUY  1 unit at $%.3f, total balance $%.3f'
                      % (t, trend_test[t], current_money))
                
            # SELL
            elif ((action == 2) and                            # If model choose to sell
                  (len(inventory) > 0)                         # If there is stock to sell
                  ):
                bought_price = inventory.pop(0)                # Take the oldest stock
                current_money += trend_test[t]                 # Update money
                states_sell.append(t)                          # Sell state
                
                try:
                    invest = ((trend_test[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                    
                print('Day %d: SELL 1 unit at $%.3f, investment %.3f%%, total balance $%.3f'
                      % (t, trend_test[t], invest, current_money))
            
            # SIT
            else:
                print('Day %d: SIT  %d, total balance $%.3f'
                      % (t, len(inventory), current_money))
                
            state = next_state
        
        inventory_worth_bought = np.sum(inventory)
        inventory_worth_current = len(inventory)*trend_test[t]
        
        net_worth = current_money + inventory_worth_current
            
        print('\nBalance $%.3f' % (current_money))
        print('Holdings: %d, Current Worth: $%.3f' % (len(inventory), inventory_worth_current))
        print('Holdings: %d, Purchase Cost: $%.3f' % (len(inventory), inventory_worth_bought))
        print('Networth: $%.3f' % (net_worth))
            
        invest = ((net_worth - initial_money) / initial_money) * 100
        total_gains = net_worth - initial_money
        
        return (
                states_buy, states_sell, 
                inventory_worth_bought, inventory_worth_current,
                total_gains, invest, net_worth
                )
                
    
                