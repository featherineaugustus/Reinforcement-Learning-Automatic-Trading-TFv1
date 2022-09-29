# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:12:09 2022

@author: WeiYanPEH

https://www.analyticsvidhya.com/blog/2021/01/bear-run-or-bull-run-can-reinforcement-learning-help-in-automated-trading/
"""

#%%############################################################################
# Import libraries
###############################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()
#!pip install yfinance --upgrade --no-cache-dir
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from collections import deque
import random
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

from Agent import Agent

plt.close('all')
tf.InteractiveSession()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))



#%%############################################################################
# Create Folders
###############################################################################
if not os.path.exists('Data'):
    os.makedirs('Data')
if not os.path.exists('Results'):
    os.makedirs('Results')
if not os.path.exists('Checkpoints'):
    os.makedirs('Checkpoints')

#%%############################################################################
# Extract data
###############################################################################
#yf.pdr_override()
df_train = pdr.get_data_yahoo('TSLA', 
                              start='2015-01-01',
                              end='2022-01-01',
                              interval='d')
df_train = df_train.reset_index()
df_train.to_csv('Data/TSLA.csv',index=False)
print(df_train.head())
print(len(df_train))

close_train = df_train['Close'].values.tolist()

#%%############################################################################
# Initalize variables
###############################################################################
initial_money = 10000
window_size = 30
skip = 1
batch_size = 32


#%%############################################################################
# Call Agent
###############################################################################
agent = Agent(state_size = window_size, 
              window_size = window_size, 
              trend = close_train,
              skip = skip, 
              batch_size = batch_size)


#%%############################################################################
# Train Agent
###############################################################################
# agent.train(iterations = 200, 
#             checkpoint = 10, 
#             initial_money = initial_money)




#%%############################################################################
# Load Agent
###############################################################################
agent = Agent(state_size = window_size, 
              window_size = window_size, 
              trend = close_train,
              skip = skip, 
              batch_size = batch_size)

saver = tf.compat.v1.train.Saver()

# Later, launch the model, initialize the variables, do some work, save the
# variables to disk.
sess = tf.compat.v1.InteractiveSession()
# Do some work with the model.
# Save the variables to disk.
save_path = saver.restore(sess, 'Checkpoints/my_test_model')
print("Model saved in file: %s" % save_path)

agent.sess = sess

#%%############################################################################
# Perform prediction (TEST)
###############################################################################
company_list = ['TSLA', 'MSFT', 'AMZN',
                'AAPL','GOOG','FB', 
                'AMD', 'ADSK', 'BLK',
                'HPQ', 'IBM', 'JPM',
                'META', 'MU', 'NVDA',
                'PYPL', 'TWTR', 'USB',
                'V'
                ]

results = []

for company in company_list:

    df_test = pdr.get_data_yahoo(company, 
                                 start='2020-01-01',
                                 end='2022-01-01',
                                 interval='d')
    df_test = df_test.reset_index()
    df_test.to_csv('Data/' + company + '.csv',index=False)
    print(df_test.head())
    print(len(df_test))
    
    close_test = df_test['Close'].values.tolist()
    
    
    (states_buy, 
     states_sell,
     inventory_worth_bought, 
     inventory_worth_current,
     total_gains, 
     invest,
     net_worth) = agent.buy(initial_money = initial_money,
                            trend_test = close_test)
    
    results.append([company, 
                    len(states_buy), 
                    len(states_sell),
                    initial_money,
                    net_worth,
                    total_gains,
                    invest,
                    inventory_worth_bought, 
                    inventory_worth_current]
                   )
                         
    # Plot results
    fig = plt.figure(figsize = (15,5))
    
    plt.plot(close_test, color='r', lw=2.)
    
    plt.plot(close_test, '^', 
             markersize=10, 
             color='m', 
             label = 'buying signal', 
             markevery = states_buy)
    
    plt.plot(close_test, 'v', 
             markersize=10, 
             color='k', label = 
             'selling signal',
             markevery = states_sell)
    
    plt.title(company + ' - ' + 
              'NW: \$' + str(round(net_worth,3)) + ' - ' +
              'Total gains: \$' + str(round(total_gains,1)) + ' - ' + 
              'Investment Rate: ' + str(round(invest,3)) + '% - ' + 
              'Buy: ' + str(len(states_buy)) + ' - ' + 
              'Sell: ' + str(len(states_sell))
              )


    plt.xlabel('Days')
    plt.ylabel('Price')
    
    plt.legend()
    plt.savefig('Results/' + company + '.png')
    plt.show()

results = pd.DataFrame(results, 
                       columns=['Company', 'Buy', 'Sell', 
                                'Initial Captial',
                                'Networth',
                                'Gains',
                                'Investment %',
                                'Inventory (bought)', 'Inventory (sell)'])

results.to_csv('Results/Results.csv', index=False)

#%%############################################################################
# Plot Results
###############################################################################
results.head()
results = pd.read_csv('Results/Results.csv')
results = results.sort_values('Company', ascending=[False])

plt.close('all')
ax = results.plot.barh(x='Company', y='Investment %', figsize=(10,6))
plt.title('Profit')
plt.xlabel('Investment %')















