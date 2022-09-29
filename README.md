# Reinforcement Learning Automatic Trading - TFv1

We utilized reinforcement learning to perform automatic trading. Here, we implemented Deep Q-learning (DQL).
The project is implemented in tensorflow (TF) version 1.

There are several limitations in this project:
- We can only train and test on a single stock at a given time
- We can only perform one action per day (HOLD, BUY, or SELL)
- We can only buy or sell one single unit a day
- The simulation can end with units in the inventory
- The model is optimized based on the optimizing the actions, rather than maximizing the reward (profit)

The project is editted from:

https://www.analyticsvidhya.com/blog/2021/01/bear-run-or-bull-run-can-reinforcement-learning-help-in-automated-trading/

We editted the class to enable model saving and loading.
