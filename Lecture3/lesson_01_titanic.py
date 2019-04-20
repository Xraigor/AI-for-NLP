# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 09:57:16 2019

@author: Xraigor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# import data and drop nan values
content = pd.read_csv('E:/NLP/train.csv')
content = content.dropna()

# store the age and fare data as list type variable
ages = content['Age'].tolist()
fares = content['Fare'].tolist()

# filter the variables within range to better fit linear Regression
age_with_fares = content[
    (content['Age'] > 22) & (content['Fare'] < 400) & (content['Fare'] > 130)
    ]
sub_fare = age_with_fares['Fare']
sub_age = age_with_fares['Age']


# plt.scatter(sub_age, sub_fare)
# plt.show()

# Linear equations
def func(age, k, b): return k * age + b


# loss function
def loss(y, yhat):
    """

    :param y: the real fares
    :param yhat: the estimated fares
    :return: how good is the estimated fares
    """
    return np.mean(np.abs(y - yhat))


#   return np.mean(np.square(y - yhat))
#   return np.mean(np.sqrt(y - yhat))

# initialize the infinite number
min_error_rate = float('inf')

# best_k, best_b = None, None

# search times
loop_times = 10000
losses = []

# change searching direction method
change_directions = [
    # (k, b)
    (+1, -1),  # k increase, b decrease
    (+1, +1),
    (-1, +1),
    (-1, -1)  # k decrease, b decrease
]

# initialize the LR function's parameter
k_hat = random.random() * 20 - 10
b_hat = random.random() * 20 - 10

# initialize the better parameter when searching convergence functions
best_k, best_b = k_hat, b_hat
best_direction = None


# step function as Step size (scalar)
def step(): return random.random() * 1


# initalize the first searching directions
direction = random.choice(change_directions)


# define the derivate functions of k and b parameter to calculate the
# back propagation functions to find the best convergence function
def derivate_k(y, yhat, x):
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]

    return np.mean([a * -x_i for a, x_i in zip(abs_values, x)])


def derivate_b(y, yhat):
    abs_values = [1 if (y_i - yhat_i) > 0 else -1 for y_i, yhat_i in zip(y, yhat)]
    return np.mean([a * -1 for a in abs_values])


# back propagation rate (like the scale of changing steps)
learning_rate = 1e-1

# back propagation method
while loop_times > 0:
    # method of arbitrary changing directions(very powerful in low dimention cases)
    #    k_delta_direction, b_delta_direction = best_direction or random.choice(change_directions)

    #    k_delta = k_delta_direction * step()
    #    b_delta = b_delta_direction * step()

    #    new_k = k_hat + k_delta
    #    new_b = b_hat + b_delta

    #    estimated_fares = func(sub_age, new_k, new_b)
    #    error_rate = loss(y=sub_fare, yhat=estimated_fares)
    #    print(error_rate)

    #    if error_rate < min_error_rate:
    #        min_error_rate = error_rate
    #        best_k, best_b = k_hat, b_hat

    #        direction = (k_delta_direction, b_delta_direction)
    #        print(min_error_rate)

    #        print('loop == {}'.format(1000 - loop_times))
    #        losses.append(min_error_rate)
    #        print('f(age) = {} * age + {}, with error rate : {}'.format(best_k, best_b, min_error_rate))
    #    else:
    #        direction = random.choice(change_directions)
    #    loop_times -= 1

    k_delta = -1 * learning_rate * derivate_k(sub_fare, func(sub_age, k_hat, b_hat), sub_age)
    b_delta = -1 * learning_rate * derivate_b(sub_fare, func(sub_age, k_hat, b_hat))

    k_hat += k_delta
    b_hat += b_delta

    estimated_fares = func(sub_age, k_hat, b_hat)
    error_rate = loss(y=sub_fare, yhat=estimated_fares)

    print('loop == {}'.format(loop_times))

    print('f(age) = {} * age + {}, with error rate: {}'.format(k_hat, b_hat, error_rate))

    losses.append(error_rate)

    loop_times -= 1

plt.scatter(sub_age, sub_fare)
# plt.plot(sub_age, func(sub_age, best_k, best_b), c='r')
plt.plot(sub_age, func(sub_age, k_hat, b_hat), c='r')
plt.show()

# show the convergence that means this function f is learnable
# plt.plot(range(len(losses)), losses)
# plt.show()