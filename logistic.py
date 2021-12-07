#Reyes, Kerwiwn & Navarro, Jeric
#CS - 302

import math
import numpy as np
import pandas as pd
import scipy.optimize as optim
import matplotlib.pyplot as plt

#import the data
data = pd.read_csv('Philippine_data_logistic.csv', sep=';')
data = data['total_cases']
data = data.reset_index(drop=False)
data.columns = ['Timestep', 'Total Cases']

#Define the function with the coeficients to estimate
def my_logistic(t, a, b, c):
    return c / (1 + a * np.exp(-b*t))

#Randomly initialize the coefficients
p0 = np.random.exponential(size=3)

#Set min bound 0 on all coefficients, and set different max bounds for each coefficient
bounds = (0, [100000., 3., 1000000000.])

#convert pd.Series to np.Array and use Scipy's curve fit to find ther best Non linear Lease Aquares coefficients
x = np.array(data['Timestep']) + 1
y = np.array(data['Total Cases'])

#Show the coefficeints
(a,b,c),cov = optim.curve_fit(my_logistic, x, y, bounds=bounds, p0=p0)

print(a,b,c)

#redefine the fuction with the new a,b and c
def my_logistic(t):
    return c / (1 + a * np.exp(-b*t))

#Plot
plt.scatter(x, y)
plt.plot(x, my_logistic(x))
plt.title('Logistic Model vs Real Observations of Philippine Coronavirus')
plt.legend([ 'Logistic Model', 'Real data'])
plt.xlabel('Time')
plt.ylabel('Infections')

plt.show()
