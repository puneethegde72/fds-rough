# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 20:49:15 2022

@author: Aroma
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

#data_array = np.genfromtxt('inputdata3.csv', delimiter=',', names=True) #skip_header = 1,
data_array = pd.read_csv('inputdata3.csv', header = 0, sep = ',')
print(data_array)

a = data_array["Rainfall"]       #.loc(axis=0)[:,]
b = data_array["Productivity"]   #loc(axis=1)[:,] #x = data[:,0] y = data[:,1]

model = LinearRegression()
X = a[:,np.newaxis] # .fit wants a matrix
X.shape

#t= np.array(X, reshape(-1,1), dtype= 'float')
t= np.reshape(X, data_array(-1,1), dtype= 'float')

model.fit(X,b)
y_pred = model.predict(X)

x_pred = 350
y_pred = model.predict(x_pred)
print(y_pred)

plt.scatter(a,b,color = 'black')
plt.plot(a,y_pred,color='blue')
plt.scatter(x_pred, y_pred, color= 'Red')

plt.xlabel("Amount of precipitations (mm per year)")  #To label the x-axis 
plt.ylabel("Productivity coefficient")  #To label the y-axis
plt.title("Scatter plot representing the rain per year and field productivity")  #To mention the title for the line graph
plt.legend()  #To show the label names in form of a box
plt.savefig('Scatter plot.png') #Saving the image of the line graph
plt.show()