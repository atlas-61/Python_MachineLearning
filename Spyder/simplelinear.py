# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("hw_25000.csv")

height = data.Height.values.reshape(-1,1)
weight = data.Weight.values.reshape(-1,1)

regression = LinearRegression()
regression.fit(height,weight)
print(regression.predict(np.array([[60]])))
print(regression.predict(np.array([62]).reshape(-1, 1)))
print(regression.predict(np.array([64]).reshape(-1, 1)))
print(regression.predict(np.array([66]).reshape(-1, 1)))
print(regression.predict(np.array([68]).reshape(-1, 1)))
print(regression.predict(np.array([70]).reshape(-1, 1)))


print(data.columns)



plt.scatter(data.Height, data.Weight, color = "gray")
# 2 methods for show to linear model
x = np.arange(min(data.Height), max(data.Height)).reshape(-1,1)
plt.plot(data.Height, regression.predict(height), color = "red")
plt.plot(x, regression.predict(x), color = "green")

plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Simple Linear Regression Model")
plt.show()

print("Accuracy: ", "{0:.2f}".format(r2_score(weight, regression.predict(height))*100), "%")

