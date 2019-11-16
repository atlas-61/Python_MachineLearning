# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

data = pd.read_csv("positions.csv")
#print(data.head())
#print(data.count())
#print(data.describe())

level = data.iloc[:,1].values.reshape(-1,1)
salary = data.iloc[:,2].values.reshape(-1,1)

regression = LinearRegression()
regression.fit(level, salary)

predict = regression.predict(np.array([8.3]).reshape(-1, 1))




regressionPoly = PolynomialFeatures(degree = 4)
levelPoly = regressionPoly.fit_transform(level)
regression2 = LinearRegression()
regression2.fit(levelPoly, salary)

value = regressionPoly.fit_transform(np.array([8.3]).reshape(-1,1))
predict2 = regression2.predict(value)

print("Linear Reggresion Accuracy: ", "{0:.2f}".format(r2_score(salary, regression.predict(level))*100), "%")
print("Polynominal Regression Accuracy: ", "{0:.2f}".format(r2_score(salary, regression2.predict(levelPoly))*100), "%")

plt.scatter(level, salary, color = "red")
plt.plot(level, regression.predict(level), color = "green")
plt.plot(level, regression2.predict(levelPoly), color = "blue")
plt.show()

