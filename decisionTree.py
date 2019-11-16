# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

data = pd.read_csv("positions.csv")
print(data.columns)

level = data.Level.values.reshape(-1,1)
salary = data.Salary.values.reshape(-1,1)

regression = DecisionTreeRegressor()
regression.fit(level, salary)

print(regression.predict(np.array(8.5).reshape(-1,1)))
print("Accuracy: ", "{0:.2f}".format(r2_score(salary, regression.predict(level))*100), "%")

plt.scatter(level, salary, color="red")
x = np.arange(min(level), max(level),0.01).reshape(-1,1)
plt.plot(x, regression.predict(x), color = "green")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
