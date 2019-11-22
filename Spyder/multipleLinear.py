# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("insurance.csv")
print(data.columns)
print(data.describe())

# y axis
expenses = data.expenses.values.reshape(-1,1)
# x axis
ageBmis = data.iloc[:, [0,2]].values

regression = LinearRegression()
regression.fit(ageBmis, expenses)

print(regression.predict(np.array([[30, 20], [20, 21], [20, 22], [20, 23], [20, 24]])))
print("Accuracy: ", "{0:.2f}".format(r2_score(expenses, regression.predict(ageBmis))*100), "%")

