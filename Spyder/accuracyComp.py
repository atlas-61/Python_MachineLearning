# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

data = pd.read_csv("hw_25000(cm_kg).csv")


#df_test = data.loc[(data['Height'] >= 190) & (data['Height'] < 191)].copy()
#print(df_test["Weight"].mean())

value = np.array(176).reshape(-1,1)
print(data.columns)

height = data.Height.values.reshape(-1,1)
weight = data.Weight.values.reshape(-1,1)
ranForWeight = data.Weight.values

linRegression = LinearRegression()
linRegression.fit(height, weight)
print("Linear Regression Prediction", linRegression.predict(value))
print("Linear Regression Accuracy: ", "{0:.2f}".format(r2_score(weight, linRegression.predict(height))*100), "%")


polyRegression = PolynomialFeatures(degree = 3)
heightPoly = polyRegression.fit_transform(height)
fittedValue = polyRegression.fit_transform(value)
polyRegression2 = LinearRegression()
polyRegression2.fit(heightPoly, weight)

print("Polynominal Regression Prediction", polyRegression2.predict(fittedValue))
print("Polynominal Regression Accuracy: ", "{0:.2f}".format(r2_score(weight, polyRegression2.predict(heightPoly))*100), "%")

randForRegression = RandomForestRegressor(n_estimators = 1000)
randForRegression.fit(height, ranForWeight)

print("Random Forest Prediction", randForRegression.predict(value))
print("Accuracy: ", "{0:.2f}".format(r2_score(ranForWeight, randForRegression.predict(height))*100), "%")


