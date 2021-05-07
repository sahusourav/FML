#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm

# Download the SIPEstimator.xlsx file.
# Copy and paste the PATH in the plca of --PATH--

df = pd.read_excel(r"--PATH--", columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_Index_Price'])
ir = pd.DataFrame(df, columns = ['Interest_Rate'])
ur = pd.DataFrame(df, columns = ['Unemployment_Rate'])
sip = pd.DataFrame(df, columns = ['Stock_Index_Price'])

# Plotting Interest Rate vs Stock Index Price

plt.scatter(ir, sip, color = "r", marker = "o", s = 25)
plt.xlabel("Interest Rate")
plt.ylabel("Stock Index Price")
plt.show()


# Plotting Unemployment Rate vs Stock Index Price


plt.scatter(ur, sip, color = "b", marker = "x", s = 30)
plt.xlabel("Unemployment Rate")
plt.ylabel("Stock Index Price")
plt.show()


X = df[['Interest_Rate', 'Unemployment_Rate']]
Y = df['Stock_Index_Price']


# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# prediction with sklearn
New_Interest_Rate = 2.55
New_Unemployment_Rate = 5.4
print ('Predicted Stock Index Price: \n', regr.predict([[New_Interest_Rate, New_Unemployment_Rate]]))


# with statsmodels
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 


print_model = model.summary()
print(print_model)
