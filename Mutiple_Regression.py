#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm


# In[21]:


df = pd.read_excel(r"D:\e-books\6th Sem\FML\LABS\SIPEstimator.xlsx", columns=['Year','Month','Interest_Rate','Unemployment_Rate','Stock_Index_Price'])
ir = pd.DataFrame(df, columns = ['Interest_Rate'])
ur = pd.DataFrame(df, columns = ['Unemployment_Rate'])
sip = pd.DataFrame(df, columns = ['Stock_Index_Price'])


# In[17]:


# Plotting Interest Rate vs Stock Index Price


# In[16]:


plt.scatter(ir, sip, color = "r", marker = "o", s = 25)
plt.xlabel("Interest Rate")
plt.ylabel("Stock Index Price")
plt.show()


# In[4]:


# Plotting Unemployment Rate vs Stock Index Price


# In[8]:


plt.scatter(ur, sip, color = "b", marker = "x", s = 30)
plt.xlabel("Unemployment Rate")
plt.ylabel("Stock Index Price")
plt.show()


# In[24]:


X = df[['Interest_Rate', 'Unemployment_Rate']]
Y = df['Stock_Index_Price']


# In[25]:


# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# In[35]:


# prediction with sklearn
New_Interest_Rate = 2.55
New_Unemployment_Rate = 5.4
print ('Predicted Stock Index Price: \n', regr.predict([[New_Interest_Rate, New_Unemployment_Rate]]))


# In[27]:


X = sm.add_constant(X)


# In[28]:


model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 


# In[29]:


print_model = model.summary()
print(print_model)


# In[ ]:




