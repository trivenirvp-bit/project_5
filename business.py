#!/usr/bin/env python
# coding: utf-8

# In[34]:



# In[35]:


import matplotlib.pyplot as plt
x= [1,2,3,4]
y=[10,20,30,40]
plt.plot(x,y)
plt.title("test graph")
plt.show( )


# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[37]:


print("\n========Business context ========\n")


# In[38]:


print("""A company wants to predict sales based on advertising budget

Goal: build a Linear Progression model that predicts sales using Tv advertising budget """)


# In[39]:


data = {
        'TV_advertising': [50, 60, 70, 80, 90, 100,110, 120, 130, 140],
        'Sales': [5 ,6, 7, 9, 10, 12, 13, 14, 15,17 ]
       }
df = pd.DataFrame(data)

print(df)


# In[40]:


# visualizations
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plt.scatter(df['TV_advertising'],df['Sales'])
plt.xlabel("TV advertising Budget")
plt.ylabel("sales")
plt.title("Advertising vs sales")
plt.grid(True)
plt.show()
           


# In[41]:


df.info()


# In[42]:


df.describe( )# statistical summary


# In[43]:


df.isnull().sum() #missing values


# In[45]:


#X=[df['TV_advertising'] ]
#Y=[df['Sales']]


# In[51]:


X = df.drop("TV_advertising", axis=1)
Y = df["Sales"]


# In[52]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=43
)


# In[53]:


model=LinearRegression( )
model.fit(X_train,Y_train)
y_pred=model.predict(X_test)
print(" model training completed succesfully")


# In[55]:


#evaluation
mae = mean_absolute_error(Y_test,y_pred)
mse = mean_squared_error(Y_test,y_pred)
r2= r2_score(Y_test,y_pred)


# In[59]:


print(f"mean_absolute_error: {mae}")
print(f"mean_squared_error: {mse}")
print(f"r2_score: {r2}")


# In[60]:


plt.figure(figsize=(8,5))
plt.scatter(X_test,Y_test)
plt.plot(X_test,y_pred)
plt.xlabel("TV advertising Budget")
plt.ylabel("sales")
plt.title(" Linear Regression model prediction")
plt.grid(True)
plt.show( )


# In[68]:


#predict new value
new_budget = np.array([[170]])

predicted_sales = model.predict(new_budget)

print(f"If TV Advertising Budget = 170")
print(f"Predicted Sales = {predicted_sales[0]:.2f}")


# In[70]:


import joblib
joblib.dump(model,"Linear_Regression_model.pkl")
print("project completed")


# In[ ]:




