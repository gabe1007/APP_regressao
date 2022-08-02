#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[2]:


def evauation_model(pred, y_val):
  score_MSE = round(mean_squared_error(pred, y_val),2)
  score_MAE = round(mean_absolute_error(pred, y_val),2)
  score_r2score = round(r2_score(pred, y_val),2)
  return score_MSE, score_MAE, score_r2score


# In[5]:


data = pd.read_csv('insurance.csv')

data['sex'] = data['sex'].replace(['male', 'female'], ['masculino', 'feminino'])
data['region'] = data['region'].replace(['southeast', 'southwest', 'northeast', 'northwest'], 
                                        ['sudeste', 'sudoeste', 'noroeste', 'nordeste'])
data['smoker'] = data['smoker'].replace(['yes', 'no'], ['sim', 'nao'])
data.rename(columns={'age':'idade', 'sex':'genero', 'bmi':'imc', 'children':'filhos', 'smoker':'fumante', 
                     'region':'regiao', 'charges':'total'}, inplace=True)


# In[6]:


X = data.drop('total', axis=1)
y = data['total']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[8]:


# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")


# In[9]:


pred = best_xgboost_model.predict(X_test)
score_MSE, score_MAE, score_r2score = evauation_model(pred, y_test)
print(score_MSE, score_MAE, score_r2score)


# In[ ]:




