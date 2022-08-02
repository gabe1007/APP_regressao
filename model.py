#!/usr/bin/env python
# coding: utf-8

# In[124]:


import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import warnings
warnings.filterwarnings(action = 'ignore')


# In[125]:


def evauation_model(pred, y_val):
  score_MSE = round(mean_squared_error(pred, y_val),2)
  score_MAE = round(mean_absolute_error(pred, y_val),2)
  score_r2score = round(r2_score(pred, y_val),2)
  return score_MSE, score_MAE, score_r2score


# In[126]:


data = pd.read_csv('insurance.csv')


# In[127]:


data['sex'] = data['sex'].replace(['male', 'female'], ['masculino', 'feminino'])
data['region'] = data['region'].replace(['southeast', 'southwest', 'northeast', 'northwest'], 
                                        ['sudeste', 'sudoeste', 'noroeste', 'nordeste'])
data['smoker'] = data['smoker'].replace(['yes', 'no'], ['sim', 'nao'])
data.rename(columns={'age':'idade', 'sex':'genero', 'bmi':'imc', 'children':'filhos', 'smoker':'fumante', 
                     'region':'regiao', 'charges':'total'}, inplace=True)


# In[128]:


# Vamos utilizar o método get_dummies para converter as variáveis categóricas 
data = pd.get_dummies(data)

# cria as variáveis X e y
X = data.drop('total', axis=1).values
y = data.total.values


# In[129]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[130]:


def evaluate_model(model, X, y):
    kf = KFold(n_splits=12, random_state=42, shuffle=True)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=kf, n_jobs=-1, error_score='raise')
    return scores


# In[131]:


#Utilizaremos dois modelos baseline

def get_models():
  models = dict()

  models['Xgb'] = XGBRegressor()
  
  models['RFR'] = RandomForestRegressor()

  return models


# In[132]:


models = get_models()

# avaliar os modelos e guardar os resultados 
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
 
# plotar o desempenho para comparação
plt.boxplot(results, labels=names, showmeans=True)
plt.show()


# In[147]:


xgb_model = XGBRegressor().fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)


# In[148]:


score_MSE, score_MAE, score_r2score = evauation_model(y_pred,y_test)


# In[149]:


xgb_result_scores = pd.DataFrame(columns=["model","mse","mae","r2score"])
to_append = ["XGB",score_MSE, score_MAE, score_r2score]
df_result_scores.loc[len(to_append)] = to_append


# In[150]:


df_result_scores


# In[151]:


xgb_model.save_model("best_model.json")


# In[ ]:




