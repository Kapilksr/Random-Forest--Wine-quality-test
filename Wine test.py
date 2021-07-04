#!/usr/bin/env python
# coding: utf-8

# ## Import relevant libraries

# In[105]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config
set_config(print_changed_only=False)


# ## Load the data

# In[106]:


df=pd.read_csv('winequality-red.csv')
df.head()


# ## eyeball the data

# In[107]:


df1=df.copy()
df


# In[108]:


df1.info()


# In[109]:


## we see that the data doesn't have null values


# In[110]:


df1.describe()


# ## Data Preprocessing and Cleaning

# In[111]:


## make a new column 'wine_quality' where we map the values of quality as 1 or 0...we would like to assume that -
## - quality better than or equal to 7 is good i.e. 1 and quality less than that is not upto the marks i.e. 0
df1['wine_quality']=np.where(df1['quality']>=7,1,0)


# In[112]:


df1.head(10)


# In[113]:


# check if dataset is balanced (what % of targets are 1s)
# targets.sum() will give us the number of 1s that there are
# the shape[0] will give us the length of the targets array
df1.wine_quality.sum()/df1.shape[0]


# In[114]:


df1.wine_quality.sum()


# In[115]:


df1.shape


# In[116]:


##data is imbalanced as only 217 values are from class of 1s and others from class of 0s


# In[117]:


##let's plot the countplot to visualize this abnormality better 
sns.countplot(data=df1,x='wine_quality')


# In[118]:


sns.countplot(data=df1,x='quality')


# In[119]:


## we can drop the 'quality' column  as we have created another feature for our target variable
df1.drop('quality',axis=1,inplace=True)


# ## EDA

# In[120]:


## to check the correlation of variables with one another and with the target variable
plt.figure(figsize = (10, 6))
sns.heatmap(df1.corr(), annot = True)


# In[121]:


## we see that wine quality is highly correlated with the alcohol


# # Define our targets and inputs

# In[122]:



x=df1.drop(['wine_quality'],axis=1)
y=df1['wine_quality']


# # create test and train dataset

# In[123]:


from sklearn.model_selection import train_test_split


# In[124]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=500)


# In[125]:


print(x_train.shape,y_train.shape)


# In[126]:


print(x_test.shape,y_test.shape)


# # import the model and fit on training dataset

# In[127]:


from sklearn.ensemble import RandomForestClassifier


# In[128]:


rfc=RandomForestClassifier()


# In[129]:


print(rfc.get_params)


# In[130]:


rfc.fit(x_train,y_train)


# In[131]:


## let's predict  the values on test dataset
y_pred=rfc.predict(x_test)


# ## check the accuracy of the model

# In[132]:


from sklearn import metrics


# In[133]:


metrics.accuracy_score(y_test,y_pred)


# In[134]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
print(confusion_matrix)


# In[135]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# ## let's use K-fold validation

# In[136]:


from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)


# In[137]:


scores = cross_val_score(rfc, x, y, scoring='accuracy', cv=cv, n_jobs=-1)


# In[138]:


scores


# In[139]:


scores.mean() ## our mean accuracy


# ## as we saw that the dataset was imbalanced we can balance it with SMOTE

# In[140]:


from imblearn.over_sampling import SMOTE


# In[141]:


oversample=SMOTE()


# In[142]:


x,y=oversample.fit_resample(x,y)


# In[143]:


y.sum()


# In[144]:


y.shape


# In[145]:


x.shape


# In[146]:


y.sum()/y.shape[0]


# In[147]:


## now we see that the dataset is balanced


# ## now we re run all the commands to see if our model performed better

# In[148]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=400)


# In[149]:


print(x_train.shape,y_train.shape)


# In[150]:


print(x_test.shape,y_test.shape)


# In[151]:


rfc=RandomForestClassifier()


# In[152]:


rfc.fit(x_train,y_train)


# In[153]:


y_pred=rfc.predict(x_test)


# In[154]:


metrics.accuracy_score(y_test,y_pred)


# In[155]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
print(confusion_matrix)


# In[156]:


print(classification_report(y_test,y_pred))


# In[157]:


cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(rfc, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores


# In[158]:


scores.mean()


# ### we see that our accuracy has increased a bit... let's run a grid search to tune the hyperparameters

# In[91]:


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [ 3,5,10,None],
    'n_estimators': [100,200,300,400,500],
    'max_features': [1,2,3,4,5,6,7,8,9],
    'criterion':['gini','entropy'],
    'min_samples_leaf': [1,2,3, 4, 5],
    
   
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(x_train,y_train)


# In[159]:


## to find our best parameters
grid_search.best_params_


# In[160]:


## use these parameters for our model
rfc1=RandomForestClassifier(bootstrap= True,criterion='entropy',
                           max_depth= None,max_features= 1,min_samples_leaf= 1,n_estimators= 300)


# In[161]:


rfc1.fit(x_train,y_train)


# In[162]:


y_pred=rfc1.predict(x_test)


# In[163]:


metrics.accuracy_score(y_test,y_pred)


# In[164]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
print(confusion_matrix)


# In[165]:


## we see that our results have improved as we decreased the number of false positives


# In[166]:


print(classification_report(y_test,y_pred))


# # Now we save our model

# In[103]:


import pickle


# In[167]:


with open('model','wb') as file:
    pickle.dump(rfc1,file)


# In[ ]:




