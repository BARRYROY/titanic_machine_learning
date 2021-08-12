#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


# In[2]:


raw_data = pd.read_csv('train.csv')
raw_data.head()


# In[3]:


raw_data.describe()


# In[4]:


raw_data.isnull().sum()


# In[5]:


raw_data = raw_data.drop(columns=['PassengerId', 'Name', 'Cabin', 'Fare'])
raw_data


# In[6]:


raw_data['Sex'] = raw_data['Sex'].map({'male':1, 'female':0})
raw_data['Embarked'] = raw_data['Embarked'].map({'S':0, 'C':1, 'Q':2})


# In[7]:


raw_data['Ticket'] = pd.to_numeric(raw_data.iloc[:,-2], errors='ignore')


# In[8]:


plt.scatter(raw_data['Ticket'], raw_data['Survived'])
plt.show()


# In[9]:


raw_data.head()


# In[10]:


raw_data['Embarked'].unique()


# ### Dealing with msiing data

# In[11]:


def missing_percentage(df):
    nan_percent = 100*(raw_data.isnull().sum()/len(raw_data))
    nan_percent = nan_percent[nan_percent>0].sort_values()
    return nan_percent


# In[12]:


missing_percentage(raw_data)


# In[13]:


raw_data['Age'].isnull().sum()


# In[14]:


raw_data['Age'].describe()


# In[15]:


raw_data['Age'].mean()


# In[16]:


import math
female_mean, male_mean = raw_data.groupby('Sex')['Age'].mean()
def fill_age (age, sex):
    if math.isnan(age):
        if sex == 'male':
            return male_mean
        else:
            return female_mean
    else:
        return age


# In[17]:


raw_data['Age'] = raw_data.apply(lambda row: fill_age(row['Age'], row['Sex']), axis=1)


# In[18]:


raw_data['Age'].isnull().sum()


# In[19]:


raw_data['Embarked'].isnull().sum()


# In[20]:


raw_data = raw_data.dropna()
raw_data.head(10)


# In[21]:


sns.pairplot(raw_data)


# In[22]:


X = raw_data.drop(columns=['Survived', 'Ticket'])
y = raw_data['Survived']


# In[23]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8)


# In[24]:


scaler = StandardScaler()


# In[25]:


scaler.fit(X_train)


# In[26]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[27]:


model = LogisticRegression()


# In[28]:


model.fit(X_train,y_train)


# In[29]:


y_pred = model.predict(X_test)
y_pred


# In[30]:


model.score(X_train,y_train)


# In[31]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix


# In[32]:


confusion_matrix(y_test,y_pred)


# In[33]:


accuracy_score(y_test, y_pred)


# In[34]:


print(classification_report(y_test, y_pred))


# ### KNN Method

# In[35]:


from sklearn.neighbors import KNeighborsClassifier
test_error_rate = []

for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    
    y_pred_knn = knn.predict(X_test)
    
    e = 1 - accuracy_score(y_test, y_pred_knn)
    test_error_rate.append(e)


# In[36]:


test_error_rate


# In[37]:


plt.figure(figsize=(10,6))
plt.plot(range(1,30), test_error_rate, label='Test Error')
plt.legend()
plt.ylabel('Error Rate')
plt.xlabel('K Value')
plt.show()


# In[38]:


knn_model_7 = KNeighborsClassifier(n_neighbors=7)
knn_model_7.fit(X_train,y_train)
y_knn_pred_7 = knn_model_7.predict(X_test)


# In[39]:


knn_model_15 = KNeighborsClassifier(n_neighbors=15)
knn_model_15.fit(X_train,y_train)
y_knn_pred_15 = knn_model_15.predict(X_test)


# In[40]:


print(classification_report(y_test, y_knn_pred_7))


# In[41]:


print(classification_report(y_test,y_knn_pred_15))


# ### Testing The Model

# In[42]:


model.score(X_test, y_test)


# In[43]:


predicted_proba = model.predict_proba(X_test)
predicted_proba


# In[44]:


import pickle

with open('model','wb')as file:
    pickle.dump(model, file)
    
with open('scaler', 'wb') as file:
    pickle.dump(scaler, file)


# In[ ]:




