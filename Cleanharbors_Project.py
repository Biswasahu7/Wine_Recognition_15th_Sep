#!/usr/bin/env python
# coding: utf-8

# # Cleanharbors_Project

# In[1]:


# Importing necessary libraries...
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import csv


# In[2]:


# Convert raw data to csv format...(One time job)
# read_file = pd.read_csv(r'C:\Users\BISWA\Documents\wine_txt.txt')
# dataframe1.to_csv(r'C:\Users\BISWA\Documents\winedata.csv', index=None)


# In[3]:


# Reading data set from local PC...
df=pd.read_csv(r'C:\Users\BISWA\Documents\winedata.csv',header=None)

# Display first 5 line of dataset...
df.head(5)


# In[4]:


# Checking the dataset shape...
print(df.shape)


# In[5]:


# Adding header as per the txt file...
df.columns = ['class identifier','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids'
             	,'Nonflavanoid phenols','Proanthocyanins','Color intensity Intensity','Hue','OD280/OD315 of diluted wines','Proline']


# In[6]:


# Save new dataframe with all columns into csv format...(One time job)
# df.to_csv(r'C:\Users\BISWA\Documents\Raw.csv')


# In[7]:


# Display header...
df.head(5)


# In[8]:


# Checking null value from the dataset...
df.isnull().sum()


# In[9]:


# Looking statistical analysis for independent variables...
Ddata = df.drop(['class identifier'], axis=1)
Ddata.describe()


# In[10]:


# Taking the count from class identifier columns...(Total wine classes)
df['class identifier'].value_counts()


# In[11]:


# Applying countplot froms seaborn...
sns.countplot(df['class identifier'])
plt.show()


# In[12]:


# Appling Histograms for all dataset...
df.hist(bins=10,figsize=(13, 10))
plt.show()


# In[13]:


# plt.scatter(df['Alcohol'],df['Malic acid'], c ="pink", linewidths = 2, marker ="s", edgecolor ="green", s = 100)
# plt.scatter(df['Alcohol'],df['Malic acid'], c ="yellow", linewidths = 2, marker ="^", edgecolor ="red", s = 10)
# plt.show()


# In[14]:


# Applying correlation method to check the relation between the attributes...
#corr = df[df.columns].corr()
#sns.heatmap(corr, cmap="YlGnBu", annot = True)


# In[15]:


# Applying correlation method to check the relation between the attributes...
plt.figure(figsize=(12,10))
cor = df[df.columns].corr()
sns.heatmap(cor,annot=True,cmap= plt.cm.CMRmap_r)
plt.show()


# In[16]:


# Drop class attribute from the datset to perpare new dataset for model training...
X= df.drop(['class identifier'], axis=1)


# In[17]:


# Looking X dataset header...
X.head()


# In[18]:


# Taking only "class identifier" attribute for training model...
Y = df.iloc[:,:1]

# Looking Y dataset header...
Y.head(2)


# In[19]:


# Split dataset into train and test...
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)


# In[20]:


# Checking shape of X dataset...
print(X_train.shape)
print(X_test.shape)


# In[21]:


# Checking shape of Y dataset...
print(Y_train.shape)
print(Y_test.shape)


# In[22]:


# Foom sklearn we are import all classifier fot testing...
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[23]:


# Creat a empty list and append all model classifier...
models = []
models.append(("Logistic Regression:",LogisticRegression()))
models.append(("Naive Bayes:",GaussianNB()))
models.append(("K-Nearest Neighbour:",KNeighborsClassifier(n_neighbors=3)))
models.append(("Decision Tree:",DecisionTreeClassifier()))
models.append(("Support Vector Machine-linear:",SVC(kernel="linear")))
models.append(("Support Vector Machine-rbf:",SVC(kernel="rbf")))
models.append(("Random Forest:",RandomForestClassifier(n_estimators=7)))
models.append(("eXtreme Gradient Boost:",XGBClassifier()))
models.append(("MLP:",MLPClassifier(hidden_layer_sizes=(45,30,15),solver='sgd',learning_rate_init=0.01,max_iter=1000)))
models.append(("AdaBoostClassifier:",AdaBoostClassifier()))
models.append(("GradientBoostingClassifier:",GradientBoostingClassifier()))


# In[24]:


# Models details...
print(models)


# In[25]:


# Running for loop into the modeles to fit our dataset to get the accuracy...
results = []
names = []
for name,model in models:
    kfold = KFold(n_splits=10, random_state=0)
    cv_result = cross_val_score(model,X_train,Y_train.values.ravel(), cv = kfold,scoring = "accuracy")
    names.append(name)
    results.append(cv_result)
for i in range(len(names)):
    print(names[i],results[i].mean()*100)


# In[26]:


# All classifier Result...

# Logistic Regression: 95.04761904761905
# Naive Bayes: 97.90476190476191
# K-Nearest Neighbour: 66.1904761904762
# Decision Tree: 93.66666666666667
# Support Vector Machine-linear: 96.42857142857142
# Support Vector Machine-rbf: 69.14285714285715
# Random Forest: 97.1904761904762
# eXtreme Gradient Boost: 96.52380952380952
# MLP: 32.38095238095238
# AdaBoostClassifier: 90.28571428571428
# GradientBoostingClassifier: 93.0


# # Random forest classifier implement...

# In[27]:


# Applying Random forest algorithm...
from sklearn.ensemble import RandomForestClassifier
rmodel=RandomForestClassifier(n_estimators=500)
rmodel.fit(X_train,Y_train)
Y_pred=rmodel.predict(X_test)


# In[28]:


# Checking accuracy...
from sklearn import metrics
print("Random forest Model Accuracy:",metrics.accuracy_score(Y_test, Y_pred))


# In[29]:


# Checking confusion matrix...
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, Y_pred)


# # Naive Bayes implement...

# In[30]:


# Applying Naive bayes algorithm...
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
n_pred = gnb.predict(X_test)


# In[31]:


# Checking accuracy...
from sklearn import metrics
print("Naive Bayes Model Accuracy:",metrics.accuracy_score(Y_test, n_pred))


# In[32]:


# Checking confusion matrix...
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, n_pred)


# In[ ]:




