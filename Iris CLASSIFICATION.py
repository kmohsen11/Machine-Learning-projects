#!/usr/bin/env python
# coding: utf-8

# In[56]:


from sklearn.datasets import load_iris
iris_dataset = load_iris()
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np

#print("keys of iris_dataset:  \:n{}".format(iris_dataset.keys()))
#print(iris_dataset['DESCR'][:193]+"\n...")


# In[57]:


print("first five columns of data:\n{}".format(iris_dataset['data'][:5]))


# In[58]:


print(':\n{}'.format(iris_dataset['target']))


# In[59]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state = 0)


# In[60]:


print(X_train.shape)


# In[61]:


iris_dataframe = pd.DataFrame(X_train, columns = iris_dataset.feature_names)
#grr =  pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',     hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)


# In[62]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[63]:


knn.fit(X_train, y_train)


# In[64]:


X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))


# In[67]:


prediction = knn.predict(X_new)
print("prediction: {}".format(prediction))
print("predicted target name:{}".format(
    iris_dataset["target_names"][prediction]))


# In[68]:


y_pred = knn.predict(X_test)
print('test set predictions: \n {}'.format(y_pred))


# In[69]:


print("test set score: {:.2f}".format(np.mean(y_pred==y_test)))


# In[72]:


print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# In[1]:


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer
print("cancer.keys(): \n{}".format(cancer.keys()))


# In[ ]:




