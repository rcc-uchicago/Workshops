import sklearn.datasets
import os
import pandas as pd
import numpy as np

iris_path = os.path.join(sklearn.datasets.__path__[0], 'data', 'iris.csv')
iris_df=pd.read_csv(iris_path)
#give proper names to different columns
iris_df.columns=['s_l','s_w','p_l','p_w','target']
#print a few lines from DataFrame
iris_df.head()

#crate training set as features and target classes
#each as numpy array
X=np.array(iris_df.drop('target',axis=1))
y=np.array(iris_df.target)
X.shape,y.shape
