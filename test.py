# Used to import since sklearn showld be the same version. i could not able to create in Jupyter notebook

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
#import statsmodels.api as sm
#import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set()
import pickle
data =pd.read_csv('Admission_Prediction.csv')


data['University Rating'] = data['University Rating'].fillna(data['University Rating'].mode()[0])
data['TOEFL Score'] = data['TOEFL Score'].fillna(data['TOEFL Score'].mean())
data['GRE Score']  = data['GRE Score'].fillna(data['GRE Score'].mean())

data= data.drop(columns = ['Serial No.'])

y = data['Chance of Admit']
X =data.drop(columns = ['Chance of Admit'])


scaler =StandardScaler()

X_scaled = scaler.fit_transform(X)

x_train,x_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.25,random_state=355)


# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression = regression.fit(x_train,y_train)

# saving the model to the local file system
filename = 'finalized_model.pickle'
pickle.dump(regression, open(filename, 'wb'))

filename = 'scaler_model.pickle'
pickle.dump(scaler, open(filename, 'wb'))

reg_score= regression.score(x_train,y_train)

print ("score:", reg_score)






