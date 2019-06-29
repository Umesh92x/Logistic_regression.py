# TODO,logistic Regression, finding the people who bought SUV, YES or NOT
#importing modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#grab the data csv file
dataset=pd.read_csv('C://Users//vinod//PycharmProjects//Excercise//Udemy//Social_Network_Ads.csv')
X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

#split into test and train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=0)

#Apply Feature scaling for accuracy and increase speed.
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)
#x_test=np.dtype([x_test,np.float64])

# apply main classifier
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

#make prediction
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

# Visualising graph,plotting the people who bought SUV or NOT
# red for not and yello for YES
from matplotlib.colors import ListedColormap
X_set,y_set=x_test,y_test

X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,
                            stop=X_set[:,0].max()+1,
                            step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),
                                                X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','yellow')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate([0,1]):
   plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
               c=ListedColormap(('red','green'))(i),Label=j)
plt.title('Logistic Regression for Predicted value')
plt.xlabel('Age')
plt.ylable('Estimate Salary')
plt.legend()
plt.show()

# for training purpose
from matplotlib.colors import ListedColormap
X_set,y_set=x_train,y_train

X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,
                            stop=X_set[:,0].max()+1,
                            step=0.01),
                  np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),
                                                X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75,cmap=ListedColormap(('red','yellow')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

for i,j in enumerate([0,1]):
   plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
               c=ListedColormap(('red','green'))(i),Label=j)
plt.title('Logistic Regression for Predicted value')
plt.xlabel('Age')
plt.ylable('Estimate Salary')
plt.legend()
plt.show()
