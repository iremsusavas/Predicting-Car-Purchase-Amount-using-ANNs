#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importing the Data Set
data_set = pd.read_csv("Car_Purchasing_Data.csv", encoding = 'ISO-8859-1')
print(data_set.head(20)) 
print(data_set.describe())
sns.pairplot(data_set)

#Create Testing And Training Data Set / Data Cleaning
X = data_set.drop(columns = ['Country','Customer Name','Customer e-mail','Car Purchase Amount'])
y = data_set.iloc[:,-1]  
y = np.array(y).reshape(-1,1)

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
y = std_scaler.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 0)

#Model Training
import tensorflow.keras 
from keras.models import Sequential
from keras.layers import Dense,Activation

model = Sequential()
model.add(Dense(25,input_dim = 5,activation ='relu'))
model.add(Dense(25,activation = 'relu'))
model.add(Dense(1,activation = 'linear'))
model.summary()

model.compile(optimizer = 'adam', loss ='mean_squared_error' )

epochs = model.fit(X_train,y_train, epochs=25,verbose =1, batch_size =50, validation_split = 0.2)

#Model Evaluation
epochs.history.keys()
plt.plot(epochs.history['loss'])
plt.plot(epochs.history['val_loss'])
plt.title('Model Loss Progress')
plt.legend(['Training Loss','Validation Loss'])

#Gender, Age, Annual Salary, Credit Card Debt, Net Worth
X_test = np.array([[1,50,5000,1000,6000000]])
y_predicted = model.predict(X_test)

print('Expected Purchase Amount =>',y_predicted)