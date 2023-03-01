#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[1]:


import pandas as pd
df = pd.read_csv('traffic.csv')
df


# In[2]:


df.head()


# In[3]:


df.tail()


# In[4]:


df['Datetime']


# In[5]:


df1 = df.reset_index()['Traffic']
df1.shape


# In[6]:


df1


# In[7]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[8]:


import numpy as np
df1


# In[9]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[10]:


df1


# In[11]:


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[12]:


training_size,test_size


# In[13]:


train_data,test_data


# In[14]:


import numpy

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]  
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)


# In[15]:


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


# In[16]:


X_train.shape , y_train.shape


# In[17]:


X_test.shape, y_test.shape


# In[18]:


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[19]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[20]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[21]:


model.summary()


# In[23]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)


# In[24]:


import tensorflow as tf


# In[25]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[26]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[27]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[29]:


math.sqrt(mean_squared_error(y_test,test_predict))


# In[30]:


look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[31]:


len(test_data)


# In[32]:


x_input = test_data[341:].reshape(1,-1)
x_input.shape


# In[33]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[34]:


temp_input


# In[36]:


from numpy import array

lst_output=[]
n_steps=42104
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)


# In[37]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[38]:


import matplotlib.pyplot as plt


# In[39]:


len(df1)


# In[50]:


plt.plot(day_pred,scaler.inverse_transform(lst_output))


# In[53]:


df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[121273:])


# In[54]:


df3=scaler.inverse_transform(df3).tolist()


# In[55]:


plt.plot(df3)


# In[ ]:




