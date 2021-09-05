# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 00:29:32 2021

@author: STANLEY NNAMANI - TEAM LEAD OF IMPACT HUB
"""
#import necessary data for data manipulation
import pandas as pd
import numpy as np


#import data set for several regions to be studied by model
data1= pd.read_csv("Datasets\\AEP_hourly.csv")
data2= pd.read_csv("Datasets\\COMED_hourly.csv")
data3= pd.read_csv("Datasets\\DAYTON_hourly.csv")
data4= pd.read_csv("Datasets\\DEOK_hourly.csv")
data5= pd.read_csv("Datasets\\DOM_hourly.csv")
data6= pd.read_csv("Datasets\\DUQ_hourly.csv")
data7= pd.read_csv("Datasets\\EKPC_hourly.csv")
data8= pd.read_csv("Datasets\\FE_hourly.csv")
data9= pd.read_csv("Datasets\\NI_hourly.csv")
data10= pd.read_csv("Datasets\\PJME_hourly.csv")
data11= pd.read_csv("Datasets\\PJMW_hourly.csv")




#Lets work first on the data from the region with dataset AEP


#import all needed libraries
import pandas as pd         
import numpy as np
import datetime as dates
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as ploter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib


#convert data1 datetime column to datetime objects
data1["Datetime"]=pd.to_datetime(data1["Datetime"])


#Split the datetime in to year, month, day and time
data1["Year"]=pd.DatetimeIndex(data1["Datetime"]).year
data1["Month"]=pd.DatetimeIndex(data1["Datetime"]).month
data1["Week"]=pd.DatetimeIndex(data1["Datetime"]).week
data1["Day"]=pd.DatetimeIndex(data1["Datetime"]).day
data1["Time"]=pd.DatetimeIndex(data1["Datetime"]).hour
data1["Date"]= pd.DatetimeIndex(data1["Datetime"]).date
data1["Day of the week"]=data1["Datetime"].dt.dayofweek     #this converts the day to a number representing the day of the week, 0 represent monday,1 tuesday, 6 sunday and so on..

#check for weekends and add this feature to the dataframe

data1["Weekends"]= data1["Datetime"].apply(lambda is_weekend: 1 if is_weekend.dayofweek >= 5 else 0 )       #this gives a value of 1 for weekends and 0 for weekdays
data1["Period of the Day"]=data1["Time"].apply(lambda period: 1 if period<13 else(2 if period<16 else 3))    #this gives a value of 1 for morning, 2 for noon and 3 for evening

data1=data1.set_index("Datetime")


#Data analysis and Visualization
max_year=data1["Year"].max()
min_year=data1["Year"].min()




#Create a list containing all the years in the interval
years=[x for x in range(min_year,max_year+1)]

#make a plot showing the trend of energy usage for each year

fig, ax =ploter.subplots(figsize=(50,50))


for index in range(len(years)):
    ploter.subplot(len(years),1, index+1)
    year= years[index]
    power_consumed=data1[str(year)]
    ploter.plot(power_consumed["AEP_MW"])
    ploter.title(str(year), y=0, loc="left")
ploter.show()
fig.tight_layout()



#make an histogram chart showing the trend of energy usage for each year

fig, ax =ploter.subplots(figsize=(50,50))


for index in range(len(years)):
    ploter.subplot(len(years),1, index+1)
    year= years[index]
    power_consumed=data1[str(year)]
    power_consumed["AEP_MW"].hist(bins=200)
    ploter.title(str(year), y=0, loc="left")
ploter.show()
fig.tight_layout()



#make a plot showing the trend of energy usage for each month in each of the years

months=[x for x in range(1,13)]  #list containing all the months



for index in years:
    fig, ax =ploter.subplots(figsize=(50,50))
    for i in range(len(months)):
        ploter.subplot(len(months), 1, i+1)
        month = str(index)+"-"+str(months[i])
        power_consumed= data1[month]
        ploter.plot(power_consumed["AEP_MW"])
        ploter.title(month, y=0, loc="right")
    ploter.show()
    fig.tight_layout()



#make a histogram chart showing the trend of energy usage for each month in each of the years

months=[x for x in range(1,13)]  #list containing all the months



for index in years:
    fig, ax =ploter.subplots(figsize=(50,50))
    for i in range(len(months)):
        ploter.subplot(len(months), 1, i+1)
        month = str(index)+"-"+str(months[i])
        power_consumed= data1[month]
        power_consumed["AEP_MW"].hist(bins=300)
        ploter.title(month, y=0, loc="right")
    ploter.show()
    fig.tight_layout()


#get information about data
data1.describe()
data1.info()


features=data1[["Year","Month","Day","Time","Day of the week","Weekends","Period of the Day"]]

label=data1["AEP_MW"]

x_train, x_test, y_train, y_test = train_test_split(features,label, shuffle=False)   #split train and test dataset

#Convert data to numpy array
x_train=np.array(features)
y_train=np.array(label)

#Create model using LSTM for Multi Step prediction

#Data preparation for LSTM multi prediction
data_train,label_train = [], []


for energy in range(24, len(y_train)-24):
    data_train.append(y_train[energy-24:energy])
    label_train.append(y_train[energy:energy+24])
    
    
data_train, label_train =np.array(data_train), np.array(data_train)
data_train.shape,label_train.shape


#Carryout normalization using min max scaler

scaler =MinMaxScaler()
data_train = scaler.fit_transform(data_train)
label_train = scaler.fit_transform(label_train)

#reshape data to 3D for LSTM
data_train = data_train.reshape(121225, 24, 1)


#Build Model

model = Sequential([
    LSTM (units=200, activation="relu", input_shape=(24,1)),
    Dense (units=24, ),
    ])

model.compile(optimizer = 'adam', loss="mean_squared_error")

model.fit(data_train, label_train, epochs=10 batch_size=24)


from sklearn.externals import joblib
filename= "C:\\Users\\USER\\Desktop\\AI model program\\model.pkl"
with open(filename, "wb") as file:
    joblib.dump(model, file)
    
#The model was trained to be predict the hourly power consumption for a day given recent power consumptions

#Now is time to test the model using the test data

#Prepare test data
data_test,label_test = [], []


for energy in range(24, len(y_test)-24):
    data_test.append(y_test[energy-24:energy])
    label_test.append(y_test[energy:energy+24])
    
data_test, label_test =np.array(data_test), np.array(data_test)
data_test.shape,label_test.shape

#Carryout normalization using min max scaler on data_test only

data_test = scaler.fit_transform(data_test)

#reshape data to 3D for testing
data_test = data_test.reshape(data_test.shape[0], data_test.shape[1] , 1)\

#predict the future hourly consumptions using data_test
predictions=model.predict(data_test)

#inverse transform predictions
predictions=scaler.inverse_transform(predictions)


#Evaluate the model
#To evaluate the model, we will use RMSE since we are dealing with power consumption


#Evaluate one or more daily forcasts against expected values
def Evaluate(label_test, predictions):
    scores=[]
    
    #calculate RSME score for each hour
    for i in range(label_test.shape[1]):
        #calculate RMSE
        mse= mean_squared_error(label_test[:,i],predictions[:,i])
        rmse=np.sqrt(mse)
        
        #store values
        scores.append(rmse)
    #Calculate overall RSME
    total_score= 0
    for row in range(label_test.shape[0]):
        for col in range(label_test.shape[1]):
            total_score = total_score + (label_test[row,col] - predictions[row,col])**2
        score=np.sqrt(total_score / (label_test.shape[0]*predictions.shape[1]))
        return score, scores
    
Evaluate(label_test,predictions)


#To check that the model is performing well, we will get the standard deviation
np.std(label_test[0])

#We can now save our model
filename= "model.pkl"
with open(filename, "wb") as file:
    joblib.dump(model, file)
    
#Since RMSE is less that standard deviation, the model is a good model
#We can also train similar models like this one for the other datasets imported 
#With the techniques we can create models hosted in cloud to predict power consumptions of different regions

