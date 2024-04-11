import csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd 
import time

data=pd.read_csv('DataForTrainingAndTesting.csv')
features=['Year','GDPperCapita','OilPrice','Population','inflationRate','covidCases','unemploymentRate']
y=data.TotalCars

X=data[features]

trainX,valX,trainy,valy=train_test_split(X,y,random_state=0)

model=RandomForestRegressor(random_state=1)
model.fit(trainX,trainy)
predictions=model.predict(valX)
mape = 100 * (abs(valy - predictions) / valy).mean()

print("The Mean Absolute Percentage Error in model's predictions upon testing:", mape,'%')