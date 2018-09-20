import pandas as pd                                          #import necessery modules
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get("SSE/GGQ1", authtoken="epNLDM2RMctwx_QFrGKf")   #Get stock price data from Quandl
df['High_low change'] = (df['High'] - df['Last']) / df['Last'] *100    #Produce features from data that can be of more help
df['PCT_change'] = (df['Last']-df['Previous Day Price'])/df['Previous Day Price'] *100
df.fillna(value=33, inplace=True)                                        # Remove all None values (if any)
df = df[['Last', 'High_low change', 'PCT_change', 'Previous Day Price']]
print(len(df))
forecast_output = int(math.ceil(0.001*(len(df))))      # How many days further do you need to predict the price. Here its 1 day
df['label'] = df['Last'].shift(-forecast_output)      # Go those many days into the future
df.dropna(inplace=True)                               # When u shift, the last few values are left with NaN, remove those rows

X_features = np.asarray(df.drop(['label'], 1))        # Except Label, use all others as features to calculate labels
y_label = np.asarray(df['label'])
X_features = preprocessing.scale(X_features)          # Scale them for better results, but scaling takes time forlarge datasets

#Split the data into test and train to validate the accuracy of the predicted model
X_featurestrain, X_featurestest, y_labeltrain,y_labeltest = cross_validation.train_test_split(X_features, y_label, test_size=0.2)
print(len(X_featurestrain),len(y_labeltest))

# Using direct function to predict the model and fit the features.
classifier1 = LinearRegression()
classifier1.fit(X_featurestrain, y_labeltrain)
accuracy = classifier1.score(X_featurestest, y_labeltest) # accuracy from 75-95% found.
print(accuracy*100)
#print(df.head())
#print(df.tail(), forecast_output)

#using SVM SVR regression model
classifier1 = svm.SVR()
classifier1.fit(X_featurestrain, y_labeltrain)
accuracy = classifier1.score(X_featurestest, y_labeltest) # accuracy from 60-80% found.
print(accuracy*100)

#using SVM SVR regression model with polynomial kernel
classifier1 = svm.SVR(kernel='poly')
classifier1.fit(X_featurestrain, y_labeltrain)
accuracy = classifier1.score(X_featurestest, y_labeltest) # accuracy from 20-60% found.
print(accuracy*100)



