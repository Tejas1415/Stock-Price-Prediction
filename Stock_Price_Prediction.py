import pandas as pd                                          #import necessery modules
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import pickle                                                            # To save the trained classifier, saving time from not to train again and again

df = quandl.get("SSE/GGQ1", authtoken="epNLDM2RMctwx_QFrGKf")            # Get stock price data from Quandl
df['High_low change'] = (df['High'] - df['Last']) / df['Last'] *100      # Produce features from data that can be of more help
df['PCT_change'] = (df['Last']-df['Previous Day Price'])/df['Previous Day Price'] *100
df.fillna(value=33, inplace=True)                                        # Remove all None values (if any)
df = df[['Last', 'High_low change', 'PCT_change', 'Previous Day Price']]
print(len(df))
forecast_output = int(math.ceil(0.1*(len(df))))         # How many days further do you need to predict the price. Here its 36 days. use 0.01 for 4 days, use 0.001 for 1 day
df['label'] = df['Last'].shift(-forecast_output)        # Go those many days into the future

X_features = np.asarray(df.drop(['label'], 1))          # Except Label, use all others as features to calculate labels
X_NaNfeatures = X_features[-forecast_output:]           # The future dates with NaN values in xfeatures
df.dropna(inplace=True)
X_features = np.asarray(df.drop(['label'], 1))          # Except Label, use all others as features to calculate labels
X_features = preprocessing.scale(X_features)
y_label = np.asarray(df['label'])


X_featurestrain, X_featurestest, y_labeltrain,y_labeltest = cross_validation.train_test_split(X_features, y_label, test_size=0.2)
print(len(X_featurestrain),len(y_labeltest))

# Using direct function to predict the model and fit the features.
classifier1 = LinearRegression(n_jobs= -1)                # Thread it to process in as many processors as possible in your computer.
classifier1.fit(X_featurestrain, y_labeltrain)

# once the classifier is trained save it using pickle
# so that no need to train the classifier again and again
with open('Linearmodel1.pickle','wb') as temp_variable:
    pickle.dump(classifier1, temp_variable)
# Now to open the file again and again for further use, use the following commands
pickle_in= open('Linearmodel1.pickle','rb')
classifier1= pickle.load(pickle_in)

accuracy = classifier1.score(X_featurestest, y_labeltest) # accuracy from 10-40% found. (since it is 36 days, it's giving out bad prediction)
forecast_set = classifier1.predict(X_NaNfeatures)         # Predict all the NaN values that are produced for the next 36 days.
print(forecast_set, forecast_output)

print(accuracy*100)
print(len(X_NaNfeatures), len(X_features))
