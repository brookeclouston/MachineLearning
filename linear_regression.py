""" Simple ML script demonstrating a Linear Regression Model predicting 
a student's final grade based on their previous term grades, study time, 
failures and absences. """
import pandas as pd 
import numpy as np 
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"

X = np.array(data.drop([predict], 1)) # all of the data without G3
Y = np.array(data[predict])
# creating the model 
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    # splits data into 10% test cohorts to train off of 

"""
best = 0
for _ in range(30):
    # train model 30 times and take the best accuracy
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train) # fits the data to find best fit line
    acc = linear.score(x_test, y_test) # returns value to describe model
    if acc > best:
        best = acc

        with open("studentmodel.pickle", "wb") as f:
            # saves instance of the model 
            pickle.dump(linear, f)

print("Accuracy: ", best)
"""

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    # prints the prediction associated with the actual data 
    print("prediction: %s, Data points: %s, Actual value: %s" % (predictions[x], x_test[x], y_test[x]))


p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()