""" Simple ML script demonstrating an SVM Model predicting 
a malignant and benign tomours using sklearns built in data sets. """
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()


X = cancer.data
Y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

classes = ["malignant", "benign"]


clf.fit(x_train, y_train)
predicted = clf.predict(x_test)


for x in range(len(predicted)):
    print("predicted: ", classes[predicted[x]], "actual: ", classes[y_test[x]])
  