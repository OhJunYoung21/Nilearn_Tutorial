from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.svm import SVC
import numpy as np


def linear(X,y):

    X = [i.flatten() for i in X]
    y = y

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

    model = LinearRegression()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_pred,y_test)
    return mse


def ADHD_SVC(X,y):

    X = np.array(X).reshape(10, -1)
    y = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    model = SVC()
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("model's accuracy:", accuracy)

    return True






