from numpy.core.multiarray import ndarray
from sklearn.svm import SVR
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

from ranker import Ranker


class SupportVectorRegression(Ranker):

    def __init__(self, epsilon=0.2, kernel='linear', C=1):
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        super().__init__()

    def fit(self, X_train, y_train):
        self.model.fit(X=X_train, y=y_train)

    def predict(self,X) -> ndarray:
        return self.model.predict(X=X)

class PointwiseNN(Ranker):
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(128))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        super().__init__()

    def fit(self, X_train,y_train):
        self.model.fit(x=X_train,y=y_train)

    def predict(self,X) -> ndarray:
        return self.model.predict(x=X)
