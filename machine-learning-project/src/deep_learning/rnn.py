from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.optimizers import Adam

class RNNModel:
    def __init__(self, input_shape, units=50, output_units=1, learning_rate=0.001):
        self.model = Sequential()
        self.model.add(SimpleRNN(units, input_shape=input_shape, activation='tanh'))
        self.model.add(Dense(output_units, activation='sigmoid'))
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=20, batch_size=32, validation_data=None):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)