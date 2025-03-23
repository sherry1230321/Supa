from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class LSTMModel:
    def __init__(self, input_shape, lstm_units=50, dropout_rate=0.2):
        self.model = Sequential()
        self.model.add(LSTM(lstm_units, input_shape=input_shape, return_sequences=True))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(lstm_units))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=20, batch_size=32, validation_data=None):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)