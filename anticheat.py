import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sys import argv
from random import shuffle
import math
import matplotlib.pyplot as plt

class AntiAimBot:
  model = None
  trainX = None
  trainY = None
  testX = None
  testY = None
  X = None
  Y = None

  def __init__(self, input: str, output: str) -> None:
    self.X = read_csv(input).values
    self.Y = read_csv(output).values

  # Reshape data for (Sample, TimeSteps, Features)
  # Parameter n_steps defines the number of time steps
  # Sample: number of samples = len(sequences)/n_steps
  def split_sequences(self, n_steps: int) -> None:
    X = list()
    i = 0
    while i < len(self.X):
      # find the end of this pattern
      end_ix = i + n_steps
      # check if we are beyond the dataset
      if end_ix > len(self.X):
        break
      # gather input parts of the pattern
      seq_x = self.X[i:end_ix]
      X.append(seq_x)
      i += n_steps
    
    self.X = np.array(X)
  
  def shuffleData(self) -> None:
    c = list(zip(self.X, self.Y))
    shuffle(c)
    dfX, dfY = zip(*c)
    self.X = np.array(dfX)
    self.Y = np.array(dfY)

  # Parameter split_percent defines the ratio of training examples
  def get_train_test(self, split_percent: float =0.8) -> None:
    nX = len(self.X)
    nY = len(self.Y)
    # Point for splitting data into train and test
    splitX = int(nX*split_percent)
    splitY = int(nY*split_percent)

    self.trainX = self.X[range(splitX)]
    self.testX = self.X[splitX:]
    self.trainY = self.Y[range(splitY)]
    self.testY = self.Y[splitY:]

  def create_RNN(self, hidden_units, activation) -> None:
    # Define the model
    # Input shape = (timesteps, features)
    input_shape = (self.trainX.shape[1], self.trainX.shape[2])
    # Output shape = output
    dense_units = self.trainY.shape[1]
    self.model = Sequential()
    self.model.add(LSTM(hidden_units, input_shape=input_shape, activation=activation[0]))
    self.model.add(Dense(units=dense_units, activation=activation[1]))
    self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

  # Train the model
  def train(self, epochs: int, verbose: int, output: ModelCheckpoint) -> None:
    self.model.fit(self.trainX, self.trainY, epochs=epochs, verbose=verbose, callbacks=[output])

  # Load weights from a previous training
  def load_weights(self, weights: str) -> None:
    self.model.load_weights(weights)

  def print_error(self, train_predict, test_predict) -> None:    
    # Error of predictions
    train_rmse = math.sqrt(mean_squared_error(self.trainY, train_predict))
    test_rmse = math.sqrt(mean_squared_error(self.testY, test_predict))
    # Print RMSE
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    print('Test RMSE: %.3f RMSE' % (test_rmse))    

  # Plot the result
  def plot_result(self, train_predict, test_predict) -> None:
    actual = np.append(self.trainY, self.testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.axvline(x=len(self.trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Sunspots scaled')
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')

    plt.show()

def main(argv: list[str]):
  # Load data
  time_steps = 10

  anti = AntiAimBot(argv[0], argv[1])

  anti.split_sequences(time_steps)
  anti.shuffleData()
  anti.get_train_test(0.8)
  anti.create_RNN(hidden_units=200, activation=['tanh', 'tanh'])

  checkPoint = ModelCheckpoint(argv[2], save_weights_only=True, verbose=1)
  # Save the model
  anti.train(epochs=100, verbose=2, output=checkPoint)

  # make predictions
  train_predict = anti.model.predict(anti.trainX)
  test_predict = anti.model.predict(anti.testX)

  # Print error
  anti.print_error(train_predict, test_predict)

  #Plot result
  anti.plot_result(train_predict, test_predict)

if __name__ == "__main__":
  if len(argv) != 4:
    print("Usage: python3 example.py <input.csv> <output.csv> <wheights.hdf5>")
    exit(1)
  main(argv[1:])
