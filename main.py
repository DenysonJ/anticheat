import argparse
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from anticheat import AntiAimBot
from keras.utils import plot_model

def parse():
  parser = argparse.ArgumentParser(description='AntiCheat')
  parser.add_argument('--version', action='version', version='%(prog)s 0.1')

  parser.add_argument('input', type=str, help='Path to the file with inputs to be splited to train and test or to be predicted')
  parser.add_argument('output', type=str, help='Path to the file with outputs to be splited to train and test or to be predicted')

  parser.add_argument('--load', type=str, help='Path to the file with the model to be loaded')
  parser.add_argument('--save', type=str, default='model.keras', help='Path to the file with the model to be saved')

  parser.add_argument('-e', '--epochs', type=int, default=150, help='Number of epochs to train the model')
  parser.add_argument('--verbose', type=int, default=2, help='Verbosity of the training process')
  parser.add_argument('-s', '--steps', type=int, default=10, help='Number of steps to be used in the RNN')
  parser.add_argument('--hidden', type=int, default=100, help='Number of hidden units to be used in the RNN')
  parser.add_argument('--split_percent', type=float, default=0.8, help='Percentage of the data to be used in the training')
  parser.add_argument('--activation', type=str, default='tanh', help='Activation function to be used in the RNN')
  parser.add_argument('-o', '--optimizer', type=str, default='adam', help='Optimizer to be used in the RNN')
  parser.add_argument('--loss', type=str, default='mean_squared_error', help='Loss function to be used in the RNN')
  parser.add_argument('-m', '--metrics', type=str, default='accuracy', help='Metrics to be used in the RNN')

  parser.add_argument('--load_weights', type=str, help='Path to the file with the weights to be loaded')
  parser.add_argument('--save_weights', type=str, default='weights.hdf5', help='Path to the file with the weights to be saved')

  parser.add_argument('--plot', action='store_true', help='Plot the result of the training')
  parser.add_argument('--print_error', action='store_true', help='Print the error of the training')

  parser.add_argument('--topology', action='store_true', help='Print the topology of the RNN')

  parser.add_argument('--save_predict', type=str, default='predict.csv', help='Path to the file with the predictions to be saved')
  parser.add_argument('--trigger', type=float, default=0.55, help='Trigger to be used in the predictions')

  parser.add_argument('--not_shuffle', action='store_false', help='Not shuffle the data before splitting into train and test')

  return parser

def main():
  parser = parse()
  args = parser.parse_args()

  # Create the AntiCheat object
  ac = AntiAimBot(args.input, args.output)
  ac.split_sequences(args.steps)

  # Split the data into train and test
  if args.not_shuffle:
    ac.shuffleData()
  ac.get_train_test(args.split_percent)

  if args.load is None:
    # Create the RNN
    ac.create_RNN(args.hidden, [args.activation, args.activation], args.loss, args.optimizer, args.metrics)
    ac.model.save(args.save)
    ac.model.summary()

  # Load the model
  if args.load:
    ac.load_model(args.load)
    ac.model.summary()
  
  if args.topology:
    plot_model(ac.model, to_file='model.png', show_shapes=True)

  # Train the RNN
  if args.load_weights is None:
    if args.split_percent == 0:
      print("You can't train the model without splitting the data into train and test")
      exit(1)
    h = ac.train(args.epochs, args.verbose, args.save_weights)
    pd.DataFrame(h.history).to_csv('history.csv', index=False)

  # Load the weights
  if args.load_weights:
    ac.load_weights(args.load_weights)

  test_predict = ac.model.predict(ac.testX)
  test_predict = test_predict.flatten()

  # Print the error
  if args.print_error:
    ac.print_error(test_predict)

  if args.plot:
    ac.plot_result(test_predict)

  # Save the predictions
  predicted = [ 1 if test_predict[i] > args.trigger else 0 for i in range(len(test_predict))  ]
  predict = { 'Y': predicted, 'Y_real': ac.testY.flatten() , 'Y_probability': test_predict.flatten()}

  pd.DataFrame(predict).to_csv(args.save_predict, index=False)


if __name__ == "__main__":
  main()