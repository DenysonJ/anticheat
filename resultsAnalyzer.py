import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def parse():
  parser = argparse.ArgumentParser(description='Model Analyzer')
  parser.add_argument('--version', action='version', version='%(prog)s 0.1')

  parser.add_argument('input', type=str, help='Path to the file with the inputs to be analyzed')
  parser.add_argument('--history', type=str, help='Path to the file with the history of the training')
  parser.add_argument('--trigger', type=float, default=0.95, help='Trigger to be used in the predictions')

  return parser

def analyzer(df: pd.DataFrame, trigger: float):
  actual = df['Y_real'].values
  predictions = df['Y_probability'].values
  rows = len(actual)
  Y_trigger = np.array([1 if predictions[i] > trigger else 0 for i in range(rows)])
  Y_filtered = Y_trigger[Y_trigger == actual]
  Y_false_positive = Y_trigger[(Y_trigger == 1) & (actual == 0)]
  Y_false_negative = Y_trigger[(Y_trigger == 0) & (actual == 1)]

  print("Acertos: " + str(len(Y_filtered)))
  print("Erros: " + str(rows - len(Y_filtered)))
  print("Porcentagem de acertos: " + str(len(Y_filtered) / rows * 100) + "%")
  print("Falso positivo: " + str(len(Y_false_positive) / rows * 100) + "%")
  print("Número de falsos positivos: " + str(len(Y_false_positive)))
  print("Falso negativo: " + str(len(Y_false_negative) / rows * 100) + "%")
  print("Número de falsos negativos: " + str(len(Y_false_negative)))
  print("Número de cheaters: " + str(len(Y_trigger[Y_trigger == 1])))


def plot(nparray: np.ndarray, legend: str = 'Precisão', title: str = 'Precisão ao longo das épocas'):
  print(max(nparray))
  rows = len(nparray)
  plt.figure(figsize=(15, 6), dpi=80)
  plt.plot(range(rows), nparray)
  plt.legend([legend])
  plt.title(title)
  plt.show()

def main(args):
  df = pd.read_csv(args.input)
  analyzer(df, args.trigger)
  if args.history:
    history = pd.read_csv(args.history)
    plot(history['accuracy'].values)
    plot(history['loss'].values, legend='Loss', title='Erro ao longo das épocas')
  

if __name__ == "__main__":
  parser = parse()
  args = parser.parse_args()
  main(args)