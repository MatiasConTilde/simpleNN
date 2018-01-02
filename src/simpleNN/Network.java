package simpleNN;
import Jama.*;

public class Network {
  static double activate(double x) {
    return 1 / (1 + Math.pow(Math.E, -x)); // Sigmoid
    // return Math.max(0, x); // ReLU
  }

  static double derivative(double x) {
    return x * (1 - x); // Sigmoid
    // return x < 0 ? 0 : 1; // ReLU
  }

  public Matrix[] weights;
  double learningRate;

  public Network(int[] layers, double lr) {
    weights = new Matrix[layers.length - 1];

    for (int i = 0; i < weights.length; i++) {
      weights[i] = new Matrix(layers[i + 1], layers[i]);

      for (int row = 0; row < weights[i].getRowDimension(); row++) {
        for (int col = 0; col < weights[i].getColumnDimension(); col++) {
          weights[i].set(row, col, Math.random() * 2 - 1);
        }
      }
    }

    learningRate = lr;
  }

  public Matrix test(double[] input) {
    return test(new Matrix(input, input.length));
  }

  public Matrix test(Matrix input) {
    Matrix output = input.copy();
    for (int i = 0; i < weights.length; i++) {
      output = weights[i].times(output);

      for (int row = 0; row < output.getRowDimension(); row++) {
        for (int col = 0; col < output.getColumnDimension(); col++) {
          output.set(row, col, Network.activate(output.get(row, col)));
        }
      }
    }

    return output;
  }

  public void train(double[] input, double[] desired) {
    train(new Matrix(input, input.length), new Matrix(desired, desired.length));
  }

  public void train(Matrix input, Matrix desired) {
    Matrix[] outputs = new Matrix[weights.length];

    Matrix output = input.copy();
    for (int i = 0; i < weights.length; i++) {
      output = weights[i].times(output);

      for (int row = 0; row < output.getRowDimension(); row++) {
        for (int col = 0; col < output.getColumnDimension(); col++) {
          output.set(row, col, Network.activate(output.get(row, col)));
        }
      }

      outputs[i] = output;
    }

    Matrix error = desired.minus(outputs[outputs.length - 1]);

    for (int i = outputs.length - 1; i > 0; i--) {
      Matrix gradient = outputs[i].copy();
      for (int row = 0; row < gradient.getRowDimension(); row++) {
        for (int col = 0; col < gradient.getColumnDimension(); col++) {
          gradient.set(row, col, Network.derivative(gradient.get(row, col)));
        }
      }
      gradient = gradient.arrayTimes(error);
      gradient.timesEquals(learningRate);

      weights[i].plus(gradient.times(outputs[i - 1].transpose()));

      error = weights[i].transpose().times(error);
    }
  }
}
