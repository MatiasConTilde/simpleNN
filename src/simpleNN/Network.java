package simpleNN;

public class Network {
  private class Layer {
    private class Neuron {
      Neuron[] inputs;
      float[] weights;
      float output;
      float error;

      Neuron() {
        error = 0;
      }

      Neuron(Neuron[] pInputs) {
        inputs = new Neuron[pInputs.length];
        weights = new float[pInputs.length];

        for (int i = 0; i < pInputs.length; i++) {
          inputs[i] = pInputs[i];
          weights[i] = (float) Math.random() * 2 - 1;
        }
      }

      void feedFd() {
        float sum = 0;
        for (int i = 0; i < inputs.length; i++) {
          sum += inputs[i].getOutput() * weights[i];
        }
        output = sigmoid(sum);
        error = 0;
      }

      void train() {
        float diff = (1f - output) * (1f + output) * error * (float) 0.01;
        for (int i = 0; i < inputs.length; i++) {
          inputs[i].error += weights[i] * error;
          weights[i] += inputs[i].output * diff;
        }
      }

      void setError(float desired) {
        error = desired - output;
      }

      float getOutput() {
        return output;
      }

      void setOutput(float input) {
        output = input;
      }
    }

    Neuron[] neurons;

    Layer(int size) {
      neurons = new Neuron[size];
      for (int i = 0; i < size; i++) {
        neurons[i] = new Neuron();
      }
    }

    Layer(int size, Layer previous) {
      neurons = new Neuron[size];
      for (int i = 0; i < size; i++) {
        neurons[i] = new Neuron(previous.getNeurons());
      }
    }

    void train() {
      for (Neuron n : neurons) n.train();
    }

    void feedFd() {
      for (Neuron n : neurons) n.feedFd();
    }

    void setErrors(float[] desired) {
      for (int i = 0; i < desired.length; i++) {
        neurons[i].setError(desired[i]);
      }
    }

    void setOutputs(float[] inputs) {
      for (int i = 0; i < inputs.length; i++) {
        neurons[i].setOutput(inputs[i]);
      }
    }

    float[] getOutputs() {
      float[] out = new float[neurons.length];
      for (int i = 0; i < out.length; i++) {
        out[i] = neurons[i].getOutput();
      }
      return out;
    }

    Neuron[] getNeurons() {
      return neurons;
    }
  }

  Layer[] layers;

  public Network(int[] layerSizes) {
    layers = new Layer[layerSizes.length];

    layers[0] = new Layer(layerSizes[0]);
    for (int i = 1; i < layers.length; i++) {
      layers[i] = new Layer(layerSizes[i], layers[i-1]);
    }
  }

  public void train(float[] input, float[] desired) {
    test(input);

    layers[layers.length-1].setErrors(desired);
    for (int i = layers.length-1; i >= 1; i--) {
      layers[i].train();
    }
  }

  public void test(float[] inputs) {
    layers[0].setOutputs(inputs);

    for (int i = 1; i < layers.length; i++) {
      layers[i].feedFd();
    }
  }

  public float[] output() {
    return layers[layers.length-1].getOutputs();
  }

  private float sigmoid(float x) {
    return 2f / (1f + (float) Math.exp(-2f * x)) - 1f;
  }
}
