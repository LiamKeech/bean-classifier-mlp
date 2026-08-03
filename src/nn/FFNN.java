package nn;

import java.util.Random;

// Feedforward neural network, one hidden layer, sigmoid activations, trained with backpropagation + gradient descent.
public class FFNN {
    private final int numInputs;
    private final int numHidden;
    private final int numOutputs;

    private final double[][] hiddenWeights; // numInputs x numHidden
    private final double[] hiddenBiases;    // numHidden
    private final double[][] outputWeights; // numHidden x numOutputs
    private final double[] outputBiases;    // numOutputs

    public FFNN(int numInputs, int numHidden, int numOutputs) {
        this.numInputs = numInputs;
        this.numHidden = numHidden;
        this.numOutputs = numOutputs;
        hiddenWeights = new double[numInputs][numHidden];
        hiddenBiases = new double[numHidden];
        outputWeights = new double[numHidden][numOutputs];
        outputBiases = new double[numOutputs];
    }

    public void randomInit(long seed, double range) {
        Random r = new Random(seed);
        for (int i = 0; i < numInputs; i++)
            for (int j = 0; j < numHidden; j++)
                hiddenWeights[i][j] = (r.nextDouble() * 2 - 1) * range;
        for (int j = 0; j < numHidden; j++)
            hiddenBiases[j] = (r.nextDouble() * 2 - 1) * range;
        for (int i = 0; i < numHidden; i++)
            for (int j = 0; j < numOutputs; j++)
                outputWeights[i][j] = (r.nextDouble() * 2 - 1) * range;
        for (int j = 0; j < numOutputs; j++)
            outputBiases[j] = (r.nextDouble() * 2 - 1) * range;
    }

    public void fanInInit(long seed) {
        Random r = new Random(seed);
        double limitH = 1.0 / Math.sqrt(numInputs);
        for (int i = 0; i < numInputs; i++)
            for (int j = 0; j < numHidden; j++)
                hiddenWeights[i][j] = (r.nextDouble() * 2 - 1) * limitH;
        for (int j = 0; j < numHidden; j++)
            hiddenBiases[j] = (r.nextDouble() * 2 - 1) * limitH;

        double limitO = 1.0 / Math.sqrt(numHidden);
        for (int i = 0; i < numHidden; i++)
            for (int j = 0; j < numOutputs; j++)
                outputWeights[i][j] = (r.nextDouble() * 2 - 1) * limitO;
        for (int j = 0; j < numOutputs; j++)
            outputBiases[j] = (r.nextDouble() * 2 - 1) * limitO;
    }

    public double[] fire(double[] inputs) {
        return forward(inputs)[1];
    }

    // Forward pass, both layers. result[0] = hidden, result[1] = output.
    private double[][] forward(double[] inputs) {
        double[] hidden = new double[numHidden];
        for (int j = 0; j < numHidden; j++) {
            double sum = hiddenBiases[j];
            for (int i = 0; i < numInputs; i++) sum += inputs[i] * hiddenWeights[i][j];
            hidden[j] = sigmoid(sum);
        }
        double[] output = new double[numOutputs];
        for (int j = 0; j < numOutputs; j++) {
            double sum = outputBiases[j];
            for (int i = 0; i < numHidden; i++) sum += hidden[i] * outputWeights[i][j];
            output[j] = sigmoid(sum);
        }
        return new double[][]{hidden, output};
    }

    // Present one pattern, backpropagate, update every weight/bias scaled by the learning rate. Returns the pattern's SSE before the update.
    public double trainOn(double[] input, double[] target, double learningRate) {
        double[][] fwd = forward(input);
        double[] hidden = fwd[0];
        double[] output = fwd[1];

        double sse = 0;
        double[] deltaOut = new double[numOutputs];
        for (int k = 0; k < numOutputs; k++) {
            double err = target[k] - output[k];
            sse += err * err;
            deltaOut[k] = err * output[k] * (1 - output[k]);
        }

        double[] deltaHid = new double[numHidden];
        for (int j = 0; j < numHidden; j++) {
            double sum = 0;
            for (int k = 0; k < numOutputs; k++) sum += deltaOut[k] * outputWeights[j][k];
            deltaHid[j] = sum * hidden[j] * (1 - hidden[j]);
        }

        for (int j = 0; j < numHidden; j++)
            for (int k = 0; k < numOutputs; k++)
                outputWeights[j][k] += learningRate * deltaOut[k] * hidden[j];
        for (int k = 0; k < numOutputs; k++)
            outputBiases[k] += learningRate * deltaOut[k];

        for (int i = 0; i < numInputs; i++)
            for (int j = 0; j < numHidden; j++)
                hiddenWeights[i][j] += learningRate * deltaHid[j] * input[i];
        for (int j = 0; j < numHidden; j++)
            hiddenBiases[j] += learningRate * deltaHid[j];

        return sse;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // Flatten every weight and bias into one array, and read them back.
    public double[] getWeights() {
        double[] w = new double[(numInputs * numHidden) + numHidden + (numHidden * numOutputs) + numOutputs];
        int index = 0;
        for (int i = 0; i < numInputs; i++)
            for (int j = 0; j < numHidden; j++) w[index++] = hiddenWeights[i][j];
        for (int j = 0; j < numHidden; j++) w[index++] = hiddenBiases[j];
        for (int i = 0; i < numHidden; i++)
            for (int j = 0; j < numOutputs; j++) w[index++] = outputWeights[i][j];
        for (int j = 0; j < numOutputs; j++) w[index++] = outputBiases[j];
        return w;
    }

    public void setWeights(double[] w) {
        int index = 0;
        for (int i = 0; i < numInputs; i++)
            for (int j = 0; j < numHidden; j++) hiddenWeights[i][j] = w[index++];
        for (int j = 0; j < numHidden; j++) hiddenBiases[j] = w[index++];
        for (int i = 0; i < numHidden; i++)
            for (int j = 0; j < numOutputs; j++) outputWeights[i][j] = w[index++];
        for (int j = 0; j < numOutputs; j++) outputBiases[j] = w[index++];
    }
}
