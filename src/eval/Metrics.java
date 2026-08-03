package eval;

import data.Dataset;
import nn.FFNN;

// Accuracy and SSE over a whole dataset.
public class Metrics {

    public static double accuracy(FFNN net, Dataset ds) {
        int correct = 0;
        for (int p = 0; p < ds.size(); p++) {
            int predicted = Dataset.argMax(net.fire(ds.X[p]));
            int actual = Dataset.argMax(ds.Y[p]);
            if (predicted == actual) correct++;
        }
        return (double) correct / ds.size();
    }

    public static double sse(FFNN net, Dataset ds) {
        double sse = 0;
        for (int p = 0; p < ds.size(); p++) {
            double[] out = net.fire(ds.X[p]);
            for (int k = 0; k < out.length; k++) {
                double err = ds.Y[p][k] - out[k];
                sse += err * err;
            }
        }
        return sse;
    }
}
