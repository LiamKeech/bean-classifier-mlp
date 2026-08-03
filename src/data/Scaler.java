package data;

// Standardises each feature to mean 0, std 1: x' = (x - mean) / std.
// Without this the sigmoid neurons saturate on raw values like Area (~27000)
// and the network never learns. fit() must only see training data; the same
// mean/std is then applied to validation, test, and Demo.

public class Scaler {

    private double[] mean;
    private double[] std;

    public void fit(double[][] X) {
        int n = X.length;
        int d = X[0].length;
        mean = new double[d];
        std = new double[d];

        for (double[] row : X) {
            for (int j = 0; j < d; j++) mean[j] += row[j];
        }
        for (int j = 0; j < d; j++) mean[j] /= n;

        for (double[] row : X) {
            for (int j = 0; j < d; j++) {
                double diff = row[j] - mean[j];
                std[j] += diff * diff;
            }
        }
        for (int j = 0; j < d; j++) {
            std[j] = Math.sqrt(std[j] / n);
            if (std[j] == 0) std[j] = 1e-8; // guard against divide-by-zero
        }
    }

    public void transform(Dataset ds) {
        for (double[] row : ds.X) {
            for (int j = 0; j < row.length; j++) {
                row[j] = (row[j] - mean[j]) / std[j];
            }
        }
    }
}
