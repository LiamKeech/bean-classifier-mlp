package data;

import java.util.Random;

// Holds a set of patterns: inputs X (n x 16) and one-hot targets Y (n x 7).
// Y is null for unlabelled data (Demo.csv).
public class Dataset {

    public static final String[] CLASSES = {"DERMASON", "SIRA", "SEKER", "BARBUNYA", "HOROZ", "CALI", "BOMBAY"};

    public final double[][] X; // n x numFeatures (16)
    public final double[][] Y; // n x numClasses (7), one-hot; null if unlabelled

    public Dataset(double[][] x, double[][] y) {
        this.X = x;
        this.Y = y;
    }

    public int size() {
        return X.length;
    }

    public static int classIndex(String name) {
        for (int i = 0; i < CLASSES.length; i++) {
            if (CLASSES[i].equalsIgnoreCase(name.trim())) return i;
        }
        return -1;
    }

    public static double[] oneHot(int classIndex) {
        double[] v = new double[CLASSES.length];
        v[classIndex] = 1.0;
        return v;
    }

    public static int argMax(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) best = i;
        }
        return best;
    }

    // Shuffle X and Y together
    public void shuffle(long seed) {
        Random r = new Random(seed);
        for (int i = X.length - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            double[] tx = X[i]; X[i] = X[j]; X[j] = tx;
            if (Y != null) {
                double[] ty = Y[i]; Y[i] = Y[j]; Y[j] = ty;
            }
        }
    }

    // Split into [train, validation, test] by fraction. Assumes already shuffled.
    // e.g. split(0.7, 0.15) -> 70% train, 15% val, 15% test.
    public Dataset[] split(double trainFrac, double valFrac) {
        int n = size();
        int nTrain = (int) Math.round(n * trainFrac);
        int nVal = (int) Math.round(n * valFrac);
        int nTest = n - nTrain - nVal;

        Dataset train = slice(0, nTrain);
        Dataset val = slice(nTrain, nTrain + nVal);
        Dataset test = slice(nTrain + nVal, nTrain + nVal + nTest);
        return new Dataset[]{train, val, test};
    }

    private Dataset slice(int from, int to) {
        int len = to - from;
        double[][] x = new double[len][];
        double[][] y = (Y != null) ? new double[len][] : null;
        for (int i = 0; i < len; i++) {
            x[i] = X[from + i];
            if (Y != null) y[i] = Y[from + i];
        }
        return new Dataset(x, y);
    }
}
