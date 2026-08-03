package train;

import data.Dataset;
import eval.Metrics;
import nn.FFNN;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

// Trains a network for up to maxEpochs, with learning rate decay and early stopping based on validation SSE.
public class Trainer {

    private final double startLearningRate;
    private final double decay;
    private final int maxEpochs;
    private final int patience;
    private final int printEvery;

    private final List<String[]> log = new ArrayList<>();
    private int epochsRun;

    public Trainer(double learningRate, double decay, int maxEpochs, int patience, int printEvery) {
        this.startLearningRate = learningRate;
        this.decay = decay;
        this.maxEpochs = maxEpochs;
        this.patience = patience;
        this.printEvery = printEvery;
    }

    public void train(FFNN net, Dataset train, Dataset validation, long seed) {
        double learningRate = startLearningRate;
        log.clear();
        log.add(new String[]{"Epoch", "TrainSSE", "ValSSE", "LearningRate"});

        double bestValSSE = Double.POSITIVE_INFINITY;
        double[] bestWeights = net.getWeights();
        int wait = 0;

        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            epochsRun = epoch;

            train.shuffle(seed + epoch); // different order each epoch
            for (int p = 0; p < train.size(); p++) {
                net.trainOn(train.X[p], train.Y[p], learningRate);
            }

            double trainSSE = Metrics.sse(net, train);
            double valSSE = Metrics.sse(net, validation);
            log.add(new String[]{String.valueOf(epoch), fmt(trainSSE), fmt(valSSE), fmt(learningRate)});

            if (epoch % printEvery == 0) {
                System.out.printf(Locale.ROOT, "epoch %5d | trainSSE %10.4f | valSSE %9.4f | learning rate %.5f%n", epoch, trainSSE, valSSE, learningRate);
            }

            if (valSSE < bestValSSE - 1e-8) {
                bestValSSE = valSSE;
                bestWeights = net.getWeights();
                wait = 0;
            } else if (++wait >= patience) {
                System.out.printf("early stop at epoch %d (no val improvement for %d epochs)%n", epoch, patience);
                break;
            }

            learningRate *= decay;
        }

        if (patience < maxEpochs) net.setWeights(bestWeights);
    }

    public List<String[]> getLog() {
        return log;
    }

    public int getEpochsRun() {
        return epochsRun;
    }

    private static String fmt(double v) {
        return String.format(Locale.ROOT, "%.6f", v);
    }
}
