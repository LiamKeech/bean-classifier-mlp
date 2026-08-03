package app;

import data.BeanDataLoader;
import data.Dataset;
import data.Scaler;
import eval.Metrics;
import nn.FFNN;
import train.Trainer;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

// Trains the best configuration found during the experiments and predicts the class of each bean in Demo.csv.

public class Main {

    static final long SEED = 1541928603;
    static final int HIDDEN = 32;
    static final double LEARNING_RATE = 0.1;
    static final double DECAY = 1.0;      // learning rate decay per epoch
    static final int MAX_EPOCHS = 500;
    static final int PATIENCE = 20;       // early stopping: epochs without val improvement
    static final int PRINT_EVERY = 25;

    public static void main(String[] args) throws IOException {
        // Load, shuffle, and split 70/15/15 into train / validation / test.
        Dataset all = BeanDataLoader.loadLabelled("data/BeanData.csv");
        all.shuffle(SEED);
        Dataset[] parts = all.split(0.70, 0.15);
        Dataset train = parts[0], val = parts[1], test = parts[2];
        System.out.printf("Loaded %d patterns: train %d | val %d | test %d%n", all.size(), train.size(), val.size(), test.size());

        // Scale features
        Scaler scaler = new Scaler();
        scaler.fit(train.X);
        scaler.transform(train);
        scaler.transform(val);
        scaler.transform(test);

        // Build, initialise, and train the network (16 -> HIDDEN -> 7).
        FFNN net = new FFNN(BeanDataLoader.NUM_FEATURES, HIDDEN, Dataset.CLASSES.length);
        net.fanInInit(SEED);
        Trainer trainer = new Trainer(LEARNING_RATE, DECAY, MAX_EPOCHS, PATIENCE, PRINT_EVERY);
        trainer.train(net, train, val, SEED);

        new File("results").mkdirs();
        writeCSV("results/SSE_Log_best.csv", trainer.getLog());

        System.out.printf(Locale.ROOT, "%nTrain: SSE %.4f | %.5f%% correct%n", Metrics.sse(net, train), 100 * Metrics.accuracy(net, train));
        System.out.printf(Locale.ROOT, "Test:  SSE %.4f | %.5f%% correct%n", Metrics.sse(net, test), 100 * Metrics.accuracy(net, test));

        // Predict Demo.csv using the same scaler as the training data.
        Dataset demo = BeanDataLoader.loadUnlabelled("data/Demo.csv");
        scaler.transform(demo);
        List<String[]> predictions = new ArrayList<>();
        predictions.add(new String[]{"Bean", "PredictedClass"});
        for (int i = 0; i < demo.size(); i++) {
            int cls = Dataset.argMax(net.fire(demo.X[i]));
            predictions.add(new String[]{String.valueOf(i + 1), Dataset.CLASSES[cls]});
            System.out.printf("Bean %2d: %s%n", i + 1, Dataset.CLASSES[cls]);
        }
        writeCSV("results/demo_predictions.csv", predictions);
    }

    static void writeCSV(String path, List<String[]> rows) throws IOException {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
            for (String[] row : rows) {
                bw.write(String.join(",", row));
                bw.newLine();
            }
        }
    }
}
