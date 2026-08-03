package data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

// Reads BeanData.csv / Demo.csv into a Dataset.
// BeanData.csv: 16 numeric feature columns, then a "Class" column (string).
// Demo.csv: 16 numeric feature columns, no class column.
public class BeanDataLoader {

    public static final int NUM_FEATURES = 16;

    public static Dataset loadLabelled(String path) throws IOException {
        List<double[]> xs = new ArrayList<>();
        List<double[]> ys = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            br.readLine(); // skip header row
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isBlank()) continue;
                String[] t = line.split(",");

                double[] x = new double[NUM_FEATURES];
                for (int i = 0; i < NUM_FEATURES; i++) {
                    x[i] = Double.parseDouble(t[i].trim());
                }

                String className = t[NUM_FEATURES].trim();
                int ci = Dataset.classIndex(className);
                if (ci < 0) {
                    throw new IOException("Unknown class '" + className + "' in " + path);
                }

                xs.add(x);
                ys.add(Dataset.oneHot(ci));
            }
        }
        return new Dataset(xs.toArray(new double[0][]), ys.toArray(new double[0][]));
    }

    public static Dataset loadUnlabelled(String path) throws IOException {
        List<double[]> xs = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            br.readLine(); // skip header row
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isBlank()) continue;
                String[] t = line.split(",");

                double[] x = new double[NUM_FEATURES];
                for (int i = 0; i < NUM_FEATURES; i++) {
                    x[i] = Double.parseDouble(t[i].trim());
                }
                xs.add(x);
            }
        }
        return new Dataset(xs.toArray(new double[0][]), null);
    }
}
