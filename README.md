# Dry Bean Classifier - Feedforward Neural Network

A single-hidden-layer feedforward neural network, trained by backpropagation and
gradient descent, implemented entirely from scratch in Java with no ML libraries.
It classifies dry bean varieties from 16 geometric features extracted by computer
vision, and reaches **99.946% accuracy on the held-out test set**.

The repository is not just the network: it is an ablation study over five training
factors, every run sharing one fixed seed and one fixed data split so the
comparisons are honest.

## Headline result

| | |
|---|---|
| **Test accuracy** | **99.946%** (1845 of 1846 held-out patterns) |
| Training accuracy | 100.000% |
| Test SSE | 6.0383 |
| Validation SSE | 2.2247 |
| Epochs to converge | 302 (early stopped, patience 20) |
| Configuration | 32 hidden neurons, η 0.10 constant, fan-in-scaled init |

The reference configuration used as the control for the ablation (the same network
with decay 0.95) reaches 99.892%. Removing the decay is what buys the last
0.054 percentage points.

## Ablation study

Every run below varies **one** factor against the reference configuration
(32 hidden, η 0.10, decay 0.95, fan-in-scaled init, early stopping with patience
20). Everything else, including the seed and the split, is held fixed.

| Factor | Setting | Epochs | Val SSE | Test SSE | Test accuracy |
|---|---|---:|---:|---:|---:|
| **Learning rate** | η 0.05 | 440 | 25.6319 | 24.8815 | 99.892% |
| | η 0.10 *(reference)* | 417 | 15.9774 | 17.8119 | 99.892% |
| | η 0.20 | 422 | 9.6224 | 12.5854 | 99.892% |
| **Decay per epoch** | 1.00 (off) | 302 | 2.2247 | 6.0383 | **99.946%** |
| | 0.99 | 500 | 5.0686 | 8.7468 | 99.946% |
| | 0.95 *(reference)* | 417 | 15.9774 | 17.8119 | 99.892% |
| | 0.50 | 54 | 81.3461 | 72.6975 | 99.621% |
| **Hidden units** | 16 | 419 | 17.5452 | 18.0534 | 99.892% |
| | 32 *(reference)* | 417 | 15.9774 | 17.8119 | 99.892% |
| | 64 | 431 | 16.5187 | 17.8342 | 99.837% |
| **Initialisation** | fan-in-scaled *(reference)* | 417 | 15.9774 | 17.8119 | 99.892% |
| | uniform [-0.5, 0.5] | 418 | 16.1610 | 17.4823 | 99.837% |
| **Early stopping** | on, patience 20 *(reference)* | 417 | 15.9774 | 17.8119 | 99.892% |
| | off, full 500 epochs | 500 | 15.9774 | 17.8119 | 99.892% |
| **Baseline** | 10 hidden, η 0.10, no decay, uniform init | 500 | 4.9369 | 5.7186 | 99.837% |

### What the sweep shows

- **Learning rate barely matters here.** All three values land on the same test
  accuracy. A larger η reaches a lower training SSE before the decay schedule
  shrinks the step size towards zero, which happens within roughly 150 epochs
  whatever the starting value.
- **Decay is the factor that actually moves the result, and it moves it
  downwards.** Multiplicative per-epoch decay compounds hard over hundreds of
  epochs. Decay 0.50 starved training inside 25 epochs and stopped at 99.621%.
  On this well-conditioned, scaled dataset a constant η 0.10 needs no annealing
  at all: too aggressive a schedule denies the network the many small steps it
  needs near the minimum.
- **Fan-in-scaled initialisation edges out uniform**, though the gap is small at
  this network size. Both are far ahead of zero initialisation.
- **32 hidden units is the sweet spot.** 64 units did not improve on 32 despite
  roughly double the training cost, and 16 matched 32 on accuracy while sitting
  slightly worse on SSE.
- **Early stopping cost nothing and saved time.** With the reference
  configuration it produced numerically identical final metrics to training the
  full 500 epochs, because the decayed learning rate has effectively frozen the
  weights long before epoch 417. On the best configuration it cut the run from
  500 epochs to 302. This dataset shows essentially no overfitting, so early
  stopping is a safety net here rather than a fix.

Longer discussion in [`docs/experiment-log.md`](docs/experiment-log.md); the
fixed network configuration is in [`docs/network-config.md`](docs/network-config.md).

## Methodology

Every number above comes from runs sharing this setup, so any difference between
two rows is attributable to the factor being varied and nothing else.

| Item | Value |
|---|---|
| Random seed | `1541928603`, fixed for the shuffle and for weight initialisation |
| Split | 70/15/15 train/validation/test, uniform random shuffle before splitting |
| Split sizes | 8618 train, 1847 validation, 1846 test (12,311 patterns total) |
| Input scaling | z-score, **fitted on the training set only**, then applied to all three splits |
| Network | 16 inputs → 1 hidden layer → 7 outputs (one-hot) |
| Activation | Sigmoid on both the hidden and the output layer |
| Training | Online (per-pattern) backpropagation, gradient descent |
| Error measure | Sum Squared Error |
| Stopping | 500 epochs max, or no validation-SSE improvement for 20 consecutive epochs with best weights restored |

Fitting the scaler on the training split alone matters: fitting it on the full
dataset would leak test-set statistics into training and inflate the result.

## Interactive results viewer

Open [`viewer/index.html`](viewer/index.html) by double-clicking it. It is a
static, dependency-free page (no server, no build step, no network access) that
plots the per-epoch SSE curves for each ablation group and compares the final
test accuracy across every run.

## Building and running

There is no Maven or Gradle here on purpose. A JDK is the only dependency.
**JDK 11 or newer**; the code has been checked against JDK 25.

Run these from the repository root, because the program resolves `data/` and
`results/` relative to the working directory.

```bash
# compile every package into out/
javac -d out src/app/*.java src/data/*.java src/eval/*.java src/nn/*.java src/train/*.java

# train the best configuration and classify the demo beans
java -cp out app.Main
```

That trains the 32-hidden, η 0.10, no-decay network, prints the train and test
SSE and accuracy, classifies the 50 unlabelled beans in `data/Demo.csv`, and
writes:

- `results/SSE_Log_best.csv` - the per-epoch curve for this run
- `results/demo_predictions.csv` - the predicted class for each demo bean

Training takes on the order of a minute on a typical laptop.

### Regenerating the results bundle

The viewer reads a single bundled JSON file built from the CSVs in `results/`.
Regenerate it after any new run:

```bash
java tools/BundleResults.java
```

This is a single-file source-launch program, so it needs no compile step of its
own. It reads `results/metrics_summary.csv` plus each `results/SSE_Log_*.csv`
and writes:

- `viewer/results-bundle.json` - the bundle: run metadata (learning rate, decay,
  hidden units, initialisation, early-stopping flag, epochs run, final SSE and
  accuracy) plus the per-epoch `trainSSE`, `valSSE` and `learningRate` curves as
  parallel arrays
- `viewer/results-bundle.js` - the same JSON assigned to `window.RESULTS_BUNDLE`

Both files are written because browsers block `fetch()` on `file://` URLs. The
viewer loads the `.js` wrapper with a `<script>` tag so double-clicking the page
works; the `.json` is the portable artefact for charting anywhere else.

`SSE_Log_best.csv` is skipped by the bundler by design: it is the curve produced
by rerunning `app.Main`, and it duplicates the `decay_1.00` sweep entry that is
already in the bundle.

## Implementation

Written from scratch, no ML libraries, no third-party dependencies at all.

| Path | Contents |
|---|---|
| `src/nn/FFNN.java` | The network: layer sizes, sigmoid activations, forward pass, weight initialisation (uniform and fan-in-scaled) |
| `src/train/Trainer.java` | Backpropagation loop, learning-rate decay, early stopping with best-weight restore, per-epoch SSE logging |
| `src/data/BeanDataLoader.java` | CSV parsing for the labelled and unlabelled sets |
| `src/data/Dataset.java` | Pattern storage, one-hot encoding, shuffling, 70/15/15 splitting |
| `src/data/Scaler.java` | z-score scaling, fitted on training data only |
| `src/eval/Metrics.java` | SSE and classification accuracy |
| `src/app/Main.java` | Trains the best configuration and predicts the demo set |
| `tools/BundleResults.java` | Turns `results/*.csv` into the viewer bundle |
| `viewer/index.html` | The static results viewer |

## Dataset

The network takes 16 numeric features describing a bean's shape and classifies it
into one of 7 registered dry bean types:

`DERMASON`, `SIRA`, `SEKER`, `BARBUNYA`, `HOROZ`, `CALI`, `BOMBAY`

Over 12,000 bean grain images were segmented and reduced to 16 numeric features
(area, perimeter, axis lengths, eccentricity, shape factors and so on).

- `data/BeanData.csv` - labelled data, split into train/validation/test
- `data/Demo.csv` - 50 unlabelled beans that the trained network predicts

Data from Koklu, M. and Ozkan, I.A. (2020), *"Multiclass Classification of Dry
Beans Using Computer Vision and Machine Learning Techniques."* Computers and
Electronics in Agriculture, 174, 105507.
[DOI: 10.1016/j.compag.2020.105507](https://doi.org/10.1016/j.compag.2020.105507)
