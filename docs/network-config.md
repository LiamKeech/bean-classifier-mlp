# Network configuration (for the submission template)

Values marked TBD are filled in from `results/metrics_summary.csv` after the
experiment sweep.

## Fixed configuration

| Item | Value |
|---|---|
| Network morphology | 16 inputs -> 1 hidden layer -> 7 outputs (one-hot) |
| Activation, hidden layer | Sigmoid |
| Activation, output layer | Sigmoid |
| Training algorithm | Online (per-pattern) backpropagation, gradient descent |
| Error measure | Sum Squared Error (SSE) |
| Training set size | 8618 (70%) |
| Validation set size | 1847 (15%) |
| Test set size | 1846 (15%) |
| Split method | Uniform random shuffle (seed 1541928603), then 70/15/15 split |
| Stopping conditions | Max 500 epochs, or early stopping: no validation SSE improvement for 20 consecutive epochs (best weights restored) |
| Initial weights (baseline) | Uniform in [-0.5, 0.5] |
| Initial weights (fan-in-scaled) | Uniform in [-1/sqrt(fanIn), 1/sqrt(fanIn)] per layer -- course guide Sec 1.3.2 / textbook Sec 7.3.2 |

## Best configuration found

Selected by lowest validation SSE across the sweep (run label `decay_1.00`
in `results/metrics_summary.csv`). This network produced the Demo.csv
predictions.

| Item | Value |
|---|---|
| Best learning rate (eta) | 0.10 (constant; decay 1.00) |
| Learning rate decay per epoch | none (1.00) |
| Hidden neurons | 32 |
| Weight initialisation | Fan-in-scaled, U(-1/sqrt(fanIn), 1/sqrt(fanIn)) |
| Typical iterations (epochs) | 302 (early stopped, patience 20) |
| SSE on training set | 7.7834 |
| SSE on test set | 6.0383 |
| % correct on training set | 100.00000 |
| % correct on test set | 99.94583 |
| Validation SSE | 2.2247 |
