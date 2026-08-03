# Network configuration 

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
| Initial weights (fan-in-scaled) | Uniform in [-1/sqrt(fanIn), 1/sqrt(fanIn)] per layer |

## Best configuration found

Selected by lowest validation SSE

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
