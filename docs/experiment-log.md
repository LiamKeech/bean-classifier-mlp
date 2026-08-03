# Experiment log

All runs use seed 1541928603, the 70/15/15 split, and z-score input scaling
fitted on the training set only.

The reference configuration is: 32 hidden neurons, learning rate 0.10, decay
0.95 per epoch, fan-in-scaled initialisation, early stopping (patience 20, best weights
restored). Each experiment group varies one factor against this reference.
Per-epoch curves for every run are in `results/SSE_Log_<label>.csv`
(Epoch, TrainSSE, ValSSE, LearningRate), and the final numbers for every run
are in `results/metrics_summary.csv`.

## Learning rate (eta) values investigated

0.05, 0.10, 0.20 (with decay 0.95). Findings: all three reached the same
test accuracy (99.89166%); a larger learning rate reaches a lower training
SSE before the decay schedule shrinks it towards zero (which happens within
roughly 150 epochs regardless of the starting value)

## Technique 1: Dynamic learning rate (decay)

Decay per epoch: 1.00 (off), 0.99, 0.95, 0.50. Findings: multiplicative decay
per epoch is very aggressive over hundreds of epochs.
Decay 0.50 effectively stopped learning within 25 epochs.
Decay 0.95 plateaued at val SSE 15.98 by epoch ~150. 
Decay 0.99 did much better. 
Conclusion: on this well-conditioned, scaled dataset a constant learning rate 
of 0.10 needs no decay; if decay is used it must be very mild, or it starves
training before convergence. Decay implements an annealing schedule, and too aggressive
a schedule denies the network the many small steps it needs near the minimum.

## Technique 2: Appropriate weight initialisation

Uniform [-0.5, 0.5] vs the fan-in-scaled range,
Findings: the fan-in-scaled range did slightly better (val
SSE 15.98 vs 16.16, test accuracy 99.89166% vs 99.83749%), though the gap is
small at this network size. Both far outperform zero initialisation.

## Technique 3: Network architecture (hidden layer size)

16 vs 32 vs 64 hidden neurons. Findings: 32 gave the lowest test SSE (17.81)
and tied 16 for the best test accuracy (99.89166%); 64 did not improve on
either (test SSE 17.83, test accuracy 99.83749%) despite roughly double the
training cost. 32 hidden neurons is the best trade-off between capacity and
training cost.

## Technique 4: Prevention of overfitting (early stopping)

Early stopping (patience 20, restore best) vs training the full 500 epochs.
Findings: the dataset shows essentially no overfitting; training for the
full 500 epochs at the reference configuration reached the same test
accuracy (99.89166%) as stopping early. The practical benefit here is
training time: early stopping cut the best run to 302 epochs instead of 500
while guaranteeing the best-validation weights are kept. It remains the
safety net the textbook describes (Sec 7.2, p.86, Fig. 7.1) that would catch
overfitting if the network were larger or the data noisier.
