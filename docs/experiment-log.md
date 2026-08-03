# Experiment log

All runs use seed 1541928603, the 70/15/15 split, and z-score input scaling
fitted on the training set only (scaling itself is one of the investigated
techniques: without it the sigmoid neurons saturate on raw feature values
like Area of about 27000 and the network cannot train at all).

Every run is fully independent and reproducible: the sweep (`tools/Sweep.java`)
reloads and re-splits the data before each configuration, so any single run
can be reproduced on its own -- `src/app/Main.java` reproduces the best run
(`decay_1.00`) exactly.

The reference configuration is: 32 hidden neurons, learning rate 0.10, decay
0.95 per epoch, fan-in-scaled initialisation (course guide/textbook, see
`docs/technique-citations.md`), early stopping (patience 20, best weights
restored). Each experiment group varies one factor against this reference.
Per-epoch curves for every run are in `results/SSE_Log_<label>.csv`
(Epoch, TrainSSE, ValSSE, LearningRate), and the final numbers for every run
are in `results/metrics_summary.csv`.

## Learning rate (eta) values investigated

0.05, 0.10, 0.20 (with decay 0.95). Findings: all three reached the same
test accuracy (99.89166%); a larger learning rate reaches a lower training
SSE before the decay schedule shrinks it towards zero (which happens within
roughly 150 epochs regardless of the starting value) -- test SSE was 12.59
at 0.20, 17.81 at 0.10, 24.88 at 0.05. Without decay (see below), a learning
rate of 0.10 was stable and sufficient; no divergence or oscillation was
observed at 0.20.

## Technique 1: Dynamic learning rate (decay)

Decay per epoch: 1.00 (off), 0.99, 0.95, 0.50. Findings: multiplicative decay
per epoch is very aggressive over hundreds of epochs (0.95^150 is about
0.0005). Decay 0.50 effectively stopped learning within 25 epochs (train SSE
stuck at 355). Decay 0.95 plateaued at val SSE 15.98 by epoch ~150. Decay
0.99 did much better (val SSE 5.07, still improving at the 500-epoch cap).
No decay was best of all (val SSE 2.22 in 302 epochs). Conclusion: on this
well-conditioned, scaled dataset a constant learning rate of 0.10 needs no
decay; if decay is used it must be very mild (0.99 or gentler) or it starves
training before convergence. This matches the textbook's description (Sec
7.3.3, p.98-100): decay implements an annealing schedule, and too aggressive
a schedule denies the network the many small steps it needs near the
minimum.

## Technique 2: Appropriate weight initialisation

Uniform [-0.5, 0.5] vs the course guide's fan-in-scaled range,
U(-1/sqrt(fanIn), 1/sqrt(fanIn)) (course guide Sec 1.3.2 p.11; textbook Sec
7.3.2 p.97-98). Findings: the fan-in-scaled range did slightly better (val
SSE 15.98 vs 16.16, test accuracy 99.89166% vs 99.83749%), though the gap is
small at this network size. With only 16 inputs and 32 hidden neurons the
fan-in is small enough that a fixed [-0.5, 0.5] range does not badly
saturate the sigmoids either, so the fan-in-scaled range's theoretical
advantage (avoiding saturation as network size grows) does not show up
strongly here -- it would matter more on a much wider or deeper network.
Both far outperform zero initialisation, which the textbook notes cannot
break symmetry between hidden units at all (p.97).

## Technique 3: Network architecture (hidden layer size)

16 vs 32 vs 64 hidden neurons. Findings: 32 gave the lowest test SSE (17.81)
and tied 16 for the best test accuracy (99.89166%); 64 did not improve on
either (test SSE 17.83, test accuracy 99.83749%) despite roughly double the
training cost. 32 hidden neurons is the best trade-off between capacity and
training cost, following the textbook's principle that among networks that
fit equally well the smaller one is preferred (Sec 7.3.5, p.101-102).

## Technique 4: Prevention of overfitting (early stopping)

Early stopping (patience 20, restore best) vs training the full 500 epochs.
Findings: the dataset shows essentially no overfitting; training for the
full 500 epochs at the reference configuration reached the same test
accuracy (99.89166%) as stopping early. The practical benefit here is
training time: early stopping cut the best run to 302 epochs instead of 500
while guaranteeing the best-validation weights are kept. It remains the
safety net the textbook describes (Sec 7.2, p.86, Fig. 7.1) that would catch
overfitting if the network were larger or the data noisier.

## Baseline

Phase-1 configuration for comparison: 10 hidden neurons, learning rate 0.10,
no decay, uniform init, no early stopping, 500 epochs. Findings: reached a
low test SSE (5.72) by simply training at a constant learning rate for all
500 epochs, but with slightly lower test accuracy (99.83749%) than the best
config and no protection against overfitting. Confirms the main story: a
constant learning rate of 0.10 plus input scaling does most of the work on
this dataset; the techniques buy speed (early stopping), a small robustness
edge (fan-in-scaled init), and a principled way to pick the final model
(validation set).
