# Citations for the techniques implemented

Page/section numbers refer to the course guide (`WRCV402CourseGuide.pdf`) and
the prescribed textbook, Engelbrecht, *Computational Intelligence: An
Introduction* (2nd ed.). Use these directly in the "Variations Investigated"
section of the report.

IMPORTANT: the report may cite ONLY the course guide and the textbook. Author
names that appear below (Wessels & Barnard, Thodberg) are citations printed
INSIDE the textbook's own text; they are quoted here so the supporting
passages can be found again, but must NOT appear as sources in the report --
cite the course guide section / textbook section instead.

## Input scaling (Scaler.java)

- **Course guide, Section 1.3.2 "Performance Factors", "Normalization of
  input values"** (p.10-11): "The goal is for net to be in the the active
  range of the sigmoid function, [-sqrt(3), sqrt(3)], at the commencement of
  training."
- **Textbook, Section 7.3.1 "Data Preparation" -> "Scaling and Normalization"**
  (p.92-96): explains that unscaled inputs saturate the sigmoid, killing
  weight updates near the asymptotes, and derives **z-score normalization**
  (Eq. 7.15) as the standard scheme "very useful when there are outliers in
  the data" and when the input range is unknown — exactly `x' = (x-mean)/std`,
  what `Scaler.fit`/`transform` implement.

## Weight initialisation (FFNN.randomInit / fanInInit)

- **Course guide, Section 1.3.2, "Weight Initialisation"** (p.11): "Randomly
  initialize weights U(-1/sqrt(fanIn), 1/sqrt(fanIn))."
- **Textbook, Section 7.3.2 "Weight Initialization"** (p.97-98): "A sensible
  weight initialization strategy is to choose small random weights centered
  around 0... Wessels and Barnard showed that random weights in the range
  [-1/sqrt(fanIn), 1/sqrt(fanIn)] is a good choice" [Wessels and Barnard
  1992]. It also explains *why* this matters: large initial weights saturate
  units (zero derivative, no learning), while all-zero weights leave hidden
  units unable to break symmetry (identical gradients forever).
- `fanInInit` implements this formula exactly, per layer (limit = 1/sqrt(16)
  for the hidden layer, 1/sqrt(32) for the output layer, at hidden size 32).
  `randomInit`'s fixed [-0.5, 0.5] range, which does not scale with layer
  size, is the naive baseline it's compared against.

## Dynamic learning rate (Trainer learning-rate decay)

- **Course guide, Section 1.3.2, "Learning Rate"** (p.11): "Dynamic learning
  rate: eta(t) = eta(0) * e^(-t/tau), where tau is a constant."
- **Textbook, Section 7.3.3 "Learning Rate and Momentum"** (p.98-100):
  explains the speed/stability trade-off (too small = slow, too large =
  oscillation or overshoot into a bad local minimum) and recommends "an
  annealing schedule to gradually reduce a large learning rate to a smaller
  value... allows for large initial steps, and ensures small steps in the
  region of the minimum."
- `Trainer`'s per-epoch multiplication of the learning rate by `decay` is the discrete form of the same
  exponential schedule: eta(t) = eta(0) * decay^t = eta(0) * e^(t ln(decay)),
  i.e. the course guide's formula with tau = -1/ln(decay).

## Prevention of overfitting (Trainer early stopping)

- **Course guide, Section 1.2.2, "Stopping conditions"** (p.9, Eq. 1.27) and
  **Section 1.3.2, "Detection of overfitting"** (p.10): "When overfitting of
  the validation set is observed: Ev > mean(Ev) + std(Ev)."
- **Textbook, Section 7.2 "Analysis of Performance"** (p.86): "Estimations of
  generalization error during training can be used to detect the point of
  overfitting... there is a point at which the training error continues to
  decrease, while the generalization error starts to increase. This is the
  point of overfitting. Training should stop as soon an increase in
  generalization error is observed" (Figure 7.1).
- `Trainer`'s rule (stop after `patience` epochs with no validation SSE
  improvement, then restore the best-seen weights) is the standard
  *patience-based* variant of this same validation-monitoring idea — simpler
  to tune than the mean+std threshold. Cite the course guide's formal
  criterion (Eq. 1.27) and the textbook's Section 7.2, and note that
  patience-based stopping is the practical implementation of the same
  "stop when validation error stops improving" principle.

## Network architecture (hidden layer size sweep)

- **Course guide, Section 1.2.3, "Functioning of Hidden Neurons"** (p.10):
  "The number of hidden neurons in a FFNN must normally be manually tuned,
  because knowledge about the solution space is generally not available...
  For classification problems a hidden neuron is required for each boundary
  in the solution space."
- **Textbook, Section 7.3.5 "Architecture Selection"** (p.101-102): invokes
  Ockham's razor — "if several networks fit the training set equally well,
  then the simplest network... will on average give the best generalization
  performance" [Thodberg 1991] — and notes an exhaustive search over
  architectures is intractable, so "a simple method is to train a few
  networks of different architecture and choose the one which results in the
  lowest generalization error," i.e. exactly the trial-and-error hidden-size
  sweep (16/32/64) done here.

## Summary table for the report

| Technique | Course guide | Textbook (Engelbrecht) |
|---|---|---|
| Input scaling | Sec 1.3.2, "Normalization of input values", p.10-11 | Sec 7.3.1, p.92-96, Eq. 7.15 (z-score) |
| Weight initialisation | Sec 1.3.2, "Weight Initialisation", p.11 | Sec 7.3.2, p.97-98 [Wessels & Barnard 1992] |
| Dynamic learning rate | Sec 1.3.2, "Learning Rate", p.11 | Sec 7.3.3, p.98-100 (annealing schedule) |
| Early stopping / overfitting | Sec 1.2.2 Eq. 1.27, Sec 1.3.2 p.10 | Sec 7.2, p.86, Fig. 7.1 |
| Architecture selection | Sec 1.2.3, p.10 | Sec 7.3.5, p.101-102 [Thodberg 1991] |
