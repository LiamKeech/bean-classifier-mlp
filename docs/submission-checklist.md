# Assignment 1 Submission Checklist

Every item below comes straight from `Submission Template.docx`. Check things
off as you produce them, and note where in the repo each value/artefact
should live so you can copy it into the template at the end.

## Header / Declaration

- [ ] Date
- [ ] Name
- [ ] Student Number
- [ ] Signature (declaration that this is your own, original work)

## Introduction

- [ ] Brief description of your implementation
      *(source: write fresh, informed by `README.md`)*
- [ ] Paragraph on **network morphology** — what inputs are fed in, what is
      output *(16 input features → 7 one-hot output classes; describe hidden
      layer size/role)*

### Required values (put these in `docs/network-config.md`)

- [ ] Stopping conditions (e.g. max epochs, error threshold, early stopping rule)
- [ ] Initial weight ranges
- [ ] Training set size
- [ ] Test set size
- [ ] How the test set was selected (random split? stratified by class? seed used?)
- [ ] Activation function(s) used (hidden layer, output layer)
- [ ] Number of hidden neurons

## Variations Investigated

- [ ] Brief description of the different algorithm settings you tried and
      what you discovered *(source: `docs/experiment-log.md`)*
- [ ] Values for **eta (learning rate)** investigated — list every value tried
- [ ] Additional techniques investigated — must be **at least 3** of:
  - [ ] Prevention of overfitting
  - [ ] Noise injection
  - [ ] Appropriate weight initialisation
  - [ ] Dynamic learning rate
  - [ ] Scaling of input values
  - [ ] Network architecture (e.g. hidden layer size)

  For each technique used, note in `docs/experiment-log.md`: what you changed,
  why, and whether it helped.

## Results

- [ ] Graph of **fitness (error) vs time (epochs)** for the variations tested
      *(source: `results/fitness_vs_time_*.png`)*
- [ ] Table: Technique vs % Correct on Training Set vs % Correct on Test Set
      *(source: `results/metrics_summary.csv`)*

### Best technique found — report these numbers

- [ ] Number of iterations (typically)
- [ ] Best eta value
- [ ] Sum Squared Error (SSE) on Training Set
- [ ] SSE on Test Set
- [ ] % correctly classified on Training Set
- [ ] % correctly classified on Test Set

  *(all sourced from `docs/network-config.md` and `results/metrics_summary.csv`,
  for the run that performed best)*

- [ ] **Results on `Demo.csv`** — predicted bean class for all 50 patterns,
      numbered 1–50 to match the template's results table
      *(source: `results/demo_predictions.csv`)*

## Code

- [ ] Full source code appended at the end of the document
      *(source: everything in `src/`)*

## Formatting reminders

- [ ] Keep the written report (everything before the code appendix) to
      **~3 pages**
- [ ] Code is appended after that, not counted against the 3 pages

## Repo files this checklist depends on

| File | Purpose |
|---|---|
| `README.md` | High-level project description, overview |
| `docs/network-config.md` | All fixed config values (stopping condition, weight init, split sizes, activation functions, hidden neurons, best learning rate (eta), best SSE, best % correct) |
| `docs/experiment-log.md` | Log of every learning rate (eta) value and technique tried, with findings |
| `docs/technique-citations.md` | Course guide / textbook section and page numbers backing each technique, for citing in "Variations Investigated" |
| `results/fitness_vs_time_*.png` | Fitness/error vs epoch graphs per variation |
| `results/metrics_summary.csv` | Technique vs train/test accuracy table |
| `results/demo_predictions.csv` | Predicted class for each of the 50 `Demo.csv` rows |
| `src/` | Final code to append to the submission |

Once every box above is checked and every file in the table exists and is
filled in, you have everything you need to complete
`Submission Template.docx` without hunting back through code or notebooks.
