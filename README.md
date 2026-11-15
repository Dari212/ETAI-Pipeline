# Baseline Predictive Pipeline -- ETAI

20260488 Darija Avramoska

## Beyond COMPAS

I applied the pipeline to the Adult census data (Option A).

### How to run

    python main.py                    # COMPAS (config.yaml), unchanged
    python main.py config_adult.yaml  # Adult

`main.py` takes the config path as an optional argument and defaults to
`config.yaml`. The Adult run also writes predictions for `adult_new.csv` to
`results/adult_dataset/predictions.csv` (id, probability, 0/1 prediction; 2,000 rows).

### What broke, why, and how I fixed it

| What broke | Why | Fix |
|---|---|---|
| `fairness_report` crashed | It used COMPAS's `score_text` column | Made the COMPAS comparison conditional on `score_text` being present |
| `?` placeholders left in `workclass`, `occupation`, `native_country` | Placeholder cleaning only ran for columns listed in `canonical_categories` | Listed these columns with empty mappings, so `?` becomes NaN without renaming labels |

### Results (roc_auc, nested 5-fold CV, same folds)

| Run | Model | Tuned | roc_auc | std |
|---|---|---|---|---|
| 20261115_015914 | hist_gradient_boosting | yes | 0.910 | 0.016 |
| 20261115_203054 | gradient_boosting | yes | 0.910 | 0.016 |
| 20261115_152944 | hist_gradient_boosting (no `race`) | yes | 0.909 | 0.016 |
| 20261115_165320 | adaboost | yes | 0.906 | 0.016 |
| 20261115_003835 | xgboost | no | 0.901 | 0.012 |
| 20261115_014127 | decision_tree | no | 0.725 | 0.024 |
| (dummy floor) | | | 0.500 | |

The chosen run (20261115_203054, gradient_boosting, tuned) reaches all three
target scores in nested 5-fold CV (honest estimate, mean ± std):
ROC-AUC 0.910 ± 0.016 (target ≥ 0.90), accuracy 0.862 ± 0.010 (target ≥ 0.855)
and F1 0.680 ± 0.031 (target ≥ 0.66, threshold 0.5). It also beats the no-model
baseline (0.50 / 0.76 / 0.00) and the default decision tree (0.72 / 0.80 / 0.58).
Corrected paired t-test vs the top run (20261115_015914): mean diff 0.0000,
p = 0.999, so the top runs are statistically tied.

After the feature selection was run on all the features only 8 features out of 37 were kept (age, education_num, capital_gain, capital_loss, hours_per_week, marital_status, occupation, relationship) excluding any V features. 
After running the corrected_paired_test on the model with and without `race`, the results show that their is no evidence of difference between them. Because it does not cost accuracy and avoids using a sensitive attribute directly, I excluded race from the model.

The hardest assumption to find was the missing values hidden as ? in the categorical columns, because the function that was actually doing that was the _canonicalize_categories that was also correcting the labels. What led me to the target scores was firstly the cleaning and preprocessing of the dataset, then testing different models with different config settings like adding feature selection, tuning, removing features and comparing the results and their performance. 

## New model XGBoost

I compared XGBoost without feature selection and with it. Selection kept 2 of 10 features, age and priors_count. Both runs have a nested ROC AUC of 0.725 with std of 0.012, and the corrected paired test on the five outer-fold scores gives a mean difference of 0.000, 0/5 wins, t = 0 and p = 1.0, so the smaller model is as good. I would keep selection on because a two-feature model scores the same as a ten-feature one and is cheaper and easier to explain.

The result on the scoreboard, show hist gradient boost as the winner with 0.728 roc_auc. The winner is not far from the other models with a gap of 0.01 between the first and 4th model. All the models showed improved scores after tuning. In terms of FPR, the models didn't show much improvement (although lower than the COMPAS scores) and still have large and imbalanced rate between different races.   
  
Several new models were compared using nested 5-fold cross-validation on accuracy metric. AdaBoost scored best, with validation 0.672 and train 0.682 (gap 0.010). The oblique random forest was a close second, with validation 0.669 and train 0.684 (gap 0.015), so slightly more overfitting. Both beat the tuned decision tree (validation 0.660), but the 0.003 difference between them is well within fold-to-fold variation (std about 0.02), so they are considered comparable. AdaBoost was selected because of the slightly higher score and smaller gap. However, the model is biased toward predicting class 0. It finds most non-reoffenders (recall 0.77) but misses nearly half of reoffenders (recall 0.55). Class 0 is also the larger class, which pulls the model that way.

Added new models and preprocessing: Logistic regressions gave the same results. This time I tested random forest and it showed overfitting of 0.080.

Updates after data cleaning: Logistic Regression is still the bette model, but with bigger overfitting on the train set. However, there is less overfitting with Decision Trees comparing to the previous results.   

Current best model: Logistic Regression with test accuracy of 0.68 and test accuracy of 0.679. This model is better then Decision Tree, because there is no overfitting on the train dataset. With this we can say that Logistic Regression is better model to classify new unseen data. 

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
