# Predicting Final Student Grades (EDA + Deep Learning Regression)

**Student:** RITHI R · **College ID:** 24AD091 · **Department:** Artificial Intelligence and Data Science, KPR Institute of Engineering and Technology
**Course:** U21ADP05 – Exploratory Data Analysis and Visualization · **Assignment II – Case Study** · AY 2026–2027, Semester V

## What this project does
Explores the UCI *Student Performance* dataset (Portuguese-language course: 649 students, 32 input features, target = final grade G3, 0–20),
visualises it following Ben Fry's seven stages (Acquire, Parse, Filter, Mine, Represent, Refine, Interact), analyses it statistically
(univariate, bivariate, multivariate, probability distributions) and trains a multilayer perceptron (MLP) regressor in two settings:
**A** with the earlier grades G1 and G2, and **B** without them (early-warning setting).

## Dataset
* Source: <https://archive.ics.uci.edu/dataset/320/student+performance> · DOI 10.24432/C5TG7T · licence CC BY 4.0
* File used: `data/student-por.csv` (649 rows × 33 columns, 0 missing values, 0 duplicates). The file here is comma-separated; the original UCI file uses `;`.
* Citation: Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
  and Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. FUBUTEC 2008, 5–12.

## Project structure
```
data/                 dataset
src/                  config, data_preprocessing, eda, visualizations, train_model, evaluate_model, diagrams, run_all
outputs/eda/          EDA figures (1-12), tables, interactive HTML pages
outputs/model_evaluation/  evaluation figures (13-20), metrics
models/               trained MLPs (.keras) and fitted preprocessors (.joblib)
report/               Word + PDF report and build scripts
```

## How to run
```bash
pip install -r requirements.txt
python src/run_all.py        # EDA -> figures -> tuning/training -> evaluation (a few minutes on CPU)
```

## Results (held-out test set of 98 students, seed 42, stratified 70/15/15 split)
| Setting | MLP MAE | MLP RMSE | MLP R² | Ridge RMSE | Random forest RMSE | Mean-only RMSE |
|---|---|---|---|---|---|---|
| A – with G1, G2 | 0.91 | 1.33 | 0.82 | 1.35 | 1.12 | 3.13 |
| B – without G1, G2 | 2.03 | 2.93 | 0.12 | 2.75 | 2.73 | 3.13 |

Over 10 random splits the MLP test RMSE was 1.40 ± 0.19 (A) and 2.90 ± 0.28 (B).
The MLP is competitive with, but not better than, ridge regression and a random forest. Without G1 and G2, all models explain only a small share of the variance.

## Key insights
* G2 and G3 correlate at r = 0.92; the earlier grades dominate the prediction.
* Past failures, wish for higher education and school are the strongest non-grade associations with G3.
* 15 students have G3 = 0 (all with 0 absences); they cause the largest prediction errors.
* Absences are over-dispersed (not Poisson); G3 without the zero grades is approximately normal.

## Limitations
Small data (649 rows); single course and region; associations are not causal; hyperparameters were chosen on one validation split.

See `report/` for the full report.
