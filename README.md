Applied Project Audio – Emotion Prediction Experiments
This project investigates supervised machine learning models for predicting emotional responses from audio data, focusing on Lasso Regression and XGBoost. We analyze a dataset of 1920 features and 6423 samples, targeting five emotions: Arousal, Valence, Interest, Despair, and Joy. The work is split into two phases: model evaluation and data augmentation experiments. For a detailed analysis, see the project report (not included). The .ipynb files contain the Python code.

Developed by Zohar LasKar Koriat, Achituv Drot, and Yoel Graumann.

🔬 Phase 1: Model Evaluation and Comparison

We trained and compared Lasso Regression (linear, L1-regularized) and XGBoost (non-linear, tree-based ensemble) against a baseline Stacked LSTM model, using normalized Root Mean Squared Error (RMSE) as the metric. Experiments leveraged two aggregations:

Itamar’s Aggregation: 5 percentiles (0.1, 0.25, 0.5, 0.75, 0.9).
Our Aggregation: 13 statistics (percentiles 0.1–0.9 in 0.1 steps, mean, SD, min, max).
Experiment 1: Lasso vs. XGBoost on Itamar’s dataset.

Experiment 2: Model performance on our richer aggregation.

Experiment 3: Error analysis across emotion distributions (symmetric to highly skewed).

We varied feature sets and evaluated performance across emotion types, from symmetric (Arousal, Valence) to skewed (Joy).

🧪 Phase 2: MixUp Data Augmentation

To address data imbalance (many zero-emotion samples), we applied the MixUp technique, generating 3,000 synthetic samples per emotion. We introduced a weighted variant:

Standard MixUp: 
𝜆
∼
Beta
(
2
,
1
)
λ∼Beta(2,1), uniform pairing.
Weighted MixUp: Exponential weights 
𝑒
𝑦
𝑖
⋅
weight
∑
𝑒
𝑦
𝑗
⋅
weight
∑e 
y 
j
​
 ⋅weight
 
e 
y 
i
​
 ⋅weight
 
​
 , prioritizing high-emotion samples (weights: 0, 1, 2, 3, 4, 6, 10).
Tested on both aggregations, comparing Lasso performance with and without augmentation.

📦 What's Included

Full results: Normalized RMSE across models and settings.
Code for model training, preprocessing (standardization, PCA), and MixUp implementation.
Jupyter notebooks: Association rule mining and result analysis.
Custom aggregation scripts (13 statistics).
📉 Key Insights

Lasso outperformed XGBoost and LSTM, with improvements of 17.0% (Valence) to 28.7% (Interest) over LSTM.
XGBoost underperformed due to limited tuning, though it captured non-linear patterns.
MixUp showed mixed results: best with weight = 10 (our aggregation) and weight = 0 (Itamar’s), but rarely improved RMSE significantly.
Both models struggled with extreme values, suggesting underfitting.
Feature importance for "Interest" highlighted 0.75 and 0.9 percentiles.
