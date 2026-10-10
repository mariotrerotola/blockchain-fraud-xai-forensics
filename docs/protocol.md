# Replication protocol

Written before any notebook in this repository was executed. It fixes how each
step of the paper is implemented, so that the reported numbers are whatever the
code returns and not the outcome of choices made after seeing them.

## Rules

1. Each notebook implements the corresponding section of the paper. Where the
   paper is silent, the choice below is used and stated in the notebook.
2. Seed 42 everywhere (NumPy, scikit-learn, imbalanced-learn, TensorFlow).
3. Hyperparameters are selected only with the grids declared here, by stratified
   five-fold cross-validation on the training partition. The test partition is
   used once, for the final numbers.
4. A number is reported as produced. No grid is widened, no threshold is moved,
   no seed is changed after a result has been observed.

## Shared choices

| Step | Implementation |
|---|---|
| Label harmonisation (Sec. 2) | An entity is fraud if any source flags it as fraud: licit rows whose address also appears in a fraud source are dropped. Exact duplicates are dropped. |
| Split | 80:20, stratified, `random_state=42`. |
| Metrics | Macro precision, recall and F1; accuracy; balanced accuracy; ROC-AUC; FPR = FP / (FP + TN) with fraud as the positive class. |
| SHAP | `shap.TreeExplainer`, path-dependent, on the test partition. |
| CIU | `py-ciu`, mean CI and CU over 200 fraud-class test instances drawn with seed 42. |
| DEXiRE | An MLP is distilled from the ensemble's training-set predictions; DEXiRE (TREERULE) extracts rules from the MLP. Fidelity is measured against the ensemble on the test partition. |

## Bitcoin (Sec. 3)

| Step | Implementation |
|---|---|
| Activity filter | `total_in_degree >= 2` and `total_out_degree >= 2`, the two degree features computed by the feature code. |
| Recent sample | Isolation Forest, `contamination=0.05`, on standardised features; rows flagged as anomalies are removed. |
| Missing values | Rows with undefined time intervals (single incoming transfer per counterparty) are dropped. |
| Scenarios | 1: ChainAbuse vs recent sample. 2: ChainAbuse vs Elliptic++ licit. 3: Elliptic++ fraud vs Elliptic++ licit. |
| Models | RF, MLP, SVM, each in an imbalanced-learn pipeline with SMOTE applied to training folds only. |
| RF grid | `n_estimators` {200, 500}, `max_depth` {20, None}, `min_samples_leaf` {1, 2}. |
| MLP grid | `hidden_layer_sizes` {(64,), (128, 64)}, `alpha` {1e-4, 1e-3}. |
| SVM grid | RBF kernel, `C` {1, 10}, `gamma` {"scale"}. |
| Selection score | Macro-F1. |
| Confident Learning | Scenario 2, training partition only. Out-of-fold class probabilities from five-fold CV with the selected RF; ChainAbuse fraud rows with self-confidence below 0.3 are removed; the RF is refit and evaluated on the unchanged test partition. |

## ERC-20 (Sec. 4)

| Step | Implementation |
|---|---|
| Feature matrix | All numeric columns produced by the feature code; constant columns dropped. |
| Collinearity pruning | Scenario CA+CMC. Stage 1: keep `_mean` statistics and scalar descriptors, drop one feature from each pair with \|rho\| >= 0.999 (keeping the one more correlated with the target). Stage 2: on the training partition, repeatedly drop the weaker-target member of the most correlated pair until no pair has \|rho\| >= 0.7. |
| Model | Balanced Random Forest. |
| BRF grid | `n_estimators` {200, 300}, `max_depth` {20, None}, `min_samples_leaf` {1, 2}. |
| Selection score | Macro-F1. |
| Scenario ranking | (AUC-PR + F2) / 2 on the test partition, fraud as positive class. |
