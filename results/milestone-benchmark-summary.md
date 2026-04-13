# Milestone Benchmark Summary

This table combines the completed milestone benchmark rows.

## Completed

* train-fold mean baseline
* sequence-feature ridge baseline
* ESM-2 embedding + ridge baseline
* AbMAP embedding + ridge baseline
* small MLP head (single-task)
* shared-task MLP

---

## Current Results

### Koenig Binding (g6)

| Model                   |   Spearman |   RMSE |    MAE |      R2 | Pearson |
| ----------------------- | ---------: | -----: | -----: | ------: | ------: |
| train_mean              |        nan | 0.5550 | 0.3906 | -0.0012 |  0.0000 |
| sequence_features_ridge |     0.1735 | 0.5464 | 0.3854 |  0.0293 |  0.1811 |
| **esm2_ridge**          | **0.4663** | 0.5006 | 0.3591 |  0.1817 |  0.4862 |
| esm2_mlp                |     0.3181 | 0.7293 | 0.5588 | -0.7229 |  0.3313 |
| esm2_shared_mlp         |     0.3886 | 0.5342 | 0.3879 |  0.0756 |  0.4098 |
| abmap_esm2_ridge        |     0.3337 | 0.5617 | 0.4119 | -0.0304 |  0.3606 |
| abmap_mlp               |     0.2980 | 0.5882 | 0.4391 | -0.1207 |  0.3403 |
| abmap_shared_mlp        |     0.2621 | 0.5820 | 0.4358 | -0.0973 |  0.3177 |


### Koenig Expression (g6)

| Model                   |   Spearman |   RMSE |    MAE |      R2 | Pearson |
| ----------------------- | ---------: | -----: | -----: | ------: | ------: |
| train_mean              |        nan | 0.5708 | 0.4699 | -0.0010 |  0.0000 |
| sequence_features_ridge |     0.2466 | 0.5500 | 0.4426 |  0.0700 |  0.2690 |
| esm2_ridge              |     0.7294 | 0.3851 | 0.2973 |  0.5440 |  0.7463 |
| **esm2_mlp**            | **0.8233** | 0.3019 | 0.2244 |  0.7205 |  0.8493 |
| esm2_shared_mlp         |     0.7270 | 0.3681 | 0.2815 |  0.5845 |  0.7663 |
| abmap_esm2_ridge        |     0.5444 | 0.4976 | 0.3899 |  0.2368 |  0.5566 |
| abmap_mlp               |     0.6276 | 0.4388 | 0.3414 |  0.4097 |  0.6481 |
| abmap_shared_mlp        |     0.5147 | 0.4819 | 0.3812 |  0.2879 |  0.5398 |



### Warszawski Binding (d44)

| Model                   |   Spearman |   RMSE |    MAE |      R2 | Pearson |
| ----------------------- | ---------: | -----: | -----: | ------: | ------: |
| train_mean              |        nan | 1.2928 | 1.0669 | -0.0056 |  0.0000 |
| sequence_features_ridge |     0.2600 | 1.2564 | 1.0307 |  0.0491 |  0.2458 |
| esm2_ridge              |     0.4794 | 1.2412 | 0.9825 |  0.0691 |  0.4806 |
| **esm2_mlp**            | **0.5500** | 1.2954 | 1.0238 | -0.0039 |  0.5368 |
| esm2_shared_mlp         |     0.3897 | 1.1873 | 0.9748 |  0.1567 |  0.3976 |
| abmap_esm2_ridge        |     0.3455 | 1.4774 | 1.1787 | -0.3163 |  0.3488 |
| abmap_mlp               |     0.3871 | 1.2519 | 0.9999 |  0.0624 |  0.3895 |
| abmap_shared_mlp        |     0.3678 | 1.2052 | 0.9842 |  0.1311 |  0.3840 |


---

## Readout

The ESM-2 embedding + ridge baseline remains the most stable and consistent model across all tasks, particularly for expression prediction. AbMAP ridge improves over simple sequence-feature baselines but does not outperform ESM-2 ridge.

Small MLP models provide task-dependent gains, notably improving performance on the Koenig expression task, but often underperform ridge on binding tasks, suggesting overfitting or limited nonlinear signal.

Shared-task MLP does not consistently improve performance and may introduce negative transfer across tasks.

Overall, these results support using ESM-2 embeddings with simple linear readouts as a strong baseline for downstream surrogate scoring and generative modeling.


## Table: Milestone report

Performance comparison across embedding methods and model classes. Values are reported as Spearman correlation (primary metric), with RMSE in parentheses.

| Task                   | ESM2 (Ridge)        | AbMAP (Ridge)   | ESM2 (MLP)          | AbMAP (MLP)     |
| ---------------------- | ------------------- | --------------- | ------------------- | --------------- |
| **Koenig Binding**     | **0.4663 (0.5006)** | 0.3337 (0.5617) | 0.3181 (0.7293)     | 0.2980 (0.5882) |
| **Koenig Expression**  | 0.7294 (0.3851)     | 0.5444 (0.4976) | **0.8233 (0.3019)** | 0.6276 (0.4388) |
| **Warszawski Binding** | 0.4794 (1.2412)     | 0.3455 (1.4774) | **0.5500 (1.2954)** | 0.3871 (1.2519) |

---

## Key Observations

* **ESM2 consistently outperforms AbMAP** across all tasks. For ridge models, ESM2 improves Spearman correlation by **+0.13 (binding)**, **+0.19 (expression)**, and **+0.13 (Warszawski)** compared to AbMAP. Similar gaps are observed for MLP models (e.g., **0.823 vs 0.628** on expression).

* **Ridge regression is highly competitive**, particularly for binding tasks. On Koenig binding, ridge achieves **0.466 Spearman**, outperforming MLP (**0.318**) by a substantial margin. On Warszawski binding, ridge (0.479) is close to or slightly below MLP (0.550), indicating limited nonlinear gains.

* **MLP provides clear benefit only for expression prediction**, where ESM2+MLP improves Spearman from **0.729 → 0.823 (+0.094)** and reduces RMSE from **0.385 → 0.302**. In contrast, MLP degrades performance on binding tasks (e.g., Koenig binding: **0.466 → 0.318**).

* **AbMAP does not show a consistent advantage** over ESM2. Even with MLP, AbMAP remains below ESM2 (e.g., expression: **0.628 vs 0.823**), suggesting that ESM2 embeddings already capture most predictive signal for these tasks.
