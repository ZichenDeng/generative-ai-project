# Final Out-of-Fold (OOF) Results — ESMC-600M Embeddings

This section summarizes the out-of-fold (OOF) performance of models trained on ESMC-600M embeddings across three benchmark antibody datasets. Results are reported using cross-validation folds provided in each dataset.

Two models are evaluated:

- **Ridge regression** with fixed regularization parameter ($\alpha = 1$)
- **Multi-layer perceptron (MLP)** with two hidden layers

---

## Summary Table

| Dataset | Model | Spearman | Pearson | RMSE | MAE | R² |
|:--|:--|--:|--:|--:|--:|--:|
| Koenig Expression (g6) | MLP | **0.8526** | **0.8751** | **0.2767** | **0.2047** | **0.7652** |
| Koenig Expression (g6) | Ridge ($\alpha=1$) | 0.7392 | 0.7522 | 0.4092 | 0.3127 | 0.4865 |
| Koenig Binding (g6) | MLP | **0.4297** | **0.4399** | **0.5836** | **0.4424** | **-0.1033** |
| Koenig Binding (g6) | Ridge ($\alpha=1$) | 0.3919 | 0.4141 | 0.6242 | 0.4781 | -0.2624 |
| Warszawski Binding (d44) | MLP | **0.5712** | **0.5715** | **1.1696** | **0.9196** | **0.1817** |
| Warszawski Binding (d44) | Ridge ($\alpha=1$) | 0.4152 | 0.4274 | 1.6421 | 1.2950 | -0.6132 |

---

## Key Observations

### 1. MLP consistently outperforms Ridge
Across all datasets, the MLP achieves higher Spearman correlation and lower error metrics compared to ridge regression.

- Largest improvement observed in **Koenig Expression**
- Significant gains also seen in **Warszawski Binding**

---

### 2. Task-dependent performance differences

- **Koenig Expression (g6):**
  - Strong performance with MLP (Spearman = 0.8526)
  - Indicates embeddings capture expression-related signal well

- **Koenig Binding (g6):**
  - Moderate performance
  - Suggests binding prediction remains more challenging

- **Warszawski Binding (d44):**
  - MLP improves both correlation and RMSE substantially
  - Likely benefits from nonlinear modeling of embedding space

---

### 3. Ridge baseline is sensitive to regularization

The ridge results reported here use a fixed $\alpha = 1$. Previous experiments suggest that ridge performance can vary significantly with $\alpha$, and these results may not reflect the optimal linear baseline.

---

## Interpretation

These results indicate that:

- **ESMC-600M embeddings provide strong predictive features**, particularly for expression and certain binding tasks
- **Nonlinear models (MLP) are better suited** to extract signal from these embeddings
- Linear models remain competitive but require careful tuning

---

## Conclusion

The ESMC-600M embedding + MLP combination provides a strong baseline for antibody fitness prediction tasks. However, performance varies across datasets, suggesting that:

- different tasks may benefit from different embedding models or architectures
- combining multiple embedding sources (e.g., ESM-2 + ESMC) may further improve results

---