# Milestone Benchmark Summary

This table combines the currently completed milestone benchmark rows.

Completed:

- train-fold mean baseline
- sequence-feature ridge baseline
- ESM-2 embedding + ridge baseline on the two primary Koenig tasks

Not completed yet:

- AbMAP embedding baseline
- small MLP head
- ESM-2 on Warszawski d44

## Current Results

| Task | Model | RMSE | MAE | R2 | Pearson | Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| koenig_binding_g6 | train_mean | 0.554981 | 0.390607 | -0.001198 | -0.000000 | nan |
| koenig_binding_g6 | sequence_features_ridge | 0.546410 | 0.385399 | 0.029327 | 0.181060 | 0.173535 |
| koenig_binding_g6 | esm2_ridge | 0.500591 | 0.359145 | 0.181723 | 0.486198 | 0.466277 |
| koenig_expression_g6 | train_mean | 0.570815 | 0.469896 | -0.001043 | 0.000000 | nan |
| koenig_expression_g6 | sequence_features_ridge | 0.550028 | 0.442583 | 0.069966 | 0.269008 | 0.246561 |
| koenig_expression_g6 | esm2_ridge | 0.385098 | 0.297269 | 0.544007 | 0.746269 | 0.729351 |
| warszawski_binding_d44 | train_mean | 1.292792 | 1.066933 | -0.005579 | 0.000000 | nan |
| warszawski_binding_d44 | sequence_features_ridge | 1.256381 | 1.030706 | 0.049107 | 0.245783 | 0.259989 |

## Readout

The first ESM-2 baseline already improves substantially over the mean and simple sequence-feature baselines on both primary Koenig tasks. That supports the milestone framing: pretrained protein representations are a useful first predictor before any post-milestone surrogate-scoring or generative work.
