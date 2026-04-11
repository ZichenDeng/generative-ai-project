# Baseline Results

These results use deterministic 10-fold splits and standard-library sequence features only. They are intended as the first milestone reference point before ESM-2 and AbMAP embeddings are added.

| Task | Model | RMSE | MAE | R2 | Pearson | Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| koenig_binding_g6 | sequence_features_ridge | 0.546410 | 0.385399 | 0.029327 | 0.181060 | 0.173535 |
| koenig_binding_g6 | train_mean | 0.554981 | 0.390607 | -0.001198 | -0.000000 | nan |
| koenig_expression_g6 | sequence_features_ridge | 0.550028 | 0.442583 | 0.069966 | 0.269008 | 0.246561 |
| koenig_expression_g6 | train_mean | 0.570815 | 0.469896 | -0.001043 | 0.000000 | nan |
| warszawski_binding_d44 | sequence_features_ridge | 1.256381 | 1.030706 | 0.049107 | 0.245783 | 0.259989 |
| warszawski_binding_d44 | train_mean | 1.292792 | 1.066933 | -0.005579 | 0.000000 | nan |
