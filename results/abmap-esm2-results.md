# AbMAP (ESM-2) Ridge Results

These results use pretrained AbMAP embeddings (ESM-2 backbone, CDR augmentation) and a ridge regression head.

| Task | Model | RMSE | MAE | R2 | Pearson | Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| koenig_binding_g6 | abmap_esm2_ridge | 0.561696 | 0.411882 | -0.030366 | 0.360576 | 0.333745 |
| koenig_expression_g6 | abmap_esm2_ridge | 0.497629 | 0.389913 | 0.236787 | 0.556623 | 0.544411 |
| warszawski_binding_d44 | abmap_esm2_ridge | 1.477427 | 1.178742 | -0.316288 | 0.348840 | 0.345473 |
