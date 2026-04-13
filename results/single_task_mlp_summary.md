# Single-task MLP Summary

| task               | family   |   spearman |   pearson |    mae |   rmse |      r2 |    n |
|:-------------------|:---------|-----------:|----------:|-------:|-------:|--------:|-----:|
| koenig_binding     | abmap    |     0.298  |    0.3403 | 0.4391 | 0.5882 | -0.1207 | 4275 |
| koenig_binding     | esm2     |     0.3181 |    0.3313 | 0.5588 | 0.7293 | -0.7229 | 4275 |
| koenig_expression  | abmap    |     0.6276 |    0.6481 | 0.3414 | 0.4388 |  0.4097 | 4275 |
| koenig_expression  | esm2     |     0.8233 |    0.8493 | 0.2244 | 0.3019 |  0.7205 | 4275 |
| warszawski_binding | abmap    |     0.3871 |    0.3895 | 0.9999 | 1.2519 |  0.0624 | 2048 |
| warszawski_binding | esm2     |     0.55   |    0.5368 | 1.0238 | 1.2954 | -0.0039 | 2048 |