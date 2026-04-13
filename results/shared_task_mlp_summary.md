# Shared-task MLP Summary

| task               | family   | model           |   spearman |   pearson |    mae |   rmse |      r2 |     n |
|:-------------------|:---------|:----------------|-----------:|----------:|-------:|-------:|--------:|------:|
| all_tasks_combined | abmap    | shared_task_mlp |     0.8819 |    0.987  | 0.5197 | 0.7148 |  0.9741 | 10598 |
| koenig_binding     | abmap    | shared_task_mlp |     0.2621 |    0.3177 | 0.4358 | 0.582  | -0.0973 |  4275 |
| koenig_expression  | abmap    | shared_task_mlp |     0.5147 |    0.5398 | 0.3812 | 0.4819 |  0.2879 |  4275 |
| warszawski_binding | abmap    | shared_task_mlp |     0.3678 |    0.384  | 0.9842 | 1.2052 |  0.1311 |  2048 |
| all_tasks_combined | esm2     | shared_task_mlp |     0.9058 |    0.9887 | 0.4584 | 0.665  |  0.9776 | 10598 |
| koenig_binding     | esm2     | shared_task_mlp |     0.3886 |    0.4098 | 0.3879 | 0.5342 |  0.0756 |  4275 |
| koenig_expression  | esm2     | shared_task_mlp |     0.727  |    0.7663 | 0.2815 | 0.3681 |  0.5845 |  4275 |
| warszawski_binding | esm2     | shared_task_mlp |     0.3897 |    0.3976 | 0.9748 | 1.1873 |  0.1567 |  2048 |