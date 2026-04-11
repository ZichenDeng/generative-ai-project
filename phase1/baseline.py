import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer

def run_baseline_mean(csv_path: str) -> float:
    """return mean spearman coefficient across folds by just computing mean prediction"""
    df = pd.read_csv(csv_path)
    spearman_scores = []

    for fold in range(10):
        train = df[df['cv_fold'] != fold]
        test = df[df['cv_fold'] == fold]

        # mean of training labels for every test sample
        mean_pred = np.mean(train['fitness'])
        preds = [mean_pred] * len(test)

        spearman, _ = stats.spearmanr(preds, test['fitness'])
        spearman_scores.append(0.0 if np.isnan(spearman) else spearman)

    return np.mean(spearman_scores)

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

def run_baseline_sequence(csv_path: str) -> float:
    """return mean spearman coefficient by fitting ridge regression on frequency of each amino acid"""
    df = pd.read_csv(csv_path)
    spearman_scores = []

    for fold in range(10):
        train = df[df['cv_fold'] != fold]
        test = df[df['cv_fold'] == fold]

        # build features by counting frequency of amino acids
        vec = CountVectorizer(analyzer='char', vocabulary=list(amino_acids), lowercase=False)
        train_X = vec.fit_transform(train['heavy'] + train['light'])
        test_X = vec.transform(test['heavy'] + test['light'])

        train_y = train['fitness']
        test_y = test['fitness']

        # can adjust alpha value
        model = Ridge(alpha=0.1)
        model.fit(train_X, train_y)
        preds = model.predict(test_X)
        # compute spearman ratio
        spearman, _ = stats.spearmanr(preds, test_y)
        spearman_scores.append(0.0 if np.isnan(spearman) else spearman)

    return np.mean(spearman_scores)

def main():
    # load data
    expression_data = 'data/processed/koenig2017mutational_er_g6_folds.csv'
    binding_data = 'data/processed/koenig2017mutational_kd_g6_folds.csv'
    # mean/simple baseline
    bind_mean = run_baseline_mean(binding_data)
    expression_mean = run_baseline_mean(expression_data)
    # simple sequence-feature baseline
    bind_sequence = run_baseline_sequence(binding_data)
    expression_sequence = run_baseline_sequence(expression_data)

    print(f"Binding mean baseline: {bind_mean:.4f}")
    print(f"Expression mean baseline: {expression_mean:.4f}")
    print(f"Binding sequence baseline: {bind_sequence:.4f}")
    print(f"Expression sequence baseline: {expression_sequence:.4f}")

if __name__ == "__main__":
    main()
