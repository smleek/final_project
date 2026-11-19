"""Paper-aligned MLSMOTE using the updated cleaning/classifying notebook settings.

Run after cleaning.ipynb creates cleaned_animes.csv:
    python mlsmote_updated.py --data cleaned_animes.csv

Reference: Charte et al. (2015), MLSMOTE, Algorithms 1 and 2.
No Gaussian oversampling is used.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer


def imbalance(y):
    counts = np.asarray(y.sum(axis=0), dtype=float)
    ratios = np.full(len(counts), np.nan)
    valid = counts > 0
    if valid.any():
        ratios[valid] = counts.max() / counts[valid]
    return ratios, float(np.nanmean(ratios)) if valid.any() else np.nan


def mlsmote(X, y, k=5, random_state=1001):
    """Generate one synthetic sample per eligible seed in each minority-label bag.

    MeanIR is fixed at the beginning; each label's IRLbl is recomputed as
    augmented counts change. Neighbor bags use original training observations.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=np.uint8)
    if X.ndim != 2 or y.ndim != 2 or len(X) != len(y):
        raise ValueError('X and y must be 2D arrays with equal row counts')
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('X must be finite and y must be binary')
    if k < 1:
        raise ValueError('k must be >= 1')
    rng = np.random.default_rng(random_state)
    initial_ir, mean_ir = imbalance(y)
    counts = y.sum(axis=0).astype(np.int64)
    generated_x, generated_y = [], []
    generated_for = np.zeros(y.shape[1], dtype=int)
    skipped = []

    for label in range(y.shape[1]):
        if not np.isfinite(initial_ir[label]) or initial_ir[label] <= mean_ir:
            continue
        bag = np.flatnonzero(y[:, label])
        if len(bag) < 2:
            skipped.append(label)
            continue
        n_neighbors = min(k, len(bag) - 1)
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean')
        nn.fit(X[bag])
        neighbor_indices = nn.kneighbors(X[bag], return_distance=False)
        for local_seed, seed in enumerate(bag):
            if counts.max() / counts[label] <= mean_ir:
                break
            local_other = neighbor_indices[local_seed]
            local_other = local_other[local_other != local_seed][:n_neighbors]
            if not len(local_other):
                continue
            other = bag[local_other]
            chosen = int(rng.choice(other))
            synthetic_x = X[seed] + rng.random() * (X[chosen] - X[seed])
            votes = y[seed].astype(int) + y[other].sum(axis=0).astype(int)
            # Paper's ranking approach: at least half of seed + neighbors.
            synthetic_y = (2 * votes >= len(other) + 1).astype(np.uint8)
            assert synthetic_y[label] == 1
            generated_x.append(synthetic_x)
            generated_y.append(synthetic_y)
            generated_for[label] += 1
            counts += synthetic_y

    if generated_x:
        return (np.vstack((X, np.asarray(generated_x))),
                np.vstack((y, np.asarray(generated_y))),
                generated_for, skipped)
    return X.copy(), y.copy(), generated_for, skipped


def predict_per_genre(X_train, y_train, X_test, threshold=0.3):
    """Match classifying.ipynb: one binary logistic model per genre."""
    predictions = np.zeros((len(X_test), y_train.shape[1]), dtype=np.uint8)
    for j in range(y_train.shape[1]):
        positives = int(y_train[:, j].sum())
        if positives == 0:
            continue
        if positives == len(y_train):
            predictions[:, j] = 1
            continue
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train[:, j])
        predictions[:, j] = (model.predict_proba(X_test)[:, 1] >= threshold).astype(np.uint8)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('cleaned_animes.csv'))
    parser.add_argument('--output-prefix', type=Path, default=Path('mlsmote_updated'))
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--threshold', type=float, default=0.3)
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be between 0 and 1')
    df = pd.read_csv(args.data)
    if not {'text', 'genres'}.issubset(df.columns):
        parser.error('CSV must be output of updated cleaning.ipynb (text and genres columns)')
    df = df.dropna(subset=['text', 'genres']).copy()
    df = df[(df['text'].str.strip() != '') & (df['genres'].str.strip() != '')].reset_index(drop=True)
    # IMPORTANT: split on pipe; mlb.fit_transform(df['genres']) would encode characters!
    genre_lists = df['genres'].map(lambda s: [g.strip() for g in s.split('|') if g.strip()])
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(genre_lists).astype(np.uint8)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['text'].to_numpy(), y, test_size=0.2, random_state=args.seed
    )
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    n_components = min(100, min(X_train_tfidf.shape) - 1)
    if n_components < 1:
        raise ValueError('Insufficient observations/features for TruncatedSVD')
    svd = TruncatedSVD(n_components=n_components, random_state=args.seed)
    X_train = svd.fit_transform(X_train_tfidf)
    X_test = svd.transform(X_test_tfidf)
    X_aug, y_aug, generated_for, skipped = mlsmote(X_train, y_train, k=args.k, random_state=args.seed)

    baseline_pred = predict_per_genre(X_train, y_train, X_test, args.threshold)
    mlsmote_pred = predict_per_genre(X_aug, y_aug, X_test, args.threshold)
    scores = pd.DataFrame([
        {'Method': name,
         'Macro F1': f1_score(y_test, pred, average='macro', zero_division=0),
         'Micro F1': f1_score(y_test, pred, average='micro', zero_division=0)}
        for name, pred in [('Baseline', baseline_pred), ('MLSMOTE', mlsmote_pred)]
    ])
    counts_before = y_train.sum(axis=0)
    counts_after = y_aug.sum(axis=0)
    genre_report = pd.DataFrame({
        'Genre': mlb.classes_, 'Training Count': counts_before,
        'After MLSMOTE': counts_after, 'Added': counts_after - counts_before,
        'Directly Seeded': generated_for,
        'Baseline F1': f1_score(y_test, baseline_pred, average=None, zero_division=0),
        'MLSMOTE F1': f1_score(y_test, mlsmote_pred, average=None, zero_division=0),
    })
    genre_report['MLSMOTE Improvement'] = genre_report['MLSMOTE F1'] - genre_report['Baseline F1']
    scores.to_csv(f'{args.output_prefix}_scores.csv', index=False)
    genre_report.sort_values('Training Count').to_csv(f'{args.output_prefix}_genres.csv', index=False)
    print(f'Genres: {len(mlb.classes_)}; train: {len(y_train)}; test: {len(y_test)}')
    print(f'Synthetic observations: {len(y_aug) - len(y_train)}')
    print(f'MeanIR: {imbalance(y_train)[1]:.3f} -> {imbalance(y_aug)[1]:.3f}')
    print(scores.to_string(index=False))
    if skipped:
        print('Skipped singleton minority labels:', ', '.join(mlb.classes_[skipped]))
    print(f'Reports: {args.output_prefix}_scores.csv, {args.output_prefix}_genres.csv')


if __name__ == '__main__':
    main()
