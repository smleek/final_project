# Working Investigation

**Checkpoint:** November 12, 2026, by 11:59 PM Mountain Time  
**Team:** Brayden Humpherys, Alyssa Wilson, Sujin Leek
**Route:**A
**Date:** November 12, 2026

## Working code and evidence

Our attempts to provide an alternative to SMOTE that is more distributionally-aware and increases Macro-F1 Score are detailed below. 

### Inputs, Steps, and Command

The inputs for the investigation were the cleaned anime dataset, which was vectorized via TF-IDF and reduced to 100 components using Truncated SVD. Multilabel binarization was also applied to the 21 genres. 

##### Reproduction Command 

This command should be executed within the notebook after executing the entire notebook in order to ensure all required functions have been defined and packages have been imported. 

``` {python}
pred_global = fit_oversampled_model(X_train_svd, X_test_svd, y_train, method="global", alpha=0.5)
```

### Results

Using the cleaned dataset [`cleaned_animes.csv`](cleaned_animes.csv) from the past checkpoint, we were able to compare SMOTE and global gaussian oversampling on the data. Both SMOTE and global gaussian oversampling improved the macro f1 score while decreasing the Micro F1 score, but global oversampling was an improvement over SMOTE in both scores. 

### Performance
| Method  | Macro F1 | Micro F1
| ------------- |:-------------:|:--------------|
| Baseline (no oversampling)      | 0.244349     | 0.51008
| SMOTE      | 0.252019     | 0.252019 | 0.31472
| Global Gaussian Oversampling      | 0.297903     | 0.35897


The code can be found in [`global_guass_sample/classifying.ipynb`](global_guass_sample/classifying.ipynb). The functions of note are ```fit_oversampled_model``` and ```global_oversample```. 

## Next Steps

While global gaussian oversampling has produced favorable results, we feel that there is a lot of optimization left to be done on this technique, and we hope to find a way to distinguish this technique from the existing literature on global gaussian oversampling. 

The multi-label aspect of the data is also something we are considering further. Because our dataset has multiple labels attached to most inputs, we are considering switching to a dataset that is still a multi-class classification problem but has only one label per input. We’d also like to investigate whether combining global gaussian oversampling with MLSMOTE (Multi-label SMOTE) improves performance. 

