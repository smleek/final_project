# STAT 486 Project Plan

**Due:** Tuesday, October 27, 2026, at 11:59 PM Mountain Time  
**Team:** Sujin Leek, Brayden Humphreys, Alyssa Wilson  
**Repository:** https://github.com/smleek/final_project 
**Date updated:** 10/27/2026

Use these prompts to plan your investigation. Keep the answers focused; revise the plan when evidence or instructor feedback changes your direction. The Oct. 27 checkpoint still requires a new, dated project-plan entry in the root `README.md` of your team's GitHub repository. Link this document there if you use it; this document does not replace that entry.

## Project question and purpose

We want to investigate an alternative to SMOTE (Synthetic Minority Over-sampling Technique) that incorporates more variability into data without exacerbating data imbalance issues. 

SMOTE is a widely used oversampling technique that helps to adjust for data that does not fully encompass the possibilities of the real world, or to help models learn characteristics of minority classes. However, this approach primarily generates observations along local line segments and may not adequately capture the broader variability of a minority class. We want to investigate whether incorporating global class-level information can improve classification performance while avoiding excessive overlap between classes. Our results could help clarify when global distribution-based oversampling is useful compared with local interpolation and improve classification on minority classes.

While we think that our new method might be able to improve macro F1-score relative to standard SMOTE, it’s possible that our proposed framework could make inferences that are wholly incorrect. We hope to create a method that maintains the accuracy seen in SMOTE, but reduces some of the drawbacks created by oversampling and introducing extra variability.

## Chosen route

We plan to follow route **A. Develop something new** in an ML method or system. This route applies best to our question of whether a SMOTE alternative for text data is realistic. 


## Data source and access

We will use the MyAnimeList animes dataset from Kaggle. https://www.kaggle.com/datasets/battos/myanimelist-animes 
In this dataset, one observation represents a single anime TV show, with an attached synopsis and genre.This data is available on Kaggle. Since we plan on using TF-IDF vectorization, compute should be manageable. 

There are no limits to the use of this data. We do not anticipate handling sensitive personal information, and the data is free to distribute with attributions. There are some duplicates within the data, and some rows involve Japanese rather than English. We will remove exact duplicates and translate entries where applicable. 

### Route A: New method or system component

We will begin with logistic regression using TF-IDF features extracted from anime synopses without oversampling, then implement standard SMOTE and apply it to the same reduced feature representation used by our proposed method. We are considering taking an average plus random noise of all features in a minority class to then create the new minority class sample. Where SMOTE generates samples using k neighbors, we might generate new samples using global class information.

Using identical training, validation, and test splits, as well as the same preprocessing and dimensionality-reduction procedure, we will compare our new method’s macro-f1-score or RMSE with and without our upsampling technique, benchmarking against SMOTE. To initially verify efficacy before proceeding, we will construct a small numerical example with three minority-class feature vectors and verify that the estimated mean and standard deviation are correct. If the new change results in worse RMSE or Macro f1-score, we will reiterate our efforts. 


## Initial reference and proposed contribution

The baseline for this project is SMOTE on logistic regression for text classification. Regarding data and code, we will credit the MyAnimeList dataset and the original SMOTE publication by Chawla et al. (2002). We expect to use standard Python libraries, including pandas, scikit-learn, and imbalanced-learn, for data processing, feature extraction, dimensionality reduction, classification, and baseline oversampling. We will cite relevant documentation and any additional methods or code adopted from existing work. Our implementation and evaluation code will be documented in our repository.

## Evaluation and evidence

We will divide the original dataset into stratified training, validation, and test sets. We will use validation performance to choose necessary hyperparameters, then evaluate the selected models on a separated test set. To assess stability, we will repeat the comparison across multiple random seeds where computationally feasible and report the variability in performance.

Our primary metric will be macro f1-score, which gives equal weight to f1-score of all classes and helps us to assess how the model is performing on minority classes. 

Our baseline is the standard SMOTE model as found in the initial paper previously cited. It will be our benchmark and we will compare our results to how it would perform on the data. We will use plots to verify results and help us to understand our progress. 

## Risks and backup scope

It is possible that our new model could underperform SMOTE, meaning that it performed worse than the baseline model on our data. If this is the result we get, we will analyze where and why it went wrong as well as include in our report other ideas about how our model could be improved.

In the case that our data is too large and the model is too computationally costly for our purposes, we can filter down to a smaller dataset. This would be a little bit tricky given we are trying to use unbalanced data, but would still be a viable option. We could also try simplifying the model we choose to use.

If our initial dataset is not sufficiently unbalanced, this could lead to a failure in our model because it will be optimized for unbalanced data. We may need to find a different source for our data and change exactly what we are analyzing. Fortunately, we care more about the model than the data so this will be less of a drastic problem. In this case, we would likely fall back to a benchmark dataset.

## Reproducibility and team contributions

We will conduct most code writing in .ipynb files stored in a Notebooks folder in the repository. Instructions will be stored in the README and other instruction files. Reproduction of results, that being a better macro f1-score on classification of anime dataset, should only require the provided dataset and the provided Python script. 

Team members will coordinate to write code and design a model through in-person and asynchronous collaboration. Instructor guidance on how to prioritize macro f1-score on datasets would be helpful to us, as we do not have extensive classification expertise. 

## Sources

Data source: 
https://www.kaggle.com/datasets/battos/myanimelist-animes 

Original SMOTE paper: 
https://arxiv.org/abs/1106.1813 

Other SMOTE iterations: 
https://link.springer.com/chapter/10.1007/11538059_91 
https://arxiv.org/abs/2505.13434 
