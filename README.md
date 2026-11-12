# STAT 486 Project Plan

See ```project-plan.md``` for further details. 

## Project Proposal - October 27, 2026

We want to investigate an alternative to SMOTE (Synthetic Minority Over-sampling Technique) that incorporates more variability into data without exacerbating data imbalance issues. 

SMOTE is a widely used oversampling technique that helps to adjust for data that does not fully encompass the possibilities of the real world, or to help models learn characteristics of minority classes. However, this approach primarily generates observations along local line segments and may not adequately capture the broader variability of a minority class. We want to investigate whether incorporating global class-level information can improve classification performance while avoiding excessive overlap between classes. Our results could help clarify when global distribution-based oversampling is useful compared with local interpolation and improve classification on minority classes.

While we think that our new method might be able to improve macro F1-score relative to standard SMOTE, it’s possible that our proposed framework could make inferences that are wholly incorrect. We hope to create a method that maintains the accuracy seen in SMOTE, but reduces some of the drawbacks created by oversampling and introducing extra variability.

To assess our model, we will work with the Kaggle anime synopsis dataset at https://www.kaggle.com/datasets/battos/myanimelist-animes.

## Data Checkpoint - November 5, 2026

### Data

The data was obtained from https://www.kaggle.com/datasets/battos/myanimelist-animes. We cleaned the data by removing duplicates and missing synopses. See the cleaned dataset at  [`cleaned_animes.csv`](cleaned_animes.csv). 

Regarding cleaning and preprocessing, which was done in [`cleaning.ipynb`](cleaning.ipynb), we care mainly about the ```title_english```, ```synopsis```, and ```genres``` fields, so we selected those. We removed any rows with null values for ```synopsis``` and ```genres```.

### Evaluation plan

We will use synopsis and the title as predictor variables. We will vectorize the text. Using SMOTE we will attempt to classify the genres. We will then evaluate the same data using our modified SMOTE model for unbalanced classes and compare performance. To avoid data leakage, we will split train and test sets before standardizing or fitting a model.

### First runnable result

What we ran: We trained a logistic regression model to guess an anime’s genres from its synopsis, once on the normal data and once on data upsampled with SMOTE. As we expected, the SMOTE model did better: it scored a weighted F1 of 0.3702, compared with 0.2761 without SMOTE.

Where: [`applying_SMOTE.ipynb`](applying_SMOTE.ipynb).

Next: We plan to build our own version of SMOTE that creates new samples in a different way. One idea is to use a weighted average of the class instead of SMOTE’s nearest neighbors. But we plan to come up with better ways.

### Working Investigation - November 12, 2026

Please see [`working_investigation.md`](working_investigation.md) for a more detailed overview of our efforts. Code for this checkpoint can be found in [`global_gauss_sample/classifying.ipynb`](global_gauss_sample/classifying.ipynb).


