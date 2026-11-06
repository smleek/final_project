# Data and Reference Result

**Checkpoint:** November 5, 2026, by 11:59 PM Mountain Time  
**Team:** Brayden Humpherys, Alyssa Wilson, Sujin Leek
**Date:** November 5, 2026

Use any prompts that help you describe your progress. Add a dated entry to the root `README.md` of your team's repository and link the relevant code and output there; this file is optional supporting material.

### Data

The data was obtained from https://www.kaggle.com/datasets/battos/myanimelist-animes. We cleaned the data by removing duplicates and missing synopses. See the cleaned dataset at  ```cleaned_animes.csv```. 

Our data is an anime dataset from Kaggle. We care mainly about the ```title_english```, ```synopsis```, and ```genres``` fields, so we selected those. We removed any rows with null values for ```synopsis``` and ```genres```.

### Evaluation plan

We will use synopsis and the title as predictor variables. We will vectorize the text. Using SMOTE we will attempt to classify the genres. We will then evaluate the same data using our modified SMOTE model for unbalanced classes and compare performance. To avoid data leakage, we will split train and test sets before standardizing or fitting a model.

### First runnable result

What we ran: We trained a logistic regression model to guess an anime’s genres from its synopsis, once on the normal data and once on data upsampled with SMOTE. As we expected, the SMOTE model did better: it scored a weighted F1 of 0.3702, compared with 0.2761 without SMOTE.

Where: ```applying_SMOTE.ipynb```.

Next: We plan to build our own version of SMOTE that creates new samples in a different way. One idea is to use a weighted average of the class instead of SMOTE’s nearest neighbors. But we plan to come up with better ways.
