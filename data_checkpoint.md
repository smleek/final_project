# Data and Reference Result

**Checkpoint:** November 5, 2026, by 11:59 PM Mountain Time  
**Team:** Brayden Humpherys, Alyssa Wilson, Sujin Leek
**Date:** November 5, 2026

Use any prompts that help you describe your progress. Add a dated entry to the root `README.md` of your team's repository and link the relevant code and output there; this file is optional supporting material.

## Data

The data was obtained from https://www.kaggle.com/datasets/battos/myanimelist-animes. We cleaned the data by removing duplicates and missing synopses. 

Our data is an anime dataset from Kaggle. We care mainly about the title_english, synopsis, and genres fields, so we selected those. We removed any rows with null values for synopsis and genres. Because the dataset is just a collection of anime, it will not affect our analysis if we exclude points. We plan to fill in the titles that are missing by either looking them up or using AI to create them. This also would not affect our analysis.

## Evaluation plan

We will use synopsis and the title as predictor variables. We will vectorize the text. Using SMOTE we will attempt to classify the genres. We will then evaluate the same data using our modified SMOTE model for unbalanced classes and compare performance. To avoid data leakage we will split train and test sets before standardizing or fitting a model.

## First runnable result

- What baseline or other reference did you run? What did it produce?
- Where are the code and output? Link them, and include the command needed to rerun the result if available.
- What did this first result suggest, and what will you try next?
