# STAT 486 Project Plan

See ```project-plan-template.md``` for further details. 

## Project Proposal - October 27, 2026

We want to investigate an alternative to SMOTE (Synthetic Minority Over-sampling Technique) that incorporates more variability into data without exacerbating data imbalance issues. 

SMOTE is a widely used oversampling technique that helps to adjust for data that does not fully encompass the possibilities of the real world, or to help models learn characteristics of minority classes. However, this approach primarily generates observations along local line segments and may not adequately capture the broader variability of a minority class. We want to investigate whether incorporating global class-level information can improve classification performance while avoiding excessive overlap between classes. Our results could help clarify when global distribution-based oversampling is useful compared with local interpolation and improve classification on minority classes.

While we think that our new method might be able to improve macro F1-score relative to standard SMOTE, it’s possible that our proposed framework could make inferences that are wholly incorrect. We hope to create a method that maintains the accuracy seen in SMOTE, but reduces some of the drawbacks created by oversampling and introducing extra variability.

To assess our model, we will work with the Kaggle anime synopsis dataset at https://www.kaggle.com/datasets/battos/myanimelist-animes.