# Identifying Age-Related Conditions

This folder contains some of the code and data I used in a Kaggle competition, "ICR - Identifying Age-Related Conditions" [1].  This competition was hosted by InVitro Cell Research, LLC, a company focused on regenerative and preventative medicine.  Competition entrants were given a list of patients that did or did not have one of three age-related conditions.  Entrants were also given data about these patients, but were not informed what each column of data represented.  The goal of this competition was to predict which patients in a test set had a condition or did not.  More precisely, the metric used to evaluate each submission was a logarithmic loss weighted to balance the importance of correctly predicting positive and negative results.  Over 6000 teams entered the competition.  My submission scored in the top 25%.

The file titled "identifying-age-related-conditions-search.ipynb" is a Jupyter notebook used for a hyperparameter search of my model.  The model is an XGBoost classifier [2].  The hyperparameter search uses a grid of four parameters and evaluates the best model with a stratified 5-fold cross validation.  The training data is provided in the file titled "train.csv."

Bibliography:  
[1] https://www.kaggle.com/c/icr-identify-age-related-conditions  
[2] https://arxiv.org/abs/1603.02754
