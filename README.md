# SI-GuidedProject-133368-1661496190
Hospital Readmission Prediction Using Machine Learning

Drive Link: https://drive.google.com/drive/folders/1w_vQENT8vEpfVKbItQ1imM73zPZF9LI5?usp=sharing

Project Description:

The prime objective of this project is to predict whether a person who has been diagnosed with diabetes will be readmitted to the hospital within 30 days or not.

This will help the hospital in being ready for any number of patients that could be readmitted in the future.

The problem associated is a binary classifcation. Since the target variable in the dataset is quite imbalanced, SMOTE algorithm was used to oversample the minority class.

A number of models were used such as logistic regression, KNN Classifier, Decision Tree Classifier, Random Tree Classifier, Adaboost Classifier etc.
We will train and test the data with these algorithms.
From this, the best model is selected and saved in pkl format. We will also be deploying our model locally using Flask.
Out of all the models, random forest classifier is found to perform the best.
