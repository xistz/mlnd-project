# Machine Learning Nanodegree Capstone Proposal

Chan Jian Hui Jonathan [email](tmxistz@gmail.com)
26 December 2018

*In this project, the goal is to built a classifier to predict the type of cuisine given the recipe of the dish. Food is a cultural heritage, it puts on display what each culture celebrates about its geography. Through this project, I hope to learn, can a cuisine be defined by its ingredients.*

## Domain Background

Supervised learning is a class of machine learning algorithms that learns from data to make a prediction. There are 3 general types of predictions, classification, regression, and anomaly detection. Regression predicts continuous values, such as housing prices. Classification predicts distinct categories, such as dog breeds. Anomaly detection idenifies unusual datapoints. Since being introduced in the industry, supervised learning has streamlined the effort required to predict stock prices, identify credit card fraud, and make recommendations. Supervised learning will only further impact our lives as more industries leverage on data to understand trends. 

## Problem Statement

In this project, the problem being investigated the classification of cuisines based on the ingredients in a recipe. Food is a cultural heritage, it puts on display what the culture celebrates about its geography. For example, when we think of Indian cuisine, we think of curry, and when we think of Japan, Sushi. How do we distinguish cuisines; are there unique ingredients? If a classifier can be built to predict the cuisine from ingredients to a high degree of accuracy, it would suggest that cuisines can be delineated by their ingredients.

## Dataset and inputs

In this project, the data used will come from the [What's Cooking?](https://www.kaggle.com/c/whats-cooking-kernels-only) Kaggle competition. The dataset contains 3 files, a sample submission for the Kaggle compettion, the testing dataset, and the training dataset. The format of both the testing and training datasets are in `JSON` and contains 3 fields, `id`, `cuisine`,  and `ingredients`. There are `3669` entries and `3385` entries in the testing and training datasets respectively.

## Solution Statement

In this project, a decision tree will be used to built a classifier for predicting the cuisine. The decision tree is a god algorithm for this problem is because of the size of the dataset. The decision tree achieves high accuracy despite the size of dataset because the decision tree automatically learns which features(ingredients) are the most important when predicting the target(cuisine) and emphasizes them.

## Benchmark model

According to the public leaderboard, the highest scoring model currently achieves a score of `0.82783`, while the sample benchmark achieves a score of `0.19267`. The goal of this project will be to achieve a score greater than the sample benchmark.

## Evaluation Metrics

The metrics used in the Kaggle competition is categorization accuracy, the percentage of recipes whose cusine has been correctly predicted in the testing dataset. Since precision and recall is not of importance, accuracy will be a sufficient metric for this problem.

## Project Design

Since ingredients in the recipe are categorical, the data in the training set will first need to be preprocessed using one hot encoding. In addition, since there are lesser number of entries in the training set as compared to the testing set, K-Fold cross validation will be used to allow the decision tree to learn as much as possible from the training set. After obtaining a first decision tree, we can then optimize the model further by using grid search on paramters such as the maximum depth, the minimum number of samples in each leaf, minimum number of samples before splitting, and the maximum number of features. Once the hyperparamters of the optimal model is has been determined, a final model will be built using the optimal hyperparamters and trained on the entire training set. That model will then be used on the testing set to determine the model's accuracy.

## References

- [How to choose algorithms for Azure Machine Learning Studio](https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-choice)
- [What's Cooking?](https://www.kaggle.com/c/whats-cooking-kernels-only#description)