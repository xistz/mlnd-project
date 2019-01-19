# Machine Learning Engineer Nanodegree
## Capstone Project
Chan Jian Hui Jonathan
January 15, 2019

## I. Definition
<!-- _(approx. 1-2 pages)_ -->

### Project Overview
<!-- 
In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_
-->

Classification is an important skill. When we classify, we able to identify the object of interest and separate it from other objects. Being able to distinguish and separate objects from one other is an important skill because it then allows us to act on those objects.

It is one of the most important skills we first learn. When a baby repeats, "cake" after a parent who points at a piece of cake the baby learns not only the pronunciation of the word, the baby learns to distinguish cake from other objects. When a parent then puts the cake into the baby's mouth, the baby learns that objects identified as cake can be eaten. So the next time, when a baby sees cake, the baby is able to exclaim in delight, "cake!" before devouring it. As result, being able to distinguish cakes from toys allows us to act on the cakes and eat them.

We classify objects by considering its features. However, because objects can have numerous features, we know not to consider all of them, only an important subset. In the example of cake, we know to consider only features such as its appearance, texture, and taste but not the sound it produces. This knowledge is not instinctive, it is acquired after learning from many samples. Just as an adult is better than a baby at classifying cakes, we get better at classifying objects as we encounter more samples of the target object. That is because as we encounter more samples of the target object, we begin to learn which are the important features that help us make better classifications. 

Similar to us, machines can also learn from data and be taught to classify objects. This achieved through the use of a class of machine learning algorithms known as supervised learning algorithms. Although classification is not the only task that supervised learing algorithms can accomplish, it will be the focus of this project.

### Problem Statement
<!--
In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_
-->

In this project, the goal is to build a classifier to predict the cuisine of a recipe, given its ingredients. This project is inspired by a past Kaggle competition, [What's Cooking](https://www.kaggle.com/c/whats-cooking-kernels-only). The data that will be used in this project comes from the Kaggle competition and can be found [here](https://www.kaggle.com/c/whats-cooking-kernels-only/data).

This competition was chosen because of its emphasis on food. Food is a cultural heritage, it puts on display what each culture celebrates about its geography. As a result, the cuisines were can enjoy in the world are as numerous as the cultures themselves. We are all proud of our cuisines and quick to defend the uniqueness of our cuisine. This uniqueness lies in part because of certain ingredients can only be found in the locales of those cuisines. Therefore, I anticipate that the classification of cuisines will be defined by certain key ingredients.

I have 2 objectives for this project. The first is to build a classifier that will perform better than the sample benchmark of `0.19267` on Kaggle. The second is to better understand what defines each of the cuisines in the data.

### Metrics
<!--
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_
-->

This project will use the same metric as the Kaggle competition. The metric is categorization accuracy. The categorization accuracy is defined as the ratio of correctly classified recipes to the total number of recipes in the test dataset.

```
categorization_accuracy = no_correct / no_recipes
```

Categorization accuracy is a sufficient metric for this project because we are interested in how accurately our model predicts the cuisine given the ingredients of the recipe.


## II. Analysis
<!-- _(approx. 2-4 pages)_ -->

### Data Exploration and Exploratory Visualization
<!--
In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
- _If a dataset is present for this problem, have you thoroughly discussed certain features about the dataset? Has a data sample been provided to the reader?_
- _If a dataset is present for this problem, are statistics about the dataset calculated and reported? Have any relevant results from this calculation been discussed?_
- _If a dataset is **not** present for this problem, has discussion been made about the input space or input data for your problem?_
- _Are there any abnormalities or characteristics about the input space or dataset that need to be addressed? (categorical variables, missing values, outliers, etc.)_

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_
-->

There are 3 files provided in the Kaggle competion, `train.json`, `test.json`, and `sample_submission.csv`. Only `train.json` and `test.json` will be used for this project. According to Kaggle, this dataset was provided by Yummly.

After loading `train.json` and `test.json` using `pandas.read_json`, we find that there are 39,774 recipes in the train dataset and 9944 recipes in the test dataset. The train dataset has 3 fields, cuisine, id, and ingredients. The test dataset only contains the fields, id, and ingredients, since cuisine is the field to be predicted. There are no recipes in the train dataset containing missing cuisine, id, or ingredients. 

There are 20 unique cuisines represented in this project. They are, greek, southern_us, filipino, indian, jamaican, spanish, italian, mexican, chinese, british, thai, vietnamese, cajun_creole, brazilian, french, japanese, irish, korean, moroccan, and russian.

![Number of Unique Cuisines]()

From the figure, we can see that the distribution of cuisines in the train dataset is not uniform. This could lead to a problem when training the model, as the model could be bias towards cuisines that are more well represented in the train dataset.

There was no information about the id field in the dataset. Id probably refers to the id of the recipe in yummly's database. Since the goal of the project is to build a classifier to predict cuisine from ingredients, the id field will be dropped when preprocessing the data.

There are 6,714 unique ingredients in this project. The most common ingredient in the train dataset is salt; it was used in 18,049 recipes. The least common ingredient was white almond bark; it was only used in 1 recipe. If ingredients are not dropped, we will need to choose a classifier like a decision tree that works well with a large number features.

![Number of Ingredients per Recipe]()

Number of Ingredients per recipe by Cuisine

Most common Ingredient of each cuisine.


### Algorithms and Techniques
<!--
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_
-->
The algorithm I intend to use for this project is the Decision Tree Classifier. It was chosen because of 2 reasons:
- This is a multiclass classification problem.
- There are a large number of ingredients(features).

Besides it speed of training and prediction accuracy, Decision Tree algorithms are also known to overfit. If required, steps could be taking to prevent overfiting by using regularization (K-Fold) or by using an ensemble classifer(Random Forest) instead.

### Benchmark
<!-- 
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_
-->

Since there are 20 cuisines in this project, one possible benchmark would to achieve a categorization accuracy greater than random choice or `5%`. However, according to the public leaderboard for the Kaggle competition, the highest scoring model currently achieves a score of `0.82783`, while the sample benchmark achieves a score of `0.19267`. Since the sample benchmark is higher than random choice, the goal of this project will be to achieve a score greater or equal to the sample benchmark instead.

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
<!--
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_
-->
The following steps will be taken when preprocesing the data:
- Drop id column as it does not contribute to the prediction of the recipe's cuisine.
- One hot encode the ingredients because ingredients are categorical data.

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?