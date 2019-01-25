# Machine Learning Engineer Nanodegree
## Capstone Project
Chan Jian Hui Jonathan
January 25, 2019

## I. Definition
<!-- _(approx. 1-2 pages)_ -->

### Project Overview
<!-- In this section, look to provide a high-level overview of the project in layman’s terms. Questions to ask yourself when writing this section:
- _Has an overview of the project been provided, such as the problem domain, project origin, and related datasets or input data?_
- _Has enough background information been given so that an uninformed reader would understand the problem domain and following problem statement?_
-->

Classification is an important skill. When we classify, we able to identify the object of interest and separate it from other objects. Being able to distinguish and separate objects from one other is an important skill because it then allows us to act on those objects.

We classify objects by considering its features. However, because objects can have numerous features, we know not to consider all of them, only an important subset. For example, when identifying a piece of cake, we consider features such as, "does it taste sweet?", "is it covered in glaze or frosting?", and "does it have the texture of a simple sponge or maybe creamy like a cheesecake?". Therefore, when we we looking for cake among a bunch of objects, we actually performing classification and considering those features mentioned above to identify cake. We consider those features, and not feautures such as, "does it mew?". 

This knowledge of which features are important is not instinctive, it is acquired after learning from many samples. Just as an adult is better than a baby at classifying cakes, we get better at classifying objects as we encounter more samples of the target object. That is because as we encounter more samples of the target object, we begin to learn which are the important features that help us make better classifications. 

Similar to us, machines can also learn from data and be taught to classify objects. This achieved through the use of a class of machine learning algorithms known as supervised learning algorithms. Although classification is not the only task that supervised learing algorithms can accomplish, it will be the focus of this project.

This project focuses on the classification of text. Classification of text is particularly interesting because machine learning algorithms are only able to work with numeric data. As a result, text have to encoded numerically before classifying. However, as Natural Language Processing is still a growing field, our understanding how to convey meaning is still very much an ongoing study. Because there are no definite rules, different encoding techniques could be applied which would generate different behaviors in our models. 

Nevertheless, text classification is essential in our modern society. The amount of text data that we are generating evey day is enormous. Learning to make sense of it, by identifying important information from the noise will be of great benefit to our lives. This project is an excellent exercise in understanding how to work with text data and built a multiclass classifier.

### Problem Statement
<!-- In this section, you will want to clearly define the problem that you are trying to solve, including the strategy (outline of tasks) you will use to achieve the desired solution. You should also thoroughly discuss what the intended solution will be for this problem. Questions to ask yourself when writing this section:
- _Is the problem statement clearly defined? Will the reader understand what you are expecting to solve?_
- _Have you thoroughly discussed how you will attempt to solve the problem?_
- _Is an anticipated solution clearly defined? Will the reader understand what results you are looking for?_
-->

In this project, the goal is to build a classifier to predict the cuisine of a recipe, given its ingredients. This project is inspired by a past Kaggle competition, [What's Cooking](https://www.kaggle.com/c/whats-cooking-kernels-only). The data that will be used in this project can be found [here] on the competition's page.(https://www.kaggle.com/c/whats-cooking-kernels-only/data). The notebook that will be submitted for this project was adapted from the Kaggle kernel developed for the competition.

This competition was chosen because of its emphasis on food. Food is a cultural heritage, it puts on display what each culture celebrates about its geography. As a result, the cuisines we can enjoy in the world are as numerous as the cultures themselves. We are all proud of our cuisines and quick to defend the uniqueness of our cuisine. This uniqueness lies in part because of certain ingredients can only be found in the locales of those cuisines. Therefore, I anticipate that the classification of cuisines will be defined by certain key ingredients.

I have 2 objectives for this project. The first is to build a classifier that will perform better than the sample benchmark of `0.19267` on Kaggle. The second is to better understand what defines each of the cuisines in the data.

### Metrics
<!-- In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
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
<!-- In this section, you will be expected to analyze the data you are using for the problem. This data can either be in the form of a dataset (or datasets), input data (or input files), or even an environment. The type of data should be thoroughly described and, if possible, have basic statistics and information presented (such as discussion of input features or defining characteristics about the input or environment). Any abnormalities or interesting qualities about the data that may need to be addressed have been identified (such as features that need to be transformed or the possibility of outliers). Questions to ask yourself when writing this section:
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

![Number of Recipes by Cuisine](https://github.com/xistz/mlnd-project/blob/master/plots/no_recipes_cuisine.png?raw=true)

From the figure, we can see that the distribution of cuisines in the train dataset is not uniform. This could lead to a problem when training the model, as the model could be bias towards cuisines that are more well represented in the train dataset.

There was no information about the id field in the dataset. Id probably refers to the id of the recipe in yummly's database. Since the goal of the project is to build a classifier to predict cuisine from ingredients, the id field will be dropped when preprocessing the data.

![Most common ingredients](https://github.com/xistz/mlnd-project/blob/master/plots/most_common_ingredients.png?raw=true)

There are 6,714 unique ingredients in this project. The most common ingredient in the train dataset is salt; it was used in 18,049 recipes. 1759 ingredients were used in only 1 recipe. Because of the large number of ingredients, we will need to consider classifiers such as decision trees which work well with large number of features.

Next, we will consider what the common ingredients in each cuisine.

![Most common ingredients by cuisine](https://github.com/xistz/mlnd-project/blob/master/plots/figure_3._5_most_common_ingredients_by_cuisine.png?raw=true)

As expected from the plot of common ingredients, salt is the most common ingredient in almost half the cuisines, only Asian cuisines such Chinese, Thai, Vietnamese, Japanese and Korean use another condiment in place of salt in their recipes. Because, salt and its related condiments (soy sauce, and fish sauce) is the most common ingredient in the train dataset, it could suggest that this train dataset is also predominantly made up of savoury recipes. It could imply that the resulting model would not work as well on sweet recipes 

Considering the most common ingredients by cuisine is not as informative as it could be. As seen from the plots above, there are still many common ingredients used across cuisines. As such, it would be better to consider the most common ingredients in each cuisine that are unique to that cuisine. This would give us a better idea of what kind of ingredients tend to define a cuisine.

![Most common ingredients unique to the cuisine](https://github.com/xistz/mlnd-project/blob/master/plots/figure_4._5_most_common_ingredients_unique_to_the_cuisine.png?raw=true)

From this plot, it gives us a better idea about how to distinguish cuisines by recipe. For example, one thing that distinguishes Japanese cuisine from others it their use of [dashi powder](https://en.wikipedia.org/wiki/Dashi). Dashi is the stock base from which many Japanese recipes are based upon. As such this would also suggest that a model built from considering ingredients unique to the cuisine would be far more successful than a model built from the most common ingredients. To emphasis unique ingredients, we will make use of Term Frequency-Inverse Document Frequency[(TF-IDF)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) when preprocessing our train dataset.

Next, we will consider the number of ingredients in each recipe.

![Number of Ingredients per Recipe](https://github.com/xistz/mlnd-project/blob/master/plots/ingredients_per_recipe.png?raw=true)

From the information above, the maximum number of ingredients for a recipe in the training data is 65 while the minimum number of ingredients for a recipe is 1. Although 65 is the most number of ingredients, it appears to be more of an outlier. Most recipes have about 10 ingredients. Although, the distribution seems to be skewed to the right, having a large number of ingredients in a recipe is not the norm (only 40 recipes have more than 30 ingedients).

Finally, we will consider the number of ingredients by cuisine.

![Number of Ingredients per Recipe by Cuisine](https://github.com/xistz/mlnd-project/blob/master/plots/number_of_ingredients_by_cuisine.png?raw=true)

From the boxplot, Moroccan cuisine has the highest average number of ingredients used per recipe. Most other cuisines have an average of 10 ingredients used, as expected by the histogram plot earlier. With the exception of Morrocan cuisines, it also appears that recipe length will not help in the classification of cuisines.

Our analysis of recipe length and common ingredients shows that we might not need to consider all ingredients when building our classifier. Using TF-IDF will help us identify which are the important ingredients.

### Algorithms and Techniques
<!-- In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_
-->
The algorithm I intend to use for this project is the Decision Tree. It was chosen because of 2 reasons:
- This is a multiclass classification problem.
- There are a large number of ingredients(features).

Besides its training speed and prediction accuracy, Decision Tree algorithms are also known to overfit. To prevent overfiting, we can use regularization (K-Fold cross validation) or an ensemble classifer such as Random Forest instead.

### Benchmark
<!-- In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_
-->

Since there are 20 cuisines in this project, one possible benchmark would be to achieve a categorization accuracy greater than random choice or `5%`. However, according to the public leaderboard for the Kaggle competition, the highest scoring model currently achieves a score of `0.82783`, while the sample benchmark achieves a score of `0.19267`. Since the sample benchmark is higher than random choice, the goal of this project will be to achieve a score greater or equal to the sample benchmark instead.

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
<!-- In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_ -->

In this project the following actions will be taken to prepare the data:
- Preprocess the text data in the columns, cuisines and ingredients.
- Apply TF-IDF to the processed ingredients to extract features.

Before extracting features from the train dataset for training, the train dataset has to be preprocessed. Text data, and categorical will need to be encoded numerically as machine learning algorithms are only able to work with numeric data. Cuisines and ingredients will be preprocessed differently.

As discovered earlier, there are 20 classes in cuisines, to convert them into numeric data, we will make use of Scikit-Learn's LabelEncoder. The LabelEncoder maps each cuisine to an integer. We can also use the LabelEncoder on predictions to obtain the predicted cuisines.

Also discovered earlier, there are 6.714 unique ingredients. Some of these ingredients are common across different cuisines. Such ingredients would not help when classifying cuisines and should be left out of the extracted features. 

Besides those ingredients, there were also some similar ingredients that were different only by type or amount. This project will put emphasis on the ingredient and not the quantity. As such, we will first preprocess ingredients by removing all units, numbers, and special characters to retain only pertinent informaion about the ingredient.

Thereafter, we will fit the recipes to the TF-IDF vectorizer and extract only the most relevant features of each recipe. TF-IDF is an algorithm that counts the text frequency and scales the frequency with the inverse document frequency. What this implies is, common ingredients such as salt which would appear many times across the document will be given a smaller count in the recipe because it is scaled with the inverse document frequency.

Finally, we will split the processed targets and features into new training and validation sets. The training targets, and features will be used to train the model while the validation targets, and features will be used to evaluate the performance of our baseline model.

### Implementation
<!-- In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_ -->
We will first implement a Decision Tree classifier to obtain a baseline performance. The Decision Tree classifier was chosen because it works well with both large number of features and multiple classes. For the baseline model, none of the parameters except `random_state` will be modified from their defaults.

After training, the Decision Tree classifier achieved a training accuracy of `1.00` while achieving a validation accuracy of `0.62`. This behavior was expected as Decision Trees are known to overfit on the training data.

In the next stage, some strategies will be explored for improving the classifier's performance.

### Refinement
<!-- In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_ -->

In this stage, we will make refinements to the classifier in 2 steps:
- Explore the performance of other algorithms and decide on the best algorithm for this project.
- Tune the parameters of the chosen algorithm.

For the first step, other algorithms will be tested against the performance of the Decision Tree. This is because the Decision Tree overfitted on the training targets, and features. This meant that the Decision Tree model failed to generalize well as shown its by poor validation accuracy. In addition, according to this [article](https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f), the Decision Tree algorithm may not be the best algorithm for this project. 

The other algorithms that will be tesed for this project are Logistic Regression, Linear SVC, Multinomial Naive Bayes, and Random Forest. All algorithms wlil be trained on the full set of targets, and features using K-Fold cross validation of 5 folds. The mean validation accuracy will be used to determine the best performing model.

After determining the which is the best algorithm for the project, grid search will be used to find the best parameters of the chosen algorithm. The mean training time, and mean validation accuracy will be used to determine the best parameters for the algorithm.

## IV. Results
<!-- _(approx. 2-3 pages)_ -->

### Model Evaluation and Validation
<!-- In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_ -->

The final model was built on a Linear SVC algorithm. The Linear SVC algorithm was chosen because it achieved the best mean validation accuracy among the various algorithms tested. The mean validation score achieved by the Linear SVC over 5 folds before tuning was `0.79`. The performance of the other algorithms are also shown in the plot below.

![Accuracy score of various algorithms](https://github.com/xistz/mlnd-project/blob/master/plots/algorithm_performance.png?raw=true)

Subsequently, tuning the model's `max_iter`, and `C` parameter only yielded marginal improvements for the mean validation score.

The final model is reasonable. It achieved a mean validation score of `0.79`. The high score also shows that the model generalizes well to unseen data.

The model is dependent on the features extracted by TF-IDF. As such, the model will be affected if the input space is changed. 

### Justification
<!-- In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_ -->

The final model achieved a mean categorization accuracy score of `0.79`. This is stronger than the benchmark of `0.19267`. The final weights of the model were also visualize to find out what ingredints were the most important when classifying each cuisine.

![Unique ingredients by cuisine from the model](https://github.com/xistz/mlnd-project/blob/master/plots/figure_8._unique_ingredients_by_cuisine_from_the_model.png?raw=true)

From the plot, most of the cuisines had expected features while some were surprising. For example, it was not surprising to see that Miso was the ingredient that had the highest weight in Japanese cuisine. After all, both Miso and Dashi form the foundation of almost all of Japanese cuisine since they are used to make basic stocks. Others such as the Szechwan ingredient in Chinese cuisine, the Irish ingredient in Irish cuisine were surprising because Szechwan is a region, not a ingredient while Irish would refer to the people or the culture. This could be a result of using Unigrams or n-grams of size 1 when using TF-IDF to extract features. This might have caused the tokenization of words such as "Szechwan peppercorns" into "Szechwan" and "peppercorns", which resulted in a loss of meaning.

## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
<!-- In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_ -->
![Model features word cloud](https://github.com/xistz/mlnd-project/blob/master/plots/figure_9._model_features_word_cloud.png?raw=true)

### Reflection
<!-- In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_ -->

This multiclass text classification was an interesting project. In the Nanodegree, we were taught some techniques for working with text in the data. However those techniques were only useful for mixed datasets where the majority of the dataset contained numeric data. Therfore, working on this project helped me explore more about working with text data.

One of the interesting learning points was the usage of TF-IDF. Using TF-IDF to transform the dataset is perhaps the most important part of the project. Using TF-IDF reduced the feature size by more than half. Without it, and using just a simple one-hot encoding, I am not sure if I could have achieved the same performance.

The most difficult part of the project was the data exploration. Even though I am passionate about food and cooking, exploring the data was still difficult because I was not sure what I should explore. Looking at other kernels on Kaggle gave me some inspiration, I was also grateful for the many post about questions I had on message boards like stackoverflow that helped me learn how to manipulate the data to highlight important characteristics. 

The final model largely achieved my expectations. The model performed better than the benchmark and achieved a categorization accuracy of `0.79` on the submission. However, I still do not think this model would be good enough for production. The biggest reason being the dataset that the model was trained on was biased. Many of the ingredients were savoury recipes, and the distribution of recipes among cuisines was not uniform.

### Improvement
<!-- In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_ -->

I believe the model can be further improved by undersampling the train dataset. By undersampling the train dataset, we can build a reduced train dataset that had a more uniform distribution of recipes among cuisines. Training a model on this dataset, will then ensure that the model is not bias towards well represented cuisines. There are 2 ways we could use undersampling to build a reduced train dataset. 

The first way is to undersample by cuisine alone. This will build a reduced train dataset that has a more uniform distribution of recipes among cuisines.

The second way goes a step further and undersamples by both cuisine and recipe type, sweet or savoury. The dataset will first need to be preprocessed to label each recipe as sweet or savoury based on their ingredients. Thereafter, undersample according to both recipe type and cuisine. A model trained on this dataset could be put into production as it would be less bias towards any particular cuisine or recipe type.

-----------
<!-- 
**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported? -->