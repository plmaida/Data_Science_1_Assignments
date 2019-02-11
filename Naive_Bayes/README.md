# Naive Bayes Assignment

## Overview
The attached csv file has data on the fitness of pilots to fly given a number of categorical variables concerning the flight worthiness of the pilot. The target variable we want to predict is the second column “TestRes/Var1” where 1 indicates the the pilot is flight worthy, and 0 otherwise. Assume that the semantic meaning of the feature columns Var2 .. Var 6 are omitted for privacy and/or ethics reasons.

## Creating the Naive Bayesian Classifier
Implement a Naive Bayesian Classifier using python, that is, write python code to implement the classifier rather than use a library or module created by somebody else. DO NOT use a built-in python package such as sklearn for this (although you can use it to check your own implementation). Train your classifier on the provided dataset, and plot (using matplotlib) the ROC curve. 

### Answer
Using the Naive Bayes function created and sklearn's ROC function the following ROC curve is created:

![roc](https://user-images.githubusercontent.com/38801847/52601262-21156600-2e2c-11e9-9f08-25835c41bc1b.png)

## Creating a database
Write a python script that reads in the csv data file and stores it in a database.

### Answer
Database is put into flyfit.db as the first table flyfit

## Use the Database
Write a script that opens up a database and uses the data to train and plot the ROC curve for the classifier you implemented in part a) . Reuse your code from parts A and B as much as possible by encapsulating the functionality in functions.

### Answer
Combine the two python script. Because we want to do this task in one go we create a new table (flyfit2) in the same flyfit.db. This is done just to show that this script could run on its own. Using this we get the same curve:
![roc](https://user-images.githubusercontent.com/38801847/52601262-21156600-2e2c-11e9-9f08-25835c41bc1b.png)

As we can expect the ROC for question A and C are the same. This equivalency is because it uses the same training set and test set. 

## Concept questions
What are some advantages of the Bayesian Classifier (for example: when would you use this instead of Deep Learning)? 
Should we have split our data into training and validation sets? Justify your answer.

The advantages of the Naïve Bayes algorithm are that it is simple and easy to code. It is also fairly accurate. You also do not need as much training data due to the assumption for the Naïve Bayes classifier. Deep learning should be used when more information is needed and cross information is needed. 

http://blog.echen.me/2011/04/27/choosing-a-machine-learning-classifier/

A validation set should be used at it allows you to predict your results against a row of data that has not been seen by the program. Using unseen data for the test set when creating your predictions is ideal, as the program will be less likely to succumb to overfitting. In the code that I submitted I pull every 5th row out for the test set; however, the ideal would be to create the test set with a random set. The test set should be roughly 20/80 (meaning 20% for the test and 80% for the training set). 
