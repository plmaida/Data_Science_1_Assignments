# Naive Bayes Assignment

The attached csv file has data on the fitness of pilots to fly given a number of categorical variables concerning the flight worthiness of the pilot. The target variable we want to predict is the second column “TestRes/Var1” where 1 indicates the the pilot is flight worthy, and 0 otherwise. Assume that the semantic meaning of the feature columns Var2 .. Var 6 are omitted for privacy and/or ethics reasons.

Implement a Naive Bayesian Classifier using python, that is, write python code to implement the classifier rather than use a library or module created by somebody else. DO NOT use a built-in python package such as sklearn for this (although you can use it to check your own implementation). Train your classifier on the provided dataset, and plot (using matplotlib) the ROC curve. 

 

B) 20 marks

Write a python script that reads in the csv data file and stores it in a database.

 

C) 20 marks

Write a script that opens up a database and uses the data to train and plot the ROC curve for the classifier you implemented in part a) . Reuse your code from parts A and B as much as possible by encapsulating the functionality in functions.

 

D) 10 marks 

What are some advantages of the Bayesian Classifier (for example: when would you use this instead of Deep Learning)? 

Should we have split our data into training and validation sets? Justify your answer.
