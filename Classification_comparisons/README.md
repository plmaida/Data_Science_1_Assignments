
# READ ME

## Instructions provided
For tasks 1 to 10 you may use any combination of Python (including libraries) and Unix terminal
command pipelines.
In part 10 you must write one Python script that clearly labels where each part of the problem is
addressed using comments. 
Your data for this exercise are in the data file assess2_data.json
These data are a subset of data from a targeted marketing campaign. The original data have hundreds
of fields and hundreds of thousands of records; some other changes have been made as well. Each
data record describes a targeted consumer by various attributes.
Follow all directions and answer all questions. Be concise and precise in your answers:

Tasks 1 - 9 are preprocessing tasks and can be found in the preprocessing notebook

Task 10 is to compare different classifiers and can be found in the classification notebook

### Creating the Dendrogram
In order to create the Dendrogram we must first transform the data into useable values. We will first remove data that we do not want. This includes NAME, FIRSTDATE, LASTDATE and PEPSTRFL. We also change the RFA_2A values to numbers. We do all this in pandas. 
 
We can then create the dendrogram:
![image.png](attachment:image.png)

### Creating the Elbow Analysis
This is done in the same manner as above and is shown below:
![image.png](attachment:image.png)

### Sorting by Name
After looking through the data it is hard to differentiate people’s names, as there are the Jr. Sr. and III. Some people have middle initials as well. In order to solve this I found a nameparser online that can look at human names. I downloaded the HumanName parser and used this to get the last name of each person. Because of the way the HumanName parser works I had to return to the original dataframe that included missing values and then re-drop the indexes that we found with missing values.

Using iloc[0:10] and [20000:20010] we can see that the sort worked correctly. 
![image.png](attachment:image.png)


## Classification
The following are outputs of a run of the classification function

![Comparison](/viz/comparison_graphs.png)

The accuracy scores are high and seem to imply that the decision tree would be the best option; however, the Naïve Bayes classifier has a better area under the curve and is therefore the classifier that we would want to use. 

If we only had a budget to look at 10% of the data it would be more difficult to determine the best classifier. We would have less data and thereby a higher chance of having overfitting. We would also not have as much data for the algorithm to learn off of. We would still need to compute the ROC and AUC values to see which one performs the best, but being wary that the one that performs the best may be a misnomer, as we have less data. We should therefore use cross-validation techniques to determine the best classifier. If we only have the ability to run the model on 10% of the data this provides us with an opportunity to potentially remove some bias from the model, by having the number of responders and non-responders equal. This will allow the model to train on 50/50 data and will not bias the model. 
