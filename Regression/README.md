# Regression Analysis

What can you conclude from the performance or learning curves you generated?

Towards data science states that the learning rate is “a hyper-parameter that controls how much we are adjusting the weights of our network with respect the loss gradient.” A lower learning rate allows for the predictions to move along the gradient. A higher learning rate would therefore be subject to missing the optimal point. (https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10).  

The RMSE plots shown in the folder that the model created works at a low learning rate, as a larger learning rate causes greater error to occur. This is illustrated in the following graphs from Towards Data Science. The best learning rate for the developed model is 0.0001 and should be used when using the model. The model may have been more effective had a stochastic gradient been used, as it would have randomized the variables further leading to a better prediction in the model. 

![learningrate](https://user-images.githubusercontent.com/38801847/52601729-c977fa00-2e2d-11e9-8896-8edbc4e71d4e.png)
![learningrate2](https://user-images.githubusercontent.com/38801847/52601733-caa92700-2e2d-11e9-9bd5-f979ba964a05.png)

Images from (https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10)
