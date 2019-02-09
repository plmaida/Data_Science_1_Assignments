### Overview

This folder contains two different projects with the same goal. The goal being to attempt to cluster a dataset using euclidian distances and centroids to match the scipy dendrogram package. 

#### 2 Features
The first is on a provided dataset to show clustering using euclidean distance. Running python Kmeans_clustering_and_dendogram.py -f dataset_for_kmeans.csv. 

Using the dendrogram function gives the following Dendrogram: 
![dendrogram_q1](https://user-images.githubusercontent.com/38801847/52526779-58e5a780-2c8c-11e9-9dce-8e5797c52474.jpg)

It should be noted that the Dendrogram does not necessarily provide the best view; however, the height increases significanlty.

A view of the scipy dendrogram function can be seen here and shows the number of items per cluster:
![dendrogram with 3 lines](https://user-images.githubusercontent.com/38801847/52526790-77e43980-2c8c-11e9-9a8c-30e07c25369e.jpg)

The resulting code using euclidean distance and centroids were able to determine a very similar result:
![kmeans](https://user-images.githubusercontent.com/38801847/52526935-6dc33a80-2c8e-11e9-8d35-ee8d3f4ef266.PNG)

#### Titanic Dataset
A similar method was used on the second dataset (the titanic dataset). This file runs using python titanic_kmeans.py. The difference is mainly in the preprocessing of the data, as there are 6 features when compared to 2. 

The resulting dendrogram is shown here:
![dendrogram_q2](https://user-images.githubusercontent.com/38801847/52526968-1a9db780-2c8f-11e9-9861-0ca9c39a3563.jpg)

And shows the number of items per cluster:
![dendrogram with 6 lines q2](https://user-images.githubusercontent.com/38801847/52526972-31440e80-2c8f-11e9-8a18-3d31fdb662f2.jpg)

Using the euclidean distance provides us with the following result:
![kmeans2](https://user-images.githubusercontent.com/38801847/52526965-ffcb4300-2c8e-11e9-8ac3-31968f910414.PNG)

This is not quite as accurate as the one completed on the first dataset; though, this is expected as the dataset is more complex. 
