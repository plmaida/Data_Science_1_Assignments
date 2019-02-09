This folder contains two different projects. 

The first is one a provided dataset to show clustering. Running python Kmeans_clustering_and_dendogram.py -f dataset_for_kmeans.csv.
The code was used to determine our own way of clustering using centroids 

Using the dendrogram function gives the following Dendrogram: 
![dendrogram_q1](https://user-images.githubusercontent.com/38801847/52526779-58e5a780-2c8c-11e9-9dce-8e5797c52474.jpg)

It should be noted that the Dendrogram does not necessarily provide the best view; however, the height increases significanlty.

We want to know if the written code was able to determine a similar result to the built-in scipy dendrogram function. 
A view of the scipy dendrogram function can be seen here and shows the number of items per cluster:
![dendrogram with 3 lines](https://user-images.githubusercontent.com/38801847/52526790-77e43980-2c8c-11e9-9a8c-30e07c25369e.jpg)

The resulting code using euclidean distance and centroids were able to determine a very similar result:
![kmeans](https://user-images.githubusercontent.com/38801847/52526935-6dc33a80-2c8e-11e9-8d35-ee8d3f4ef266.PNG)
