import json
from sklearn import cluster, datasets
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import optparse
import sys
import random

filename = open("titanic.json")

features = json.load(filename)

dataset = []
target = []

companion = []
temprow = []
temptarg = []

print(features)
for i in range(0, len(features)):
    feat = [
        features[i]["Age"],
        features[i]["Embarked"],
        features[i]["Fare"],
        features[i]["Sex"],
        float(features[i]["SiblingsAndSpouses"])
        + float(features[i]["ParentsAndChildren"]),
        float(features[i]["Survived"]),
    ]  # include survived in the features
    temprow.append(feat)
    # temptarg.append(targ)
    for b in range(0, 5):
        if temprow[i][b] == "S":
            temprow[i][b] = 0
        elif temprow[i][b] == "Q":
            temprow[i][b] = 1
        elif temprow[i][b] == "C":
            temprow[i][b] = 2
        elif temprow[i][b] == "male":
            temprow[i][b] = 0
        elif temprow[i][b] == "female":
            temprow[i][b] = 1
            # elif(dataset[i][b] == ""):
            # 	dataset[i][b] = 1
    try:
        for b in range(0, 6):
            temprow[i][b] = float(temprow[i][b])

    except:
        continue
    dataset.append(temprow[i])
    target.append(temptarg)


def euclid(data, cent):
    distance = 0.0
    for i in range(0, len(data)):
        distance += (data[i] - cent[i]) ** 2
    eucdist = distance ** 0.5
    return eucdist


dummy = input("Press Enter to show non normalized data")
dataset = np.array(dataset)
print(dataset)
dummy = input("Press Enter to normalize")
data_norm = []
min_data = (
    []
)  # previously was a large number, now is the age in the first row of your table
max_data = []
for l in range(0, 6):
    min_data.append(dataset[0][l])
    max_data.append(dataset[0][l])

for a in range(0, len(dataset)):
    for l in range(0, 6):
        if dataset[a][l] < min_data[l]:
            min_data[l] = dataset[a][l]
        if dataset[a][l] > max_data[l]:
            max_data[l] = dataset[a][l]

# use a loop to normalize every value in the table by scaling it to the range 0..1
for a in range(0, len(dataset)):
    temp_norm = []
    for l in range(0, 6):
        norm = (dataset[a][l] - min_data[l]) / (max_data[l] - min_data[l])
        norm = np.float64(norm)
        temp_norm.append(norm)
    data_norm.append(temp_norm)

dataset = np.array(data_norm)

dummy = input("Press Enter to show dendrogram")
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(dataset, "ward", metric="euclidean")

plt.figure(1)
dendrogram(Z, leaf_rotation=50, leaf_font_size=8)
plt.title("Dendrogram from Scipy")
plt.xlabel("Sample")
plt.ylabel("Distance")
plt.savefig("Dendrogram_Q2.jpg")

plt.figure(2)
dendrogram(
    Z,
    truncate_mode="lastp",
    p=6,
    leaf_rotation=50,
    leaf_font_size=8,
    show_contracted=True,
)

plt.title("Dendrogram from Scipy")
plt.xlabel("Sample")
plt.ylabel("Distance")
plt.savefig("Dendrogram with 6 lines Q2.jpg")
plt.show()


dummy = input("Press Enter to complete clustering")
k = input(
    "Enter the amount of clusters based on the dendrogram (for this particular dataset use 6: "
)
k = np.int32(k)
# after seeing the dendrogram any number can be picked for the k means analysis.
# the one chosen based on the dendrogram for this data set should be 6

plt.figure(3)
centroid = []
point = []
# for iteration in range(0, 100):
for i in range(0, k):  # use three based on the dendrogram shown in part a
    centroid.append(dataset[random.randint(0, len(dataset))])
    point.append([])
    point[i].append(centroid[i])

it = [5, 10, 100]
for b in range(0, 101):
    cltlist = []
    for c in range(0, k):  # create a new list of lists with k amount of lists
        empty = []
        cltlist.append(empty)

    for a in range(0, len(dataset)):  # check for all points in the data set
        dist = []
        for center in range(
            0, k
        ):  # calculate a list of distances for each data point and centroid
            dist.append(euclid(dataset[a], centroid[center]))

        for clt in range(0, k):  # put it in the correct cluster
            if min(dist) == dist[clt]:
                cltlist[clt].append(dataset[a])

    for q in range(
        0, k
    ):  # find the new aveage of each cluster and replace the old centroid
        centroid[q] = sum(cltlist[q]) / len(cltlist[q])
    if b in it:  # put in for the requested iterations
        for clt in range(0, k):
            point[clt].append(centroid[clt])
            print("The size of the cluster at iteration", b, "is: ", len(cltlist[clt]))

    print("iteration", b)
for i in range(0, k):
    print("The final amount of points in cluster", i, "is: ", len(cltlist[i]))

plt.title("Moving Centroids")

plots = []

names = ["Age", "Embarked", "Fare", "Sex", "Companions", "Survived"]
print(np.shape(point))
print(type(point))
point = np.array(point)
print(type(point))
e = 1
i = 0

for i in range(0, 6):
    if i < 5:
        plt.subplot(2, 3, e)
        # shorthand for k = k +1#subplot creates panels
        # taking pieces of arrays (lists, tables) is called
        # "array slicing" -we've done this manually in the past
        # there's a shorthand syntax for array slicing:
        for d in range(0, k):  # http://structure.usc.edu/numarray/node26.html
            plt.scatter(point[d, :, i], point[d, :, 5])
        plt.xlabel(names[i])
        plt.ylabel(names[5])
        e += 1

plt.savefig("kmeans moving values for Q2.jpg")

c_survive = "blue"
c_dead = "red"
e = 1
colour = []

plt.figure(4)
plt.title("Clusters Survivors/Dead")
for i in range(5):
    # for j in range(5):
    if i < 5:
        plt.subplot(2, 3, e)  # subplot creates panels
        # taking pieces of arrays (lists, tables) is called
        # "array slicing" -we've done this manually in the past
        # there's a shorthand syntax for array slicing:
        # http://structure.usc.edu/numarray/node26.html
        plt.scatter(dataset[:, i], dataset[:, 5])
        plt.xlabel(names[i])
        plt.ylabel(names[5])
        e += 1  # shorthand for k = k +1
plt.savefig("kmeans clusters for Q2.jpg")
