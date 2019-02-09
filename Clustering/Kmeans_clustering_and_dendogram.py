# Setting up a dendogram for part a

from sklearn import cluster, datasets
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import optparse
import sys
import random

dataset = []

def euclid(data, cent):
	distance = 0.0
	for i in range(0, len(data)):
		distance += (data[i] - cent[i])**2
	eucdist = distance ** 0.5
	return eucdist

def processText(data):
	data = data.strip()
	print("DEBUG: ", data)
	fields = data.split(',')
	print("behold, an ordinary python list: ", fields)
	for field in fields:
		print("Here's the value of my field variable: ", field)
	fields = np.float64(fields)
	dataset.append(fields)

#create a new option parser
parser = optparse.OptionParser()
#add an option to look for the -f (question 3)
parser.add_option('-f', '--file', dest='fileName', help='file name to read from')

#get the options entered by the user at the terminal
(options, others) = parser.parse_args()

usingFile = False

#inspect the options entered by the user!
if options.fileName is None:
	print("DEBUG: the user did not enter the -f option")
else:
	print("DEBUG: the user entered the -f option")
	usingFile = True

if(usingFile == True):
	#attempt to open and read out of the file
	print("DEBUG: the file name entered was: ", options.fileName)
	file = open(options.fileName, "r") # "r" means we are opening the file for reading
	#write a loop that will read one line from the file at a time..
	for line in file:
		processText(line)


else:
	#read from standard input (shellexample3_demo.py)
	print("DEBUG: will read from standard input instead")
	for line in sys.stdin.readlines():
			processText(line)
			print("...")

dummy = input("Press Enter to show non normalized data")
dataset = np.array(dataset)
print(dataset)
dummy = input("Press Enter to normalize")
data_norm = []
min_data = [] #previously was a large number, now is the age in the first row of your table
max_data = []
for l in range(0, 2):
	min_data.append(dataset[0][l])
	max_data.append(dataset[0][l])

for a in range(0, len(dataset)):
	for l in range(0,2):
		if dataset[a][l] < min_data[l]:
			min_data[l] = dataset[a][l]
		if dataset[a][l] > max_data[l]:
			max_data[l] = dataset[a][l]


#use a loop to normalize every value in the table by scaling it to the range 0..1
for a in range(0, len(dataset)):
	temp_norm = []
	for l in range(0,2):
		norm = (dataset[a][l] - min_data[l])/(max_data[l] - min_data[l])
		norm = np.float64(norm)
		temp_norm.append(norm)
	data_norm.append(temp_norm)

dataset = np.array(data_norm)

print(dataset)

print(np.shape(dataset))

dummy = input("Press Enter to show dendrogram")
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(dataset, "ward", metric="euclidean")

plt.figure(1)
dendrogram(Z, leaf_rotation=50, leaf_font_size=8)
plt.title("Dendrogram from Scipy")
plt.xlabel("Sample")
plt.ylabel("Distance")
plt.savefig("Dendrogram_Q1.jpg")

plt.figure(2)
dendrogram(Z, truncate_mode='lastp', p=3, leaf_rotation=50, leaf_font_size=8, show_contracted = True)

plt.title("Dendrogram from Scipy")
plt.xlabel("Sample")
plt.ylabel("Distance")
plt.savefig("Dendrogram with 3 lines.jpg")
plt.show()


dummy = input("Press Enter to complete clustering")
k = input("Enter the amount of clusters based on the dendrogram (for this particular dataset use 3: ")
k = np.int32(k)

#after seeing the dendrogram any number can be picked for the k means analysis.
#the one chosen based on the dendrogram for this data set should be 3



plt.figure(3)
centroid = []
point = []
#for iteration in range(0, 100):
for i in range(0, k):	#use three based on the dendrogram shown in part a
	centroid.append(dataset[random.randint(0,len(dataset))])
	point.append([])
	point[i].append(centroid[i])
#point.append(centroid)
		#cluster.kmeans(n_clusters = k, max_iter=100, n_init=100, dataset)

print(centroid)
print(point)

it = [5, 10, 100]
for b in range(0, 101):
	cltlist = []
	for c in range(0,k):	#create a new list of lists with k amount of lists
		empty = []
		cltlist.append(empty)

	for a in range(0, len(dataset)): #check for all points in the data set
		dist = []
		for center in range(0, k): # calculate a list of distances for each data point and centroid
			dist.append(euclid(dataset[a], centroid[center]))
		for clt in range(0, k):	# put it in the correct cluster
			if (min(dist) == dist[clt]):
				cltlist[clt].append(dataset[a])

	for q in range(0, k): # find the new aveage of each cluster and replace the old centroid
		centroid[q] = sum(cltlist[q])/len(cltlist[q])
	if b in it: #put in for the requested iterations
		for clt in range(0, k):
			point[clt].append(centroid[clt])
			print("The size of the cluster at iteration", b, "is: ", len(cltlist[clt]))
		print(point)



	print("iteration", b)
for i in range(0, k):
	print("The final amount of points in cluster", i, "is: ", len(cltlist[i]))
plt.title("Moving Centroids")

for i in range (0, k):
	plt.scatter(point[i][:][0], point[i][:][1]) #plot the moving centroids. From the dataset it initializes quickly and therefore there are only a few data points
plt.xlabel("x values")
plt.ylabel("y values")
plt.savefig("kmeans_points.jpg")
