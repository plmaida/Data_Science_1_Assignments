# Create a Naive_Bayes Classifier from a given csv data file and use skLearn to create the ROC.
import csv
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def loadcsv(filename):
    # Load the csv file using an imputed name
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(1, len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def naive_bayes(condition_data, data, condition_target, target, x):
    # count the probability of the value being in column when the target is either 0 or 1
    for j in range(0, data.shape[1]):
        freq = 0
        h = 0
        prob = 0

        for i in range(0, data.shape[0]):
            if data[i][j] == condition_data:
                h = h + 1
            if data[i][j] == condition_data and target[i] == condition_target:
                freq = freq + 1
        if h == 0:
            prob = 0
            print(
                "condition data",
                condition_data,
                "---",
                "condition target",
                condition_target,
                "---",
                freq,
                "----",
                h,
                "---",
                prob,
            )
        else:
            prob = freq / (h)
            print(
                "condition data",
                condition_data,
                "---",
                "condition target",
                condition_target,
                "---",
                freq,
                "----",
                h,
                "---",
                prob,
            )
        x.append(prob)
    return x


filename_i = input(
    "What is the name of the file you would like to show: (Flying_Fitness.csv by chance?) "
)
dataset = loadcsv(filename_i)
print(dataset)

dataset = np.array(dataset)

print(dataset.shape)

testY = []
testX = []
trainX = []
trainY = []

for a in range(1, dataset.shape[0]):
    newRow = []
    for i in range(2, 7):
        newRow.append(float(dataset[a][i]))
    if a % 5 == 0:
        testX.append(newRow)
        testY.append(float(dataset[a][1]))
    else:
        trainX.append(newRow)
        trainY.append(float(dataset[a][1]))

testX = np.array(testX)
testY = np.array(testY)
trainX = np.array(trainX)
trainY = np.array(trainY)

single_prob = []
nb_list_0 = []
nb_list_1 = []
nb_list = []
yes_no_prob = []

print(nb_list)

for i in range(0, 4):
    # In this scenario we know that the highest value is 3; however, to make this more robust we could ask
    # for the "highest value" OR we could (in the function) determine when each probability hits 1 for each variable
    # and we could then make it stop once ALL the variables have a total probability of 1.
    # For the purposes of this assignment we will leave it as is.
    for j in range(0, 2):
        if j == 0:
            nb_list_0 = naive_bayes(i, trainX, j, trainY, nb_list_0)
        if j == 1:
            nb_list_1 = naive_bayes(i, trainX, j, trainY, nb_list_1)

nb_list2 = []
nb_list2_0 = []
nb_list2_1 = []


for i in range(0, len(nb_list_0)):
    if nb_list_0[i] >= nb_list_1[i]:
        nb_list.append(nb_list_0[i])
    else:
        nb_list.append(nb_list_1[i])

k = 0
while k < len(nb_list):
    nb_list2.append(nb_list[k : k + 5])
    nb_list2_0.append(nb_list_0[k : k + 5])
    nb_list2_1.append(nb_list_1[k : k + 5])
    k += 5

print(nb_list2_1)
print(nb_list2_0)

# Now we have 8 tests and we want to determine the chance of it predicting the test result correctly
# Therefore we will look at the test data to determine if the prediction is correct
predict = []
pred = 1
pred2 = 1

print(testX)
for i in range(0, testX.shape[0]):
    pred = 1
    pred2 = 1
    prediction = 0
    for j in range(0, testX.shape[1]):
        for condition in range(0, 4):
            if testX[i][j] == condition:
                pred = pred * nb_list2_1[condition][j]
                pred2 = pred2 * nb_list2_0[condition][j]
    prediction = pred / (pred + pred2)
    predict.append(prediction)

print(predict)

# Create the ROC curve using skLearn

y = np.array(testY)
scores = np.array(predict)
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)

print(fpr)
print(tpr)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.plot(
    fpr, tpr, color="darkorange", marker="o", label="ROC curve (area = %0.2f)"
)
plt.show()
