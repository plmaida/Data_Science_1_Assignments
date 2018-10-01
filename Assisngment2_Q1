#Assignment 2 - Question 1 on Boston Housing DataSet from Sklearn
#Code needs to be cleaned up. Commented out code was used to show work to professor.  

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

boston = datasets.load_boston()
#print boston.target
#print boston.data
#print np.shape(boston.target)

#dummy = raw_input("press the <ENTER> key to continue") 
#------------------------------define the regression model-------------------------------------------------
def regression(b,x):
	y = b[0]
	for i in range(1, 13):
		y = y + b[i]*x[i-1]
		#y = np.int32(y)
		#print "b", i, "x", i-1
	return y

#----------------------------------create holdout data-----------------------------------------------------
data = boston.data
target = boston.target

#print data
target = np.array(target)
#print target
#print np.shape(target)

#dummy = raw_input("press the <ENTER> key to continue") 

trainingX = []
trainingY = []

testX = []
testY = []
testcount = 0
testY1 = []

#-------------------------------------------------normalize the data--------------------------------------------------------------------------------
data_norm = []
target_norm = []
table = []
min_data = [] #previously was a large number, now is the age in the first row of your table
max_data = []
for l in range(0, 13):	
	min_data.append(data[0][l])
	max_data.append(data[0][l])

for a in range(0, len(data)):
	for l in range(0,13):
		if data[a][l] < min_data[l]:
			min_data[l] = data[a][l]
		if data[a][l] > max_data[l]:
			max_data[l] = data[a][l]
#dummy = raw_input("press the <ENTER> key to continue") 		
#print "The minimum values are: ", min_data
#print "The max values are: ", max_data
#print "Now able to normalize"
#dummy = raw_input("press the <ENTER> key to continue") 

#use a loop to normalize every value in the table by scaling it to the range 0..1
for a in range(0, len(data)):
	temp_norm = []
	for l in range(0,13):
		norm = (data[a][l] - min_data[l])/(max_data[l] - min_data[l])
		norm = np.float64(norm)
		temp_norm.append(norm)
	data_norm.append(temp_norm)
#print "Here is the normalized data", data_norm
#print np.shape(data_norm)

#-----------------------------------------------------------------Normalize the target--------------------------------------------------------------------------------

min_target = target[0]
max_target = target[0]

for a in range(0, len(data)):
	if target[a] < min_target:
		min_target = target[a] 
	if target[a] > max_target:
		max_target = target[a]
#dummy = raw_input("press the <ENTER> key to continue") 		
#print "The minimum values are: ", min_target
#print "The max values are: ", max_target
#print "Now able to normalize"
#dummy = raw_input("press the <ENTER> key to continue") 

#use a loop to normalize every value in the table by scaling it to the range 0..1
for a in range(0, len(target)):
	norm = (target[a] - min_target)/(max_target - min_target)
	target_norm.append(norm)
#plot the data

#print "this is the normalized feature variable"
#dummy = raw_input("press the <ENTER> key to continue") 
#print  data_norm
#print np.shape(data_norm)
#print "this is the normalized target variable"
#dummy = raw_input("press the <ENTER> key to continue") 
#print target_norm
#print x.shape()

#dummy = raw_input("press the <ENTER> key to continue") 
data_norm = np.array(data_norm)
target_norm = np.array(target_norm)

for i in range(0, data_norm.shape[0]):
	newRow = []	
	for a in range(0,13):	
		newRow.append(data_norm[i][a]) 	
	if(i % 10 == 0):
		testX.append(newRow)
		testY1.append(target_norm[i])
		testcount += 1
		#print "putting row", i, " ", newRow, "to test feature vector"
	else:
		trainingX.append(newRow)
		trainingY.append(target_norm[i])
		#print "putting row", i, " ", newRow, "to training feature vector"
	
print "Number of rows in the test vector is: ", testcount
#dummy = raw_input("press the <ENTER> key to continue")

def rmse(predictions, targets):
	#n = len(targets)
	r = 0	
	for i in range(0, len(predictions)):
		r = r + np.sqrt((predictions[i] - targets[i])**2)
	#calculate from h
	h = r/(len(predictions))
	return h

trainingX = np.array(trainingX)
trainingY = np.array(trainingY)
testX = np.array(testX)
testY.append(testY1)
testY = np.array(testY) 
testY = np.transpose(testY)
#print np.shape(trainingY)
#print testY
#print np.shape(testY)
#dummy = raw_input("press the <ENTER> key to continue")

#------------------------------learn the model and test it for different learning rates--------------------------------------------------------------
learning_rate = 0.00001
epoch = 0
epoch_max = 10
b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#print trainingX
#dummy = raw_input("press the <ENTER> key to continue")
#print trainingY
#dummy = raw_input("press the <ENTER> key to continue")
#print b
prediction = []
epoch_list = []
rmse_list = []
lr = 0
while(lr<6):
	while(epoch <= epoch_max):
		for i in range(0, len(trainingX)):
			error = regression(b, trainingX[i]) - trainingY[i]
			#print error
			#compute the learning rate for this data point
			b[0] = b[0] - learning_rate*error
			#print b[0]
			for r in range(1, 13):
				b[r] = b[r] - learning_rate*error*trainingX[i][r]
				#print "Epoch: ", epoch, "LearningRate: ", learning_rate, "Error: ", error  				
		predictions = []
		#print "coefficients calculated: ", b, "epoch is ", epoch
		#dummy = raw_input("press the <ENTER> key to continue")		
		#calculate the prediction of Y with testX (previously unseen data)		
		for j in range(0, len(testX)):
			#print testX[j]
			x = regression(b,testX[j])
			#prediction = np.int32(prediction)
			prediction.append(x)
		predictions.append(prediction)
		prediction = []
		predictions = np.array(predictions)
		predictions = np.transpose(predictions)
		#print "Here are the predictions", predictions
		#print np.shape(predictions)
		#print len(predictions)
		#dummy = raw_input("press the <ENTER> key to continue")
		#for i in range(0,len(testY)):
		#check the prediciton against the testY values to calculate RMSE
		rmse_var = rmse(predictions, testY)
			#rmse = np.int32(rmse)
		print "RMSE is: ", rmse_var
		epoch_list.append(epoch)
		rmse_list.append(rmse_var)		
		epoch += 1
	
		
	savename = "Question 1 "+str(learning_rate)
	plt.scatter(epoch_list, rmse_list)
	plt.title("Boston Housing Predictions for "+ str(epoch_max) +" epochs with a learning rate of" + str(learning_rate))
	plt.xlabel("Prediction Attempt")
	plt.ylabel("RMSE")
	plt.savefig(str(savename)+".png")
	plt.clf()		
	learning_rate = learning_rate*10
	print "The learning rate is: ", learning_rate	
	epoch = 0
	epoch_list = []
	rmse_list = []
	lr += 1


