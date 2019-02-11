#Question B - SQLite
import csv
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import sqlite3

def loadcsv(filename):
    #Load the csv file using an imputed name
    lines = csv.reader(open(filename, "r"))
    dataset = list(lines)
    for i in range(1, len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

filename_i = input("What is the name of the file you would like to show: (Flying_Fitness.csv by chance?) ")
dataset = loadcsv(filename_i)
data = sqlite3.connect("fly_fit.db")
d = data.cursor()
d.execute("CREATE TABLE flyfit(obs INTEGER, TestResVar1 INTEGER, Var2 INTEGER, Var3 INTEGER, Var4 INTEGER, Var5 INTEGER, Var6 INTEGER)")

for i in range(0, len(dataset)):
    d1 = tuple(dataset[i])
    #print(d1)
    d.execute("INSERT INTO flyfit VALUES(?,?,?,?,?,?,?)", d1)
d.execute("SELECT * from flyfit")
print(*d.fetchall(), sep='\n')
d.execute("DROP TABLE flyfit")
data.commit()


data.close()
