import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#1
train_imgs=pd.read_csv('MNIST/train_data.csv',  header=None)
train_labels=pd.read_csv('MNIST/train_label.csv',  header=None)
test_imgs=pd.read_csv('MNIST/test_data.csv',  header=None)
test_labels=pd.read_csv('MNIST/test_label.csv',  header=None)
print(train_imgs.iloc[1, :].values.shape)
img = train_imgs.iloc[1401, :].values.reshape(28, 28)
plt.imshow(img, cmap="Greys")
print("The below Image should be a ", train_labels.iloc[:, 0:1].values[1401])
# plt.show()

#2
from sklearn.neighbors import KNeighborsClassifier
trainData = []
testData = []
testLabels = test_labels.iloc[:, 0].values
trainLabels = train_labels.iloc[:, 0].values
for x in range(len(train_imgs.index)):
	trainData.append(train_imgs.iloc[x, :].values)
for x in range(len(test_imgs.index)):
	testData.append(test_imgs.iloc[x, :].values)
# print(len(train_imgs.index))
# neigh = KNeighborsClassifier(n_neighbors=4)
# neigh.fit(trainData, trainLabels)
clf = DecisionTreeClassifier()
clf.fit(trainData, trainLabels)
score = neigh.score(trainData, trainLabels)
print("train dp=none, accuracy=%.2f%%" % (score * 100))
score = neigh.score(testData, testLabels)
print("test dp=none, accuracy=%.2f%%" % (score * 100))

#3
# scoreTrain = []
# scoreTest = []
# for x in range(1,30,2):
# 	print(x)
# 	neigh = KNeighborsClassifier(n_neighbors=x)
# 	neigh.fit(trainData, trainLabels)
# 	scoreTrain.append(neigh.score(trainData, trainLabels))
# 	scoreTest.append(neigh.score(testData, testLabels))

# plt.plot([x for x in range(1,30,2)], scoreTrain, 'ro')
# plt.axis([0, 30, 0, 1])
# plt.xlabel('k')
# plt.ylabel('accuracy')
# plt.title('Train data accuracy')
# plt.grid()
# plt.show()
# plt.plot([x for x in range(1,30,2)], scoreTest, 'bo')
# plt.axis([0, 30, 0, 1])
# plt.xlabel('k')
# plt.ylabel('accuracy')
# plt.title('Test data accuracy')
# plt.grid()
# plt.show()
# plt.plot( 'x', 'Train', data=scoreTrain, marker='o', markerfacecolor='red', markersize=12, color='red', linewidth=4)
# plt.plot( 'x', 'Test', data=scoreTest, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
# plt.grid()
# plt.legend()


#5
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# print(test_labels)
# neighbors = neigh.kneighbors([test_imgs.iloc[1, :].values], return_distance=False)
# print(neighbors)
# c = 0
# for x in neighbors[0]:
# 	img = train_imgs.iloc[x, :].values.reshape(28, 28)
# 	plt.imshow(img, cmap="Greys")
# 	print("neighbor ", c+1,"The below Image should be a ", test_labels.iloc[:, 0].values[1])
# 	c += 1
# 	plt.show()


#6
# http://www.nickgillian.com/wiki/pmwiki.php/GRT/KNN