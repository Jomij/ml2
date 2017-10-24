import numpy as np
import matplotlib.pyplot as plf
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv("train.csv").as_matrix()
print("Matrix data\n",data)

clf = DecisionTreeClassifier()

X_train = data[0:21000,1:]
Y_train = data[0:21000,0]

X_test = data[21000:,1:]
Y_test = data[21000:,0]

clf.fit(X_train,Y_train)
disp = X_test[0]
disp_shape = (28,28)
plf.imshow(255-disp,cmap="grey")
plf.show()

p=clf.predict([X_test[8]])
print ("Predict",p)
