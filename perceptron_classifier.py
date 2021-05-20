import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris() 
# data: [[sepal.lenght, sepal.width, petal.lenght, petal.width]]
# target_names: ['setosa', 'versicolor', 'virginica']

X = iris.data[:, (2,3)] # (petal.lenght, petal.width)
y = (iris.target==1).astype(np.int) # is this iris a versicolor?

per_clf = Perceptron()
per_clf.fit(X,y)

print(per_clf.predict([[0, 7.5]])) # no
print(per_clf.predict([[20, 0.5]])) # yes