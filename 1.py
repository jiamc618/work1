import pandas

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
print(dataset.describe())
dataset.hist()
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
print(dataset.describe())
dataset.plot(x='sepal-length', y='sepal-width', kind='scatter')
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
print(dataset.describe())
dataset.plot(kind='kde')
import pandas

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
print(dataset.describe())
dataset.plot(kind='kde')
dataset.plot(kind='box', subplots=True, layout=(2, 2),
             sharex=False, sharey=False)
import pandas

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

from pandas.tools.plotting import radviz

radviz(dataset, 'class')

from pandas.tools.plotting import andrews_curves

andrews_curves(dataset, 'class')

from pandas.tools.plotting import parallel_coordinates

parallel_coordinates(dataset, 'class')
import pandas

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

from pandas.tools.plotting import scatter_matrix

scatter_matrix(dataset, alpha=0.2, figsize=(6, 6), diagonal='kde')
from sklearn.datasets import load_iris

hua = load_iris()
x = [n[0] for n in hua.data]
y = [n[1] for n in hua.data]
x = np.array(x).reshape(len(x), 1)
y = np.array(y).reshape(len(y), 1)

from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(x, y)
pre = clf.predict(x)
import matplotlib.pyplot as plt

plt.scatter(x, y, s=100)
plt.plot(x, pre, "r-", linewidth=4)
for idx, m in enumerate(x):
    plt.plot([m, m], [y[idx], pre[idx]], 'g-')
plt.show()
print(u"系数", clf.coef_)
print(u"截距", clf.intercept_)
print(np.mean(y - pre) ** 2)
print(clf.predict([[5.0]]))
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
clf = DecisionTreeClassifier()
clf.fit(iris.data, iris.target)
predicted = clf.predict(iris.data)

X = iris.data
L1 = [x[0] for x in X]

L2 = [x[1] for x in X]

import numpy as np
import matplotlib.pyplot as plt

plt.scatter(L1, L2, c=predicted, marker='x')
plt.title("DTC")
plt.show()

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris = load_iris()

train_data = np.concatenate((iris.data[0:40, :], iris.data[50:90, :], iris.data[100:140, :]), axis=0)
train_target = np.concatenate((iris.target[0:40], iris.target[50:90], iris.target[100:140]), axis=0)
test_data = np.concatenate((iris.data[40:50, :], iris.data[90:100, :], iris.data[140:150, :]), axis=0)
test_target = np.concatenate((iris.target[40:50], iris.target[90:100], iris.target[140:150]), axis=0)

clf = DecisionTreeClassifier()
clf.fit(train_data, train_target)
predict_target = clf.predict(test_data)
print(sum(predict_target == test_target))
from sklearn import metrics

print(metrics.classification_report(test_target, predict_target))
print(metrics.confusion_matrix(test_target, predict_target))
X = test_data
L1 = [n[0] for n in X]
# print L1
L2 = [n[1] for n in X]
# print L2
import numpy as np
import matplotlib.pyplot as plt

plt.scatter(L1, L2, c=predict_target, marker='x')  # cmap=plt.cm.Paired
plt.title("DecisionTreeClassifier")
plt.show()

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
clf = KMeans()
clf.fit(iris.data, iris.target)
predicted = clf.predict(iris.data)

X = iris.data
L1 = [x[0] for x in X]

L2 = [x[1] for x in X]
import numpy as np
import matplotlib.pyplot as plt

plt.scatter(L1, L2, c=predicted, marker='s', s=20, cmap=plt.cm.Paired)
plt.title("DTC")
plt.show() 