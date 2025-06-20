#Baseline model, single KKN no feature selection, use to compare against scores of above algorithms 
import time
import sklearn.neighbors as neighbors
import sklearn.metrics as metrics
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split

X, y = datasets.load_breast_cancer(return_X_y=True)
X = X.tolist()
y = y.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
knn = neighbors.KNeighborsClassifier()
print("training started")
start_time = time.perf_counter()
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
f1 = metrics.f1_score(y_test, preds, average='macro')
end_time = time.perf_counter()
runtime = end_time - start_time
print(f"Runtime: {runtime:.4f} seconds")
print("F1 score: ", (1-f1))
