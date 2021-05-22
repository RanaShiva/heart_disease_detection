import numpy as np
import pandas as pd
import warnings
import pickle
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('/home/shiva-rana/ml_in_healthcare/heart.csv')

from sklearn.model_selection import train_test_split

cols=['age', 'trestbps','chol','thalach','oldpeak']
y=df.target
X=df[cols]
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors=12)

knn_classifier.fit(X,y)
y_pred = knn_classifier.predict(X_test)
print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))

pickle.dump(knn_classifier,open('model.pkl','wb'))