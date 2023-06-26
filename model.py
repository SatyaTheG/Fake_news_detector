from sklearn.linear_model import PassiveAggressiveClassifier
from data import messages,corpus, plot_confusion_matrix
from sklearn import metrics
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
filename = 'finalized_vectorizer.sav'
pickle.dump(tfidf_v, open(filename, 'wb'))
X=tfidf_v.fit_transform(corpus).toarray()

y=messages['Label']

# Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

linear_clf = PassiveAggressiveClassifier(n_iter_no_change=50)

linear_clf.fit(X_train, y_train)
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % (score*100))
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])