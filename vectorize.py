from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import scipy.sparse
import numpy as np
import csv
from sklearn.model_selection import train_test_split

fname = 'data.1.tsv'
fname_output = 'features'
titles = []
sources = []
venues = []

with open(fname) as tsvin:
    tsvin = csv.reader(tsvin, delimiter='\t')
    for row in tsvin:
        titles.append(row[0]+' '+row[1])
        venues.append(row[2])



le = preprocessing.LabelEncoder()
le.fit(venues)
vectorizer = CountVectorizer()

X_train, X_test, y_train, y_test = train_test_split(
     titles, venues)


# X_train, X_test, y_train, y_test = train_test_split(
#      titles, venues, test_size=0.1, random_state=42)

def vectorize_dense(texts, labels, mode):
    if mode == 0:
        X = vectorizer.fit_transform(texts)
    else:    
        X = vectorizer.transform(texts)

    scipy.sparse.save_npz(fname_output+str(mode)+'.npz', X)

    Y = le.transform(labels)
    np.savetxt(fname_output+str(mode), Y, fmt='%i')


vectorize_dense(X_train, y_train, 0)
vectorize_dense(X_test, y_test, 1)