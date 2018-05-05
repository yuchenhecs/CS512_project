import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import f1_score
import scipy.sparse
import csv

fname = 'data.1.tsv'
fname_label = 'labels.txt'
fname_paper = 'cleaned_test.txt'
# fnames = [ 'text_features_training', 'text_features_validation', 'text_features_test']
# fname_output = "text_feature_predictions.txt"
# fname_stat = "precision&recall.txt"


fnames = [ 'features0', 'features1']
fname_output = "text_hin_feature_predictions.txt"
fname_stat = "precision&recall_hin.txt"

titles = []
venues = []

# read venue list
with open(fname) as tsvin:
    tsvin = csv.reader(tsvin, delimiter='\t')
    for row in tsvin:
        titles.append(row[0])
        venues.append(row[2])

le = preprocessing.LabelEncoder()
le.fit(venues)
max_venue = len(venues)

# read training feature
training_features = scipy.sparse.load_npz(fnames[0]+'.npz')

# read training label
with open(fnames[0]) as f:
    data = f.read().split('\n')
    training_labels = list(map(int, data[:-1]))








# training
X = training_features
Y = np.array(training_labels)

clf = linear_model.SGDClassifier()
clf.fit(X, Y)





# validation
# read validation feature
validation_features = scipy.sparse.load_npz(fnames[1]+'.npz')

# read validation label
with open(fnames[1]) as f:
    data = f.read().split('\n')
    validation_labels = list(map(int, data[:-1]))
Y = np.array(validation_labels)
Y_pred = clf.predict(validation_features)

# evaluate validation
tp = [0] * max_venue
fp = [0] * max_venue
fn = [0] * max_venue

for i in range(len(Y)):
    if Y_pred[i] == Y[i]:
        tp[Y[i]] += 1
    else:
        fn[Y[i]] += 1
        fp[Y_pred[i]] += 1


venue_id = le.transform(le.classes_)

fout = open(fname_stat,'w')

for index, v in enumerate(le.classes_):
    fout.write(v + '\t')
    i = venue_id[index]
    if tp[i]+fp[i] > 0:
        fout.write(str(tp[i]/(tp[i]+fp[i])))
    else:
        fout.write(str(0.0))
    fout.write('\t')

    if tp[i]+fn[i] > 0:
        fout.write(str(tp[i]/(tp[i]+fn[i])))
    else:
        fout.write(str(0.0))
    fout.write('\n')

fout.write('f1_score_macro:' + str(f1_score(Y, Y_pred, average='macro')) + '\n')
fout.write('f1_score_micro:' + str(f1_score(Y, Y_pred, average='micro')) + '\n')




















exit(0)
# read test feature
test_features = scipy.sparse.load_npz(fnames[2]+'.npz')

# read test paper id
paperId=[]
with open(fname_paper) as f:
    for line in f:
        tokens = line.split('\t')
        paperId.append(tokens[0])

Y_pred = clf.predict(test_features)


# output predict
with open(fname_output,'w') as fout:
    for i in range(len(Y_pred)):
        fout.write(paperId[i]+'\t'+str(Y_pred[i])+'\n')



