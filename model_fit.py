
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import scale
from sklearn import metrics, tree, neighbors
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import StringIO
import pydot


#Read in data
dat = pd.read_csv("C:/Users/Job Lindsen/Desktop/Python Scripts/Py34/PosterProject/mov_dat_feat.csv", encoding = "ISO-8859-1")

dat = dat.dropna()                          #Drop na's
dat = dat.loc[~dat.duplicated('IMDBID'),:]  #Drop duplicates

dat = dat.reset_index()

dat = dat.drop(dat.columns[[0, 1, 4, 5, 6]], 1)

scaled_dat = pd.DataFrame(scale(dat.iloc[:,2:16]),
                          index=dat.index,
                          columns=dat.columns[2:16])

scaled_dat = pd.concat([dat.genre, dat.all_genres, scaled_dat], axis=1)

scaled_dat.genre.groupby(dat.genre).count()

# Get equal number (600) of observations in each class, drop fantasy because
# of low number of observations
genre_list = ['horror',
              'romance',
              'comedy',
              'sci_fi',
              'thriller',
              'action',
              'family']
             
i = 0
for genre in genre_list:
    dat_sub = scaled_dat[scaled_dat.genre == genre][0:600]
    if i == 0:
        dat_clean = dat_sub 
    else:
        dat_clean = dat_clean.append(dat_sub, ignore_index = True)
    i += 1
    
dat_clean.to_csv("C:/Users/Job Lindsen/Desktop/Python Scripts/Py34/PosterProject/dat_clean_2.csv")
dat_clean = pd.read_csv("C:/Users/Job Lindsen/Desktop/Python Scripts/Py34/PosterProject/dat_clean_2.csv")
dat_clean = dat_clean.drop(dat_clean.columns[[0]], 1)

##Get a dataframe g with 1 coding with genres a movie belongs to, and a 0
##for genres it does not belong to.
#genre_list = ['western',
#              'comedy',
#              'sci_fi',
#              'thriller',
#              'action',
#              'fantasy',
#              'family']
#              
#zeros = np.zeros(shape = (dat.shape[0], len(genre_list)))
#g = pd.DataFrame(data = zeros, columns = genre_list)
#ii = 0
#for i, r in dat.iterrows():
#    str_list = dat.all_genres[i].split(' | ')
#    str_list = [x.lower() for x in str_list]
#    jj = 0    
#    for genre in genre_list:
#        if genre in str_list:
#            g.iloc[ii,jj] = 1
#        else:
#            g.iloc[ii,jj] = 0
#        jj += 1 
#        
#    ii += 1

#data = scaled_dat.iloc[:,2:14]
data = dat_clean.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14,15]]


# PCA
pca = PCA(n_components = 14)
pca.fit(data)
print(pca.explained_variance_ratio_) 

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(range(1,15),2, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

pca = PCA(n_components = 11)
data_PCA = pca.fit_transform(data)

# k-means cluster analysis
SS = []
s_score = []
for i in range(2,16):
    clust = KMeans(n_clusters=i, n_init = 75).fit(data)
    labels = clust.labels_

    silh = metrics.silhouette_score(data_PCA, labels, metric='euclidean')
    s_score.append(silh)
    
    inert = clust.inertia_
    SS.append(inert)
    
#plt.plot(range(2,16),SS)
plt.plot(range(2,16),s_score)

# Cross-validate
X_train, X_test, y_train, y_test = train_test_split(data, dat_clean.genre,
                                                    test_size=0.25,
                                                    random_state=666)

train_ind = X_train.index.values
test_ind = X_test.index.values

g_train = g.iloc[train_ind,:]
g_test = g.iloc[test_ind,:]


clf = tree.DecisionTreeClassifier(criterion='gini', 
                                  max_depth=6, 
                                  min_samples_leaf=3)
####################
clf = RandomForestClassifier(criterion='gini', 
                                  max_depth = 6, 
                                  min_samples_leaf=3,
                                  n_estimators=50)
####################
clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini',
                                                max_depth=6, 
                                                min_samples_leaf=3),
                         n_estimators = 200,
                         learning_rate = 0.1)
####################
clf = neighbors.KNeighborsClassifier(100, weights='uniform')
clf = neighbors.KNeighborsClassifier(100, weights='distance')
####################
clf = GaussianNB()
##############################
t0 = time()
param_grid = {'C': [150, 500, 750, 1000],
              'gamma': [ 0.0005, 0.001, 0.05, .01], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

clf = SVC( C = 750, kernel='rbf', class_weight='auto', gamma = 0.001, probability=True)

########################
scores = cross_val_score(clf, data_PCA, dat_clean.genre, cv=10)
print(scores)
print(scores.mean())

predicted = cross_val_predict(clf, data_PCA, dat_clean.genre, cv=10)
print "Accuracy Test: {0:.3f}".format(metrics.accuracy_score(dat_clean.genre, predicted))
print
print "Classification report:"
print metrics.classification_report(dat_clean.genre, predicted)
print 
print "Confusion matrix:"
print metrics.confusion_matrix(dat_clean.genre, predicted)

#####################
data_tree = dat_clean.iloc[:,[3,4,5,6,7,8,9,10,13,14,15]]
clf = clf.fit(data_tree, dat_clean.genre)

# Visualize tree
dot_data = StringIO.StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=list(data_tree.columns.values))
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('dectree.pdf')


# Repeat on test set
y_test_pred = clf.predict(X_test)
print "Accuracy Test: {0:.3f}".format(metrics.accuracy_score(y_test, y_test_pred))
print
print "Classification report:"
print metrics.classification_report(y_test, y_test_pred)
print 
print "Confusion matrix:"
print metrics.confusion_matrix(y_test, y_test_pred)

# Measure performance
y_pred = clf.predict_proba(X_train)

# Repeat on test set
y_test_pred = clf.predict_proba(X_test)

tt = g_test.as_matrix()
pred = tt* y_test_pred

ss = np.sum(pred, axis=1)

sss = ss.mean()

print sss