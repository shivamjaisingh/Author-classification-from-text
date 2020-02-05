import pandas as pd
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import cross_val_score


df = pd.read_csv('data_file.csv')
df = shuffle(df)
df.reset_index(inplace=True, drop=True)

print("Different authors in the data-set ", df.Author.unique())

print("Different genre in the data-set ", df['Genre'].unique())

print("Total number of sentences ", len(df['Sentence']))

# fig = plt.figure(figsize=(8, 6))
# df.groupby('Author').Sentence.count().plot.bar(ylim=0)
# plt.show()

tf_idf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2),
                         stop_words='english')

features = tf_idf.fit_transform(df.Sentence).toarray()
labels = df.Author
print(features.shape)

# (1400, 3483) - Each Sentence is represented by 3483 features

X_train, X_test, y_train, y_test = train_test_split(df['Sentence'], df['Author'], random_state=0)

count_vect = CountVectorizer()

full_X = count_vect.fit_transform(df['Sentence'])

X_train_counts = count_vect.fit_transform(X_train)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

tfidf_transformer = TfidfTransformer()

full_X_tfidf = tfidf_transformer.fit_transform(full_X)


X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


clf_NB = MultinomialNB().fit(X_train_tfidf, y_train)

clf_SVM = SGDClassifier().fit(X_train_tfidf, y_train)

clf_DT = DecisionTreeClassifier().fit(X_train_tfidf, y_train)

clf_KNN = KNeighborsClassifier().fit(X_train_tfidf, y_train)


new_data_set = df[['Author', 'Sentence']]


print(new_data_set.shape)


# sen = "Vanity and pride are different things, though the words are often used synonymously. A person may be proud " \
#       "without being vain. Pride relates more to our opinion of ourselves, vanity to what we would have others think " \
#       "of us. "

# print(clf_KNN.predict(count_vect.transform([sen])))
# print(clf_NB.predict(count_vect.transform([sen])))
# print(clf_SVM.predict(count_vect.transform([sen])))
# print(clf_DT.predict(count_vect.transform([sen])))
# print(clf_RNC.predict(count_vect.transform([sen])))

X_test_counts = count_vect.transform(X_test)

predictions_NB = clf_NB.predict(X_test_counts)
print(accuracy_score(y_test, predictions_NB))
predictions_KNN = clf_KNN.predict(X_test_counts)
print(accuracy_score(y_test, predictions_KNN))
predictions_SVM = clf_SVM.predict(X_test_counts)
print(accuracy_score(y_test, predictions_SVM))
predictions_DT = clf_DT.predict(X_test_counts)
print(accuracy_score(y_test, predictions_DT))

print("\nTen-Fold Cross-Validation\n")

scores = cross_val_score(clf_NB, full_X_tfidf, df['Author'], cv=10)
print("Mean Score ", scores.mean())
scores = cross_val_score(clf_DT, full_X_tfidf, df['Author'], cv=10)
print("Mean Score ", scores.mean())
scores = cross_val_score(clf_KNN, full_X_tfidf, df['Author'], cv=10)
print("Mean Score ", scores.mean())
scores = cross_val_score(clf_SVM, full_X_tfidf, df['Author'], cv=10)
print("Mean Score ", scores.mean())


# kf = KFold(n_splits=10, shuffle=False, random_state=None)
# print(kf)
#
# scores = []
# cv = KFold(n_splits=10, random_state=None, shuffle=False)
# for train_index, test_index in cv.split(new_data_set):
#     print("Train Index: ", train_index, "\n")
#     print("Test Index: ", test_index)
#     X_train, X_test = new_data_set[train_index], new_data_set[test_index]
#     y_train, y_test = new_data_set[train_index], new_data_set[test_index]

#
# for train_index, test_index in kf.split(df):
#       print("Train:", train_index, "Validation:",test_index)

# scores = []
# cv = KFold(n_splits=10, random_state=None, shuffle=False)
# for train_index, test_index in cv.split(new_data_set):
#     print("Train Index: ", train_index, "\n")
#     print("Test Index: ", test_index)
#
#     X_train, X_test, y_train, y_test = new_data_set[train_index], new_data_set[test_index], new_data_set[train_index], new_data_set[test_index]
#     clf_NB.fit(X_train, y_train)
#     scores.append(clf_NB.score(X_test, y_test))

