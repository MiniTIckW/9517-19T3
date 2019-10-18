from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix



#######  2.Load Digits dataset
digits = load_digits()
image = digits.images
data = digits.data
target = digits.target
# print(digits)
# print(dir(digits))
# print(digits.target_names)
# print(len(target),len(data),len(image))
# cv.imshow('0',image[19])

######## 3.Split into train/test
X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=0.25,stratify=target, random_state=0)

######## 4.Initialize the model

knn = KNeighborsClassifier(n_neighbors=5)
sgdc = SGDClassifier()
decision_tree = DecisionTreeClassifier()

######## 5. Fit the model to the training data
knn_model = knn.fit(X_train,y_train)
sgdc_model = sgdc.fit(X_train,y_train)
decision_tree_model = decision_tree.fit(X_train,y_train)


######### 6. Use the trained/fitted model to evaluate the test data.
predictions_knn = knn_model.predict(X_test)
predictions_sgdc = sgdc_model.predict(X_test)
predictions_decision_tree_model = decision_tree_model.predict(X_test)


######### 7. For each of the classifiers, evaluate the digit classification performance by calculating
#########    the accuracy.
accuracy_knn = metrics.accuracy_score(y_test, predictions_knn)
print('accuracy: %.3f%%' % (100 * accuracy_knn))
score_knn = knn_model.score(X_test, y_test)
print("score：%.3f%%" % (100 * score_knn))

accuracy_sgdc = metrics.accuracy_score(y_test, predictions_sgdc)
print('accuracy: %.3f%%' % (100 * accuracy_sgdc))
score_sgdc = sgdc_model.score(X_test, y_test)
print("score：%.3f%%" % (100 * score_sgdc))

accuracy_decision_tree = metrics.accuracy_score(y_test, predictions_decision_tree_model)
print('accuracy: %.3f%%' % (100 * accuracy_decision_tree))
score_decision_tree = decision_tree_model.score(X_test, y_test)
print("score：%.3f%%" % (100 * score_decision_tree))


recall_knn = metrics.recall_score(y_test,predictions_knn,average='macro')
print('recall: %.3f%%' % (100*recall_knn))
recall_sgdc = metrics.recall_score(y_test,predictions_sgdc,average='macro')
print('recall: %.3f%%' % (100*recall_sgdc))
recall_decision_tree = metrics.recall_score(y_test,predictions_decision_tree_model,average='macro')
print('recall: %.3f%%' % (100*recall_decision_tree))


matrix_knn = confusion_matrix(y_test, predictions_knn)
print(matrix_knn)
# matrix_sgdc = confusion_matrix(y_test, predictions_knn)
# print(matrix_sgdc)
# matrix_decision_tree = confusion_matrix(y_test, predictions_knn)
# print(matrix_decision_tree)




