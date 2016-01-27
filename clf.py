from kaggle import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

class RidgeFit(modelclass):
	clf = linear_model.RidgeClassifier(alpha = 0.001)
	cv_clf = clf

	def __str__(self):
		return "Ridge Regression"


class SVM(modelclass):
	clf = svm.SVC(kernel='rbf', gamma = 0.25)
	cv_clf = clf

	def __str__(self):
		return "SVM"

class NN(modelclass):
	clf =  MLPClassifier(algorithm='sgd', alpha=5, 
			hidden_layer_sizes=(10,10), random_state=0,
			learning_rate = 'adaptive', max_iter = 1000)
	cv_clf = clf

	def __str__(sefl):
		return "NeuralNetwork"

class Tree(modelclass):
	clf = tree.DecisionTreeClassifier()
	cv_clf = clf

	def __str__(self):
		return "Decision Tree"

class AdaBoost(modelclass):
	clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth =3), n_estimators=400)
	cv_clf = clf

	def __str__(self):
		return "AdaBoost"

class GradientBoost(modelclass):
	clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, 
			random_state=0)
	cv_clf = clf

	def __str__(self):
		return "GradientBoost"

class RandomForest(modelclass):
	clf = RandomForestClassifier(n_estimators = 200, max_features= 80, criterion = 'entropy')
	cv_clf = clf

	def __str__(self):
		return "RandomForest"

class KNN(modelclass):
	clf = KNeighborsClassifier(n_neighbors = 1, weights = 'distance')
	cv_clf = clf

	def __str__(self):
		return "KNN"

class SGD(modelclass):
	clf = SGDClassifier(alpha=0.001, n_iter = 400, penalty ='elasticnet')
	cv_clf = clf

	def __str__(self):
		return "SGDClassifier"

class logisticRegression(modelclass):
 	clf = linear_model.LogisticRegression(C=1)
 	cv_clf = clf

 	def __str__(self):
 		return "logisticRegression"






if __name__ == '__main__':


	# model = RidgeFit()
	# model = SVM()
	# model.train_data = preprocessing.scale(model.train_data)
	# model.test_data = preprocessing.scale(model.test_data)

	# model = NN()
	# model = Tree()
	# model = AdaBoost()
	# model = GradientBoost()
	# model = RandomForest()
	# model = KNN()
	# model = SGD()
	model = logisticRegression()
	model.Cross_Validation()
	# model.write_submission()



