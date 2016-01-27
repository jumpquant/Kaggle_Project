from pylab import *
from sklearn import linear_model, cross_validation, svm
import operator, pickle, datetime
import numpy as np
from sklearn import preprocessing

def read_data():
	data = genfromtxt('training_data.txt', skip_header = 1, delimiter = '|')
	train_predict = data[:,-1]
	train_data = data[:, :-1]
	# train_data = preprocessing.scale(train_data)

	with open('train_data.dat', 'w') as file:
		pickle.dump(train_data, file)
	with open('train_predict.dat', 'w') as file:
		pickle.dump(train_predict.astype(int), file)

	test_data = genfromtxt('testing_data.txt', skip_header = 1, delimiter='|')
	# test_data = preprocessing.scale(test_data)

	with open('test_data.dat', 'w') as file:
		pickle.dump(test_data, file)

def score_func(prediction, correct):
	return float(sum(map(operator.eq, prediction, correct))) / len(correct)


class modelclass:
	def __init__(self, file = ''):
		with open('test_data.dat', 'r') as file:
			self.test_data = pickle.load(file)

		with open('train_data.dat', 'r') as file:
			self.train_data = pickle.load(file)

		with open('train_predict.dat', 'r') as file:
			self.train_predict = pickle.load(file)

		self.m = len(self.train_predict)
		self.M = len(self.test_data)

	def __str__(self):
		return 'LogisticRegression'

	def train(self, data = None, predict = None):
		if data == None:
			data = self.train_data
		if predict == None:
			predict = self.train_predict
		self.clf.fit(data,predict)

	def test(self, data = None):
		if data == None:
			data = self.test_data
		return self.clf.predict(data)

	def score(self, data):
		return self.clf.decision_function(data)

	def write_submission(self, fname = ''):
		now = datetime.datetime.now()
		header = 'Id,Prediction\n'
		self.train()
		result = self.test()
		if fname == '':
			fname = 'score_%s_%s%d_%s%s.txt' % (
				self.__str__(),
				now.strftime('%b'),
				now.day,
				now.strftime('%H'),
				now.strftime('%M')
				)
		with open(fname, 'w') as file:
			file.write(header)
			for i in range(len(result)):
				file.write('%d,%d\n' % (i+1, result[i]))
	

	def Cross_Validation(self):
		print ("Start cross validation...")
		scores = cross_validation.cross_val_score(
			self.cv_clf, self.train_data, self.train_predict, cv = 5)
		print ("Cross Validation Accuracy: %10f (+/- %3f)" % (scores.mean(), scores.std() * 2) )
		return scores


if __name__ == '__main__':
	read_data()
	# model = modelclass()
	# model.Cross_Validation()
	# model.write_submission()





