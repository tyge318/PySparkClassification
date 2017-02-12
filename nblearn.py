'''
Date: 2/12/2017
Author: Franco Chen
This is an example code of using pyspark's NaiveBayes classifier to predict the polarity of users comment posts.
Details about the code please refer to the Readme file.
'''
import os
import re
import sys
import fnmatch
from pyspark.mllib.linalg import SparseVector	
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes
from pyspark import SparkContext

stopWords = set(["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"])

top_path = os.path.join(os.getcwd(), 'Data');
nd = '/negative_polarity/deceptive_from_MTurk'
nt = '/negative_polarity/truthful_from_Web'
pd = '/positive_polarity/deceptive_from_MTurk'
pt = '/positive_polarity/truthful_from_TripAdvisor'
classes = [nd, nt, pd, pt]		#0.25, 0.5, 0.75. 1
class_list = [[], [], [], []]	#tracking toekn occurrences for each class
word_to_id = {}					#map a word to an int id
feature_max = {}				#this is for normalization use
	
def preprocess(line):
	def stopWordRemoval(words):
		return [x for x in words if x not in stopWords]
	line = line.lower()		#lower-case
	tokens = re.findall(r"[\w']+", line)	#remove punctuation
	return stopWordRemoval(tokens)	#remove stop words

def getTrainingFeatures():		
	def add_dict(word, dictionary):		#occurrence tracking
		if word not in word_to_id:
			word_to_id[word] = len(word_to_id)
		word_id = word_to_id[word]
		if word_id in dictionary:
			dictionary[word_id] += 1
		else:
			dictionary[word_id] = 1
		feature_max[word_id] = max(feature_max.get(word_id, 0), dictionary[word_id])
	for i in xrange(len(classes)):
		class_dir = top_path + classes[i]
		current_class_list = class_list[i]
			
		subdirs = [ name for name in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, name)) ]
		for j in subdirs:
			current_dir = class_dir+"/"+j
			files = os.listdir(current_dir)
			#print listing		
			for k in files:
				features = {}
				if ".txt" not in k:
					continue
				with open(current_dir+"/"+k, "r") as c_file:
					for line in c_file:
						tokens = preprocess(line)
						for x in tokens:
							add_dict(x, features)
				current_class_list.append(features)

def getPredictionFeatures():
	def count_occurrence(word, dictionary):
		if word not in word_to_id:
			return;
		word_id = word_to_id[word]
		if word_id in dictionary:
			dictionary[word_id] += 1
		else:
			dictionary[word_id] = 1	
	goldStandard = {}
	with open('CorrectResult.txt', 'r') as file:
		for line in file:
			tokens = line.strip().split(' ')
			goldStandard[tokens[2]] = (tokens[:2])
	predict_files = list()
	for root, dirnames, filenames in os.walk(os.path.join(top_path, 'developing')):
	    for filename in fnmatch.filter(filenames, '*.txt'):
		predict_files.append(os.path.join(root, filename))
	predictSet = list()
	for k in predict_files:
		features = {}
		with open(k, 'r') as c_file:
			for line in c_file:
				tokens = preprocess(line)
				for x in tokens:
					count_occurrence(x, features)
		for key in features:
			temp = float(features[key])/float(feature_max[key])
			features[key] = 1 if temp >= 1 else temp
		fileName = k.replace(os.getcwd()+"/Data/developing/", "")
		predictSet.append([goldStandard[fileName], SparseVector(len(word_to_id), features)] )
	return predictSet

def training():
	sc = SparkContext("local", "Simple App")
	trainingSet = list()				
	for i in xrange(len(classes)):
		current_class_list = class_list[i]
		label = (i+1)/float(len(classes))
		for features in current_class_list:
			for key in features:	
				features[key] = float(features[key])/float(feature_max[key])
			trainingSet.append(LabeledPoint(label, SparseVector(len(word_to_id), features)) )

	# training model
	data = sc.parallelize(trainingSet)
	model = NaiveBayes.train(data, 1.0)
	return model
def runPrediction(model, predictSet):
	for predict in predictSet:
		p = model.predict(predict[1])
		ans = list()
		if p == 0.25:
			ans = ['deceptive', 'negative']
		elif p == 0.5:
			ans = ['truthful', 'negative']
		elif p == 0.75:
			ans = ['deceptive', 'positive']
		else:
			ans = ['truthful', 'positive']
		predict.append(ans)
	return predictSet

def Fscore(predict, i):
	tp = fp = tn = fn = 0
	for j in xrange(0, len(predict)):
		if i == 0:
			if predict[j][0][0] == 'truthful' and predict[j][2][0] == 'truthful':
				tp += 1
			elif predict[j][0][0] == 'truthful' and predict[j][2][0] == 'deceptive':
				fp += 1
			elif predict[j][0][0] == 'deceptive' and predict[j][2][0] == 'deceptive':
				tn += 1
			elif predict[j][0][0] == 'deceptive' and predict[j][2][0] == 'truthful':
				fn += 1
		if i == 1:
			if predict[j][0][1] == 'positive' and predict[j][2][1] == 'positive':
				tp += 1
			elif predict[j][0][1] == 'positive' and predict[j][2][1] == 'negative':
				fp += 1
			elif predict[j][0][1] == 'negative' and predict[j][2][1] == 'negative':
				tn += 1
			elif predict[j][0][1] == 'negative' and predict[j][2][1] == 'positive':
				fn += 1
	precision = float(tp)/float(tp+fp)
	recall = float(tp)/float(tp+fn)
	return (2*precision*recall)/(precision+recall)
def main():
	getTrainingFeatures()
	model = training()
	predictSet = getPredictionFeatures()
	predictSet = runPrediction(model, predictSet)
	for i in [0, 1]:
		print Fscore(predictSet, i)
if __name__ == '__main__':
	main()		