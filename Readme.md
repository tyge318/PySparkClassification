** Sentiment Classification Using NaiveBayes on PySpark
This code demos how to use PySpark and its built-in mllib to perform simple sentiment classification task.  
There are actually two dimensions of polarity: (a) positive or negative (b) truthful or deceptive.  
In this implementation, these two dimensions were combined to form a 4-class classification task.  
And the F-score measure for each dimension will be individually calculated in the end.  
The "Data" folder contains all the data we need:  
	- developing: contains the developing set to be used in testing our implementation.  
	- negative_polarity: contains training set of those posts labeled with negative.  
	- positive_polarity: contains training set of those posts labeled with positive.  
The two training set folders each contain two sub-folder for the second dimension (truthful or deceptive).  
The text file "CorrectResult.txt" stores the correct labels for posts in the developing set.  

*** Code Flow
I. Preprocessing
	Since the data are texts in natural langauge, we need to perform some preprocessing and convert them to label, features pairs.
	Here I only did stop word and punctuation removal. One can also applied packages such as NLTK to perform more sophiticated preprocessing like stemming to increase accuracy. I didn't do it here.  
	Punctuation removal is merely a regular expression matching task.  
	For the stop word removal task, I collected a list of common English stop words and store them in a set for O(1) lookup.
II. Get Features
	A simple feature to be used is unigram. That is, treating each single word as a dimension of the feature vector, and the word occurrence as its value. Of course one can use other features like bigram, trigram, TFIDF, etc.  
	The feature vectors for most NLP tasks are likely to be one with large dimensions, so it's better to store them in SparseVector data structure. The SparseVector constructor accepts python dictionary. Hence, here I first mapped the word (unigram) to an integer id, and then use this id and word occurance as the key, value pair for the feature vector. In gathering the word occurance, I also traced the max occurance for each word. This is to be used for later normalization.   
	For posts of training sets, labeling needs to be done besides feature extraction. Labeling in this task is simply converting where the file locates into its label. Because feature extraction needs to iterate through each post, I could just did this conversion in the mean time.  
	For posts of predict sets, just do feature extraction. The same word-id mapping used in extracting features of training set needs to be use here. And just skip unknown words (words never seen on traing set).
	After done feature extraction on both training set and predict set, use the max occurance vector to normalize all vectors.  
III. Training and Prediction
	This is where Spark RDD comes to play. First, creating a list of LabeledPoint with the label and vector in SparseVector form. Parallelize the list and feed it to NaiveBayes to train. The returned object is the model. Use this model to predict the predict set.  
IV. F-score Calculation
	Once we get the prediction result, perform F-score measure on truthful/deceptive and postive/negative dimension and output the results. Done.
	 