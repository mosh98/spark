from functools import reduce

from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import IDF, Tokenizer, CountVectorizer, StopWordsRemover, StringIndexer
#from pyspark.shell import spark
from pyspark.sql import SQLContext, DataFrame,SparkSession


def unionAll(*dfs):
    return reduce(DataFrame.unionAll, dfs)


if __name__ == "__main__":
    # Create a SparkContext and an SQLContext
    sc = SparkContext(appName="Sentiment Classification")
    sqlContext = SQLContext(sc)


reviews = sc.wholeTextFiles('/Users/moslehmahamud/PycharmProjects/pythonProject/txt_sentoken/*/*')
print(type(reviews))

#appending 0 and 1
reviews_f = reviews.map(lambda row: (1.0 if 'pos' in row[0] else 0.0, row[1]))

dataset = reviews_f.toDF(['class_label', 'review'])


#============== Feature engineering ================

# Tokenize the review text column into a list of words
tokenizer = Tokenizer(inputCol='review', outputCol='words')
words_data = tokenizer.transform(dataset)

(train, dev, test) = words_data.randomSplit([.6, .2, .2], seed=42)

# TODO: Count the number of instances in, respectively, train, dev and test
# Print the counts to standard output
# [FIX ME!] Write code below
print(train.count())
print(dev.count())
print(test.count())

# TODO: Count the number of positive/negative instances in, respectively, train, dev and test
# Print the class distribution for each to standard output
# The class distribution should be specified as the % of positive examples
# [FIX ME!] Write code below
# from reducer import count
print('positive/negative instances')
print('Train instances')

train.groupBy('class_label').count().show()
result = train.filter(train.class_label == 1.0).collect()
Train_postive = len(result)
print("Train_postive ",Train_postive)
result = train.filter(train.class_label == 0.0).collect()
Train_negativ = len(result)
print("Train_negativ ",Train_negativ)

print("Train percentage ",(Train_postive/(Train_postive+Train_negativ))*100)

print('Dev instances')
dev.groupBy('class_label').count().show()
dev_result = dev.filter(dev.class_label == 1.0).collect()
dev_postive = len(dev_result)
print("dev_postive ",dev_postive)

dev_result = dev.filter(dev.class_label == 0.0).collect()
dev_negativ = len(dev_result)
print("dev_negativ ",dev_negativ)

print("Dev percentage ",(dev_postive/(dev_postive+dev_negativ))*100)


print('Test instances')
test.groupBy('class_label').count().show()
test_result = test.filter(test.class_label == 1.0).collect()
test_positive = len(test_result)
print("test_positive ",test_positive)
test_result = test.filter(test.class_label == 0.0).collect()
test_negativ = len(test_result)
print("test_positive ",test_negativ)
print("Test percentage ",(test_positive/(test_positive+test_negativ))*100)



# TODO: Create a stopword list containing the 100 most frequent tokens in the training data
# Hint: see below for how to convert a list of (word, frequency) tuples to a list of words
# stopwords = [frequency_tuple[0] for frequency_tuple in list_top100_tokens]
# [FIX ME!] Write code below
train_words = train.select("words")

freq = train_words.rdd.flatMap(lambda a: [(w, 1) for w in a.words]).reduceByKey(lambda a, b: a + b).sortBy(
    lambda x: x[1], ascending=False)

sizeOfWords = 0
lzt = []
for x in freq.collect():
    lzt.append(x[0])
print("length of all the vocabulary before stopwrods = ", len(lzt))

stopwords = freq.take(100)
print(stopwords)
newList = []
for x in stopwords:
    newList.append(x[0])

print(newList)
print(len(newList))

# TODO: Replace the [] in the stopWords parameter with the name of your created list
# [FIX ME!] Modify code below
remover = StopWordsRemover(inputCol='words', outputCol='words_filtered', stopWords=newList)


# Remove stopwords from all three subsets
train_filtered = remover.transform(train)
dev_filtered = remover.transform(dev)
test_filtered = remover.transform(test)

# Transform data to a bag of words representation
# Only include tokens that have a minimum document frequency of 2
cv = CountVectorizer(inputCol='words_filtered', outputCol='BoW', minDF=2.0)
cv_model = cv.fit(train_filtered)
train_data = cv_model.transform(train_filtered)
dev_data = cv_model.transform(dev_filtered)
test_data = cv_model.transform(test_filtered)

# TODO: Print the vocabulary size (to STDOUT) after filtering out stopwords and very rare tokens
# Hint: Look at the parameters of CountVectorizer
# [FIX ME!] Write code below
print("length of all the vocabulary after stopwrods = ", len(cv_model.vocabulary))



# Create a TF-IDF representation of the data
idf = IDF(inputCol='BoW', outputCol='TFIDF')
idf_model = idf.fit(train_data)
train_tfidf = idf_model.transform(train_data)
dev_tfidf = idf_model.transform(dev_data)
test_tfidf = idf_model.transform(test_data)

# ----- PART III: MODEL SELECTION -----

# Provide information about class labels: needed for model fitting
# Only needs to be defined once for all models (but included in all pipelines)
label_indexer = StringIndexer(inputCol='class_label', outputCol='label')

# Create an evaluator for binary classification
# Only needs to be created once, can be reused for all evaluation
evaluator = BinaryClassificationEvaluator()

# Train a decision tree with default parameters (including maxDepth=5)
dt_classifier_default = DecisionTreeClassifier(labelCol='label', featuresCol='TFIDF', maxDepth=5)

# Create an ML pipeline for the decision tree model
dt_pipeline_default = Pipeline(stages=[label_indexer, dt_classifier_default])

# Apply pipeline and train model
dt_model_default = dt_pipeline_default.fit(train_tfidf)

# Apply model on devlopment data
dt_predictions_default_dev = dt_model_default.transform(dev_tfidf)

# Evaluate model using the AUC metric
auc_dt_default_dev = evaluator.evaluate(dt_predictions_default_dev, {evaluator.metricName: 'areaUnderROC'} )

# Print result to standard output
print('Decision Tree, Default Parameters, Development Set, AUC: ' + str(auc_dt_default_dev))

# TODO: Check for signs of overfitting (by evaluating the model on the training set)
# [FIX ME!] Write code below
#TODO: Decision Tree
training_data = dt_model_default.transform(train_tfidf) #applying tdif
auc_dt_default_train = evaluator.evaluate(training_data, {evaluator.metricName: 'areaUnderROC'} ) #evaluating on test set
print('Decision Tree, Default Parameters, Training Set, AUC: ' + str(auc_dt_default_train))

# TODO: Tune the decision tree model by changing one of its hyperparameters
dt_classifier_maxDepthThree = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'TFIDF', maxDepth = 3)
dt_classifer_maxDepthFour = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'TFIDF', maxDepth = 4 )
dt_pipeline_three = Pipeline( stages=[label_indexer, dt_classifier_maxDepthThree] )
dt_pipeline_four = Pipeline( stages=[label_indexer, dt_classifer_maxDepthFour] )

dt_model_three = dt_pipeline_three.fit(train_tfidf)
dt_model_four = dt_pipeline_four.fit(train_tfidf)

dt_predictions_three_dev = dt_model_three.transform(dev_tfidf)
dt_predictions_four_dev = dt_model_four.transform(dev_tfidf)


auc_dt_three_dev = evaluator.evaluate( dt_predictions_three_dev, {evaluator.metricName: 'areaUnderROC'} )
auc_dt_four_dev = evaluator.evaluate( dt_predictions_four_dev, {evaluator.metricName: 'areaUnderROC'} )

print('Decision Tree, MaxDepth 3, Development Set, AUC: ' + str(auc_dt_three_dev) )
print('Decision Tree, MaxDepth 4, Development Set, AUC; ' + str(auc_dt_four_dev) )


#from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
# Build and evalute decision trees with the following maxDepth values: 3 and 4.
# [FIX ME!] Write code below
# Train a decision tree with default parameters (including maxDepth=5)

#dt_classifier_default = DecisionTreeClassifier(labelCol='label', featuresCol='TFIDF')

# Create an ML pipeline for the decision tree model
#dt_pipeline_default = Pipeline( stages=[label_indexer, dt_classifier_default] )

#paramGrid = ParamGridBuilder().addGrid(dt_classifier_default.maxDepth, [3,4]).build()

#crossval = CrossValidator(estimator=dt_pipeline_default,estimatorParamMaps=paramGrid,evaluator=evaluator, numFolds=3)

#cvModel = crossval.fit(train_tfidf)

#prediction = cvModel.transform(test_tfidf)
#selected = prediction.select("class_label", "review","probability", "prediction")
#for row in selected.collect():
    #print(row)


###############################################################################

# Train a random forest with default parameters (including numTrees=20)
rf_classifier_default = RandomForestClassifier(labelCol='label', featuresCol='TFIDF', numTrees=20)

# Create an ML pipeline for the random forest model
rf_pipeline_default = Pipeline(stages=[label_indexer, rf_classifier_default])

# Apply pipeline and train model
rf_model_default = rf_pipeline_default.fit(train_tfidf)

# Apply model on development data
rf_predictions_default_dev = rf_model_default.transform(dev_tfidf)

# Evaluate model using the AUC metric
auc_rf_default_dev = evaluator.evaluate(rf_predictions_default_dev, {evaluator.metricName: 'areaUnderROC'})

# Print result to standard output
print('Random Forest, Default Parameters, Development Set, AUC:' + str(auc_rf_default_dev))

# TODO: Check for signs of overfitting (by evaluating the model on the training set)
# [FIX ME!] Write code below
training_data_model = rf_model_default.transform(train_tfidf) #applying tdif

auc_dt_default_train = evaluator.evaluate(training_data_model, {evaluator.metricName: 'areaUnderROC'} ) #evaluating on test set

print('Random Forest, Default Parameters, Training Set, AUC: ' + str(auc_dt_default_train))

# TODO: Tune the random forest model by changing one of its hyperparameters
# Build and evalute (on the dev set) another random forest with the following numTrees value: 100.
# [FIX ME!] Write code below


# Train a random forest with default parameters (including numTrees=20)
rf_classifier_default = RandomForestClassifier(labelCol='label', featuresCol='TFIDF', numTrees=100)

# Create an ML pipeline for the random forest model
rf_pipeline_default = Pipeline(stages=[label_indexer, rf_classifier_default])

# Apply pipeline and train model
rf_model_default = rf_pipeline_default.fit(train_tfidf)

# Apply model on development data
rf_predictions_default_dev = rf_model_default.transform(dev_tfidf)

# Evaluate model using the AUC metric
auc_rf_default_dev = evaluator.evaluate(rf_predictions_default_dev, {evaluator.metricName: 'areaUnderROC'})

# Print result to standard output
print('Random Forest, Parameters 100, Development Set, AUC:' + str(auc_rf_default_dev))




# ----- PART IV: MODEL EVALUATION -----

# Create a new dataset combining the train and dev sets
traindev_tfidf = unionAll(train_tfidf, dev_tfidf)

# TODO: Evalute the best model on the test set
# Build a new model from the concatenation of the train and dev sets in order to better utilize the data
# [FIX ME!]




