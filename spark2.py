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


# TODO: Create a stopword list containing the 100 most frequent tokens in the training data
# Hint: see below for how to convert a list of (word, frequency) tuples to a list of words
# stopwords = [frequency_tuple[0] for frequency_tuple in list_top100_tokens]
# [FIX ME!] Write code below
train.select("review").show()

#train_word_counts = train.review.collect().flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

#train_sorted_frequencies = sorted(train_word_counts.collect(), key=lambda x: x[1], reverse=True)
#print(train_word_counts)

#x = train_sorted_frequencies[:100]
#stopwords = [frequency_tuple[0] for frequency_tuple in x]
#print(stopwords)


#Ignore this===========List all words===========TEST
from pyspark.sql.functions import split, explode
trainWordSplits = (train.select(split(train.review, '\s+').alias('split')))
singleWordDf = (trainWordSplits.select(explode(trainWordSplits.split).alias('word')))
shakeWordsDF = singleWordDf.where(singleWordDf.word != " ")
shakeWordsDF.show()

train_sorted_frequencies = shakeWordsDF.rdd.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1], ascending = False).take(100)
x = train_sorted_frequencies[:100]
stopwords = [frequency_tuple[0] for frequency_tuple in x]

newList = []

for x in stopwords:
    newList.append(x.word)



# TODO: Replace the [] in the stopWords parameter with the name of your created list
# [FIX ME!] Modify code below
remover = StopWordsRemover(inputCol='words', outputCol='words_filtered', stopWords=stopwords)

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
print("without stopwords")
print(train_data.count())
print(dev_data.count())
print(test_data.count())

"""
df.rdd \
  .filter(lambda x: x[1] == "france") \ # only french stations
  .map(lambda x: (x[0], x[2])) \ # select station & temp
  .mapValues(lambda x: (x, 1)) \ # generate count
  .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])) \ # calculate sum & count
  .mapValues(lambda x: x[0]/x[1]) \ # calculate average
  .sortBy(lambda x: x[1], ascending = False) \ # sort
  .take(100)
  """
#test_result = test.filter(test.class_label == 0.0).collect()
#lez = shakeWordsDF.filter(shakeWordsDF.word).collect()
#print(lez)


#Ignore this ===========List all words===========TEST
