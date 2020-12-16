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
train_word_counts = train.select("review").flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)
train_word_counts.show()
#train_word_counts = train.groupBy('review').count().show()
train_sorted_frequencies = sorted(train_word_counts, key=lambda x: x[1], reverse=True)

#x = train_sorted_frequencies[:100]
#stopwords = [frequency_tuple[0] for frequency_tuple in x]
#print(stopwords)







#Ignore this===========List all words===========TEST
#from pyspark.sql.functions import split, explode
#trainWordSplits = (train.select(split(train.review, '\s+').alias('split')))
#singleWordDf = (trainWordSplits.select(explode(trainWordSplits.split).alias('word')))
#shakeWordsDF = singleWordDf.where(singleWordDf.word != '')
#shakeWordsDF.show()

#test_result = test.filter(test.class_label == 0.0).collect()
#lez = shakeWordsDF.filter(shakeWordsDF.word).collect()
#print(lez)


#Ignore this ===========List all words===========TEST
