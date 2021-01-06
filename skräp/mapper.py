#!/usr/bin/env python

import sys

# input comes from STDIN (standard input)

""""
What is *MAP phase*?
The map or mapper’s job is to process the input data. 
Generally the input data is in the form of file or directory and is stored in the Hadoop file system (HDFS).
The input file is passed to the mapper function line by line. 
The mapper processes the data and creates several small chunks of data.

"""

for line in sys.stdin:
	# remove trailing spaces at end of each line
	word = line.strip()
	# print mapper output to STOUT (standard output): key <tab> value
	words = line.split()

	for word in words:
		print '%s\t%s' %(word,1)


