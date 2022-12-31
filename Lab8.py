import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math



# function name: most_similar_cosine
# inputs: file_name- name of the csv file that holds classes
		# test_vec- vector that you are trying to classify
# output: a string that states what class the function is
# task: return the best class (AS A STRING) by using the cosine similarity metric
# assumptions: The csv file will not have headers!
			# Each row of the csv file is a data vector for each class
			# The last column will represent the classes. The remaining columns hold the data
def most_similar_cosine(file_name, test_vec):
	with open(file_name) as file:
		x = []
		y = []
		for i in file:
			i = i.strip().split(',')
			y.append(i[-1])
			i = [float(value) for value in i[:-1]]
			x.append(i)
		sim = []
		for row in x:
			inner = 0.0
			m1 = 0.0
			m2 = 0.0
			for i in range(len(test_vec)):
				inner += row[i] * test_vec[i]
				m1 += row[i] * row[i]
				m2 += test_vec[i] * test_vec[i]
			sim.append(abs(inner / ((m1 ** 0.5) * m2 ** 0.5)))
		mI = 0
		maxS = sim[0]
		for i in range(1, len(sim)):
			if maxS < sim[i]:
				maxS = sim[i]
				mI = i
		return y[mI]
# function name: most_similar_euclid
# inputs: file_name- name of the csv file that holds classes
# test_vec- vector that you are trying to classify
# output: a string that states what class the function is
# task: return the best class (AS A STRING) by using the euclidean distance metric
# assumptions: The csv file will not have headers!
			# Each row of the csv file is a data vector for each class
			# The last column will represent the classes. The remaining columns hold the data
def most_similar_euclid(file_name, test_vec):
	with open(file_name) as file:
		x = []
		y = []
		for i in file:
			i = i.strip().split(',')
			y.append(i[-1])
			i = [float(value) for value in i[:-1]]
			x.append(i)
		distances = []
		for row in x:
			dist = 0.0
			for i in range(len(test_vec)):
				dist += (row[i] - test_vec[i]) ** 2
			distances.append(dist ** 0.5)
		minI = 0
		minDist = distances[0]
		for i in range(1, len(distances)):
			if minDist > distances[i]:
				minDist = distances[i]
				minI = i
		return y[minI]

