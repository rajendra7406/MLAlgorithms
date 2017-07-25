# Apriori

#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transactions = []
for i in range(0,7501):
    transactions.append( [str(dataset.values[i,j]) for j in range(0,20)] )
    
#training apriori dataset
from apyori import apriori
#likely to buy 3 products per week for all transactions. so, (3*7)/7500 = 0.0028, min_support
#rules will be choose will be true 20% of time, 0.2, min_confidence
#rules will contain atleast 2 min products, min_length
#the below values choosen are problem specific
rules = apriori(transactions, min_support = 0.003,min_confidence=0.2, min_lift=3, min_length=2)

#visualizing the results
results = list(rules)

myResults = [list(x) for x in results]
