from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv("dataset/personality_dataset.csv")

# Converting catagorical columns into binary numbers

dataset = dataset.replace("Extrovert", 1)
dataset = dataset.replace("Introvert", 0)
dataset = dataset.replace("Yes", 1)
dataset = dataset.replace("No", 0)
print(dataset.head())

model = LogisticRegression(solver='liblinear', random_state=0)