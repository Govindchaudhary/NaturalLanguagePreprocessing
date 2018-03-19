#importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset (we are importing it as tab seperated values instead of csv(comma seperated values)
#so as to avoid mistakes that can occur through comma present in the reviews )

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3) #quoting = 3 is to avoid the double quotes 

