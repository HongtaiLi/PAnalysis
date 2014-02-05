import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def  pre_process(path):
	 df = pd.read_excel(path,'Prop_Ptx', index_col=None, na_values=['NA'])

	 print df.head()
	 
if __name__ == '__main__':
	pre_process("/Users/dingmia/Documents/data/environment-impact/Prop_Ptx.xlsx")