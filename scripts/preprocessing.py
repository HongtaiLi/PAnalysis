import sys
import numpy as np
import pandas as pd
import pysal as ps
import urllib2
import random

import matplotlib.pyplot as plt



def  data_import(path):
	"""
	    import dbf data, and transfer to DataFrame
	"""

	dbf = ps.open(path)
	d = dict([(col,np.array(dbf.by_col(col))) for col in dbf.header])
	df = pd.DataFrame(d)
	return df

def get_subset(df):
	"""
	    Extract subset of features from original data
	    Sampling from the data
	"""
	list_of_feature =['PID','ADDRESS','ZIP','BED','HALF_BATH','BATH','YR_BUILT','FLOORS',
	'LAND_VAL1','BLDG_VAL1','BLDG_SQFT','LOT_SIZE','ASSMTVAL1']
	# print df
	# print "Describe of total features"
	# print df.describe()

	df = df[list_of_feature] # select subset
	df = df[df['ADDRESS'].map(len)>0]
	df = df[df['ASSMTVAL1']>0]
	df = df[df['BLDG_VAL1']>0]
	df = df[df['YR_BUILT']>0]
	df = df[df['BED']>0]
	df = df[df['LOT_SIZE']>1]

	print "Info about DataFrame:"
	print df

	print "Head:"
	print df.head()

	#Random sample from the data
	rows = random.sample(df.index,len(df)*30/100)
	df_30 = df.ix[rows]

	# print "Describe:"
	print df_30.describe()

	df_30.to_csv('./subset_30.csv')
	return df_30

def  explore(df):
	print 'ADDRESS'
	print df['ADDRESS']

def  urlRequest(url):
	response = urllib2.urlopen(url)
	l = response.read().split('\n')
	return l[0]

def  get_row_geocoding(row):
	"""
	    Encoding each row by street address
	"""
    street = row['ADDRESS']
	url = 'http://stree.cs.fiu.edu/street?street='+street+'&city=Miami&state=fl'
	lon_lat = urlRequest(url).split('\\s')
	lon = lon_lat[0]
	lat = lon_lat[1]
	return pd.Series({'lat'=lat,'lon'=lont})

    

def  geo_coding(path):
	 df = pd.read_csv(path)
	 df = df.merge(df.apply(get_row_geocoding,axis=1))
     print df
     print df.describe()


if __name__ == '__main__':
	#df = data_import("/Users/dingmia/Documents/data/environment-impact/Prop_Ptx.dbf")
	df = get_subset(df)
	geo_coding('./subset.csv')

	#explore(df)