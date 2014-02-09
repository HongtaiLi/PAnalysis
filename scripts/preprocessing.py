import sys
import numpy as np
import pandas as pd
import pysal as ps
import urllib2
import random

#import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pylab as pl

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
	rows = random.sample(df.index,len(df)*10/100)
	df_sample = df.ix[rows]

	# print "Describe:"
	print df_sample.describe()

	df_sample.to_csv('./subset_10.csv')
	return df_sample

def  explore(df):
	print 'ADDRESS'
	print df['ADDRESS']

def  urlRequest(url):
	response = urllib2.urlopen(url)
	l = response.read().split('\n')
	print l
	if len(l)==3:
		return l[0]
	return 'X=0\tY=0'

def  get_row_geocoding(row):
	"""
	    Encoding each row by street address
	"""
	street = row['ADDRESS']
	#print street
	url = 'http://stree.cs.fiu.edu/street?street='+street+'&city=Miami&state=fl'
	#print url
	lon_lat = urlRequest(url).split('\t')
	lon = lon_lat[0][2:]
	lat = lon_lat[1][2:]
	#print lon,lat
	return pd.Series({'lat':lat,'lon':lon})
    

def  geo_coding(path):
	"""
	    Coding the address and get latitued and longitude
	"""
	df = pd.read_csv(path)
	print df
	df = df.merge(df.apply(get_row_geocoding,axis=1),left_index=True,right_index=True)
	df.to_csv('./subset_geo_10.csv')
	 #print df
	 #print df.describe()

def  dbscan(path):
	"""
        apply DBSCAN to cluster the properties
	"""
	
	list_of_feature =['ZIP','BED','HALF_BATH','BATH','YR_BUILT','FLOORS','LAND_VAL1','BLDG_VAL1','BLDG_SQFT','LOT_SIZE','ASSMTVAL1']
	df = pd.read_csv(path)
	df = df[list_of_feature]
	X = df.values
	####################PCA-reduced data for visualization###############
	pca = PCA(n_components=4) 
	pca.fit(X)
	print(pca.explained_variance_ratio_)

	X = pca.transform(X)
	X = StandardScaler().fit_transform(X)

	print X
	print X.shape
	# Compute DBSCAN
	db = DBSCAN(eps=0.2, min_samples=10).fit(X)
	core_samples = db.core_sample_indices_
	labels = db.labels_
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	print('Estimated number of clusters: %d' % n_clusters_)
	# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
	# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
	# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
	# print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
	# print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
	#print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

	unique_labels = set(labels)
	colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	for k, col in zip(unique_labels, colors):
		if k == -1:
			col = 'k'
			markersize = 6
		class_members = [index[0] for index in np.argwhere(labels == k)]
		cluster_core_samples = [index for index in core_samples if labels[index] == k]
	
		for index in class_members:
			x = X[index]
			if index in core_samples and k != -1:
			    markersize = 14
			else:
			    markersize = 6
			pl.plot(x[0], x[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=markersize)
	pl.title('Estimated number of clusters: %d' % n_clusters_)
	pl.show()
    
    	
if __name__ == '__main__':
	#df = data_import("/Users/dingmia/Documents/data/environment-impact/Prop_Ptx.dbf")
	#df = get_subset(df)
	#print "geo_coding"
	#geo_coding('./subset_10.csv')
	print "dbscan"
	dbscan('./subset_geo_10.csv')
	#explore(df)