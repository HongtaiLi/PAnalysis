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
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import MiniBatchKMeans, KMeans

import pylab as pl


from pandas.io import sql
import MySQLdb


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
	ori_df = pd.read_csv(path)
	df = ori_df[list_of_feature]
	X = df.values
	X = preprocessing.scale(X)

	print "head of sample"
	print df.head()
	####################PCA-reduced data for visualization###############
	pca = PCA(n_components=3) 
	pca.fit(X)
	X = pca.transform(X)
	
	print "PCA"
	print(pca.explained_variance_ratio_)
	print (pca.components_)

	#X = StandardScaler().fit_transform(X)
    

	print X
	print X.shape
	# Compute DBSCAN
	db = DBSCAN(eps=0.15, min_samples=60).fit(X)
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


	x_min, x_max = X[:, 0].min(), X[:, 0].max()
	y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
	print x_min,x_max,y_min,y_max

	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')

	unique_labels = set(labels)
	colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	for k, col in zip(unique_labels, colors):
		if k == -1:
			col = 'k'
			markersize = 2
		class_members = [index[0] for index in np.argwhere(labels == k)]
		cluster_core_samples = [index for index in core_samples if labels[index] == k]
	
		for index in class_members:
			x = X[index]
			if index in core_samples and k != -1:
			    markersize = 15
			else:
			    markersize = 2
			ax.scatter(x[0], x[1],x[2],marker='o', c=col, s=markersize)
	ax.set_title('Estimated number of clusters: %d' % n_clusters_)
	ax.set_xlim3d([-2,2])
	ax.set_ylim([-2,2])
	ax.set_zlim([-2,2])
	plt.show()
	print db.labels_
	return 	pd.concat([ori_df,pd.DataFrame({'labels':db.labels_})],axis=1)
     

def plot_cluster(reduced_data,k_means_labels,k_means_cluster_centers,n_clusters = 3):


	x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
	y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
	#z_min, z_max = reduced_data[:, 2].min(), reduced_data[:, 2].max()
	
	fig = plt.figure()
	#fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
	colors = ['#4EACC5', '#FF9C34', '#4E9A06']


	ax = fig.add_subplot(111)
	

	for k, col in zip(range(n_clusters), colors):
		my_members = k_means_labels == k
		print k
		#print 'k=%s'%k,reduced_data[my_members, 0], reduced_data[my_members, 1]
		cluster_center = k_means_cluster_centers[k]
		ax.scatter(reduced_data[my_members, 0], reduced_data[my_members, 1], c=col, marker='.',s=50)
		ax.scatter(cluster_center[0], cluster_center[1],c=col,marker='o', s=100)
	
	ax.set_title('Property Clustring')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	#ax.set_zlabel('Z Label')
	
	ax.set_xlim(-2,2)
	ax.set_ylim(-2,2)
	#ax.set_zlim3d(z_min,z_max)

	plt.show()


def kmeans(path,n_clusters):
	"""
	   kemans culstering algoritgm, apply pca to visualize
	"""

	list_of_feature =['ZIP','BED','HALF_BATH','BATH','YR_BUILT','FLOORS','LAND_VAL1','BLDG_VAL1','BLDG_SQFT','LOT_SIZE','ASSMTVAL1']
	df = pd.read_csv(path)
	df = df[list_of_feature]
	data = df.values
	data = preprocessing.scale(data)
	pca = PCA(n_components=2)
	pca.fit(data)
	reduced_data = pca.transform(data)
	print(pca.explained_variance_ratio_)
	print(pca.components_)



	k_means = KMeans(init='k-means++', n_clusters=n_clusters,n_init=10)
	k_means.fit(reduced_data)
	


	k_means_labels = k_means.labels_
	k_means_cluster_centers = k_means.cluster_centers_
	k_means_labels_unique = np.unique(k_means_labels)

	print k_means_labels,k_means_cluster_centers,k_means.inertia_

	plot_cluster(reduced_data,k_means_labels,k_means_cluster_centers,n_clusters)

def write_to_db(df):
	"""
		Write processed data to mysql 
	"""
	con = MySQLdb.connect(user='root',host='127.0.0.1',db='pa')
	sql.write_frame(df,con=con,name='pa',if_exists='replace',flavor='mysql')

if __name__ == '__main__':
	#df = data_import("/Users/dingmia/Documents/data/environment-impact/Prop_Ptx.dbf")
	#df = get_subset(df)
	#print "geo_coding"
	#geo_coding('./subset_10.csv')
	print "dbscan"
	df = dbscan('./subset_geo_10.csv')
	print df.head()

	#write_to_db(df)
	# print('K-means')
	# kmeans('./subset_geo_10.csv',2)
	df.to_csv('./subset_geo_10_label.csv')
	