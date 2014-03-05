import mkl
mkl.set_num_threads(1)
#except ImportError:
#pass

import numpy as np
import scipy
from scipy.linalg import cholesky, solve_triangular
from sklearn.externals.joblib import Parallel, delayed
from multiprocessing import cpu_count
# Thanks to Sturla Molden for the piece of code

####
def _schedule(n, nproc):
	 """ guided scheduler """
	 start = 0
	 size = (n - start) // nproc
	 while size > 100:
		 yield slice(start,start+size)
		 start += size
		 size = (n - start) // nproc
	 yield slice(start,n+1)
	 return

def _mahalanobis_distances(m, L, X):
	 cX = X - m[np.newaxis,:]
	 tmp = solve_triangular(L, cX.T, lower=True).T
	 tmp **= 2
	 return np.sqrt(tmp.sum(axis=1))

def mahalanobis_distances(m, S, X, parallel=True):
	 L = cholesky(S, lower=True)
	 n = X.shape[0]
	 if parallel:
		 nproc = cpu_count()
		 res = (Parallel(n_jobs=-1)
				(delayed(_mahalanobis_distances)
				  (m, L, X[s,:])
					for s in _schedule(n,nproc)))
		 return np.hstack(res)
	 else:
		 return _mahalanobis_distances(m, L, X)


# scipy.spatial.distance.mahalanobis for comparison

from scipy.spatial import distance

def _mahalanobis_distances_scipy(m, SI, X):
	 n = X.shape[0]
	 mahal = np.zeros(n)
	 for i in xrange(X.shape[0]):
		 x = X[i,:]
		 mahal[i] = distance.mahalanobis(x,m,SI)
	 return mahal

def mahalanobis_distances_scipy(m, S, X, parallel=True):
	 SI = np.linalg.inv(S)
	 n = X.shape[0]
	 if parallel:
		 nproc = cpu_count()
		 res = (Parallel(n_jobs=-1)
				(delayed(_mahalanobis_distances_scipy)
				 (m, SI, X[s,:])
				   for s in _schedule(n,nproc)))
		 return np.hstack(res)
	 else:
		 return _mahalanobis_distances_scipy(m, SI, X)

print "scripts loaded"

#import numpy
import os
import rasterio


print "open global variables"
dem = rasterio.open('Variables/Mollweide4_clipped_srtm_ramp2_world_jf.tif.tif')
dem2 = dem.read_band(1)
dem22 = dem2.flatten()
print "dem 1"

tree = rasterio.open('Variables/Mollweide4_clipped_LatLon.2001.tree.nd.tif.tif')
tree2 = tree.read_band(1)
tree22 = tree2.flatten()
print "tree 2"

epr = rasterio.open('Variables/Mollweide4_clipped_wc_30s_epratio.tif.tif')
epr2 = epr.read_band(1)
epr22 = epr2.flatten()
print "epr 3"

pre = rasterio.open('Variables/Mollweide4_clipped_wc_30s_prec.tif.tif')
pre2 = pre.read_band(1)
pre22 = pre2.flatten()
print "pre 4"

bio = rasterio.open('Variables/Mollweide4_clipped_wc_30s_bio.tif.tif')
bio2 = bio.read_band(1)
bio22 = bio2.flatten()
print "bio 5"

slope = rasterio.open('Variables/Mollweide4_clipped_srtm_ramp2_slope_world_jf.tif.tif')
slope2 = slope.read_band(1)
slope22 = slope2.flatten()
print "slope 6"

ndwi = rasterio.open('Variables/Mollweide4_clipped_ndwi_max0610.tif.tif')
ndwi2 = ndwi.read_band(1)
ndwi22 = ndwi2.flatten()
print "ndwi 7"

ndvi = rasterio.open('Variables/Mollweide4_clipped_ndvi_max0610.tif.tif')
ndvi2 = ndvi.read_band(1)
ndvi22 = ndvi2.flatten()
print "ndvi 8"

herb = rasterio.open('Variables/Mollweide4_clipped_LatLon.2001.herb.nd.tif.tif')
herb2 = herb.read_band(1)
herb22 = herb2.flatten()
print "herb 9"

print "All global variables imported"

ind_global = np.column_stack((dem22,bio22,pre22,epr22,herb22,ndvi22,ndwi22,slope22,tree22))

# print "open ecoreg mask"
# ecoreg = rasterio.open('global_eco2.tif')
# ecoreg2 = ecoreg.read_band(1)
# ecoreg3 = ecoreg2.flatten()

# print "masking ecoreg"
# demx = dem2[ecoreg2==1]
# demx = demx.flatten()

# print "ecomap dem 1 ok"

# biox = bio2[ecoreg2==1]
# biox = biox.flatten()

# print "ecomap bio 2 ok"

# prex = pre2[ecoreg2==1]
# prex = prex.flatten()

# print "ecomap pre 3 ok"

# eprx = epr2[ecoreg2==1]
# eprx = eprx.flatten()

# print "ecomap epr 4 ok"

# herbx = herb2[ecoreg2==1]
# herbx = herbx.flatten()

# print "ecomap herb 5 ok"

# #min_ndvi06_10_ave3 = 
# #dem4 = dem3.flatten()

# #print "ecomap 6 ok"

# ndvix = ndvi2[ecoreg2==1]
# ndvix = ndvix.flatten()

# print "ecomap ndvi 7 ok"

# ndwix = ndwi2[ecoreg2==1]
# ndwix = ndwix.flatten()

# print "ecomap ndwi 8 ok"

# slopex = slope2[ecoreg2==1]
# slopex = slopex.flatten()

# print "ecomap slope 9 ok"

# treex = tree2[ecoreg2==1]
# treex = treex.flatten()

# print "ecomap tree 10 ok"

# ind_eco = numpy.column_stack((demx,biox,prex,eprx,herbx,ndvix,slopex,treex))

# print "1st stack ok"


print "open pa mask"
pa = rasterio.open('eco_pa2.tif')
pa2 = pa.read_band(1)

print "masking pa"
dem3 = dem2[pa2==1]
dem3 = dem3.flatten()

print "pamap dem 1 ok"

bio3 = bio2[pa2==1]
bio3 = bio3.flatten()

print "pamap bio 2 ok"

pre3 = pre2[pa2==1]
pre3 = pre3.flatten()

print "pamap pre 3 ok"

epr3 = epr2[pa2==1]
epr3 = epr3.flatten()

print "pamap epr 4 ok"

herb3 = herb2[pa2==1]
herb3 = herb3.flatten()

print "pamap herb 5 ok"

#min_ndvi06_10_ave3 = 
#dem4 = dem3.flatten()

#print "pamap 6 ok"

ndvi3 = ndvi2[pa2==1]
ndvi3 = ndvi3.flatten()

print "pamap ndvi 7 ok"

ndwi3 = ndwi2[pa2==1]
ndwi3 = ndwi3.flatten()

print "pamap ndwi 8 ok"

slope3 = slope2[pa2==1]
slope3 = slope3.flatten()

print "pamap slope 9 ok"

tree3 = tree2[pa2==1]
tree3 = tree3.flatten()

print "pamap tree 10 ok"

ind_pa = np.column_stack((dem3,bio3,pre3,epr3,herb3,ndvi3,ndwi3,slope3,tree3))

print "2nd stack ok"

Ymean = np.mean(ind_pa,axis=0)
print "Ymean ok"
Ycov = np.cov(ind_pa,rowvar=False)
print "Ycov ok"
# Add real park data!
#Y = numpy.random.randn(93476*9).reshape((93476,9))
#Ymean = numpy.mean(Y,axis=0)
#Ycov = numpy.cov(Y,rowvar=False)

mh = mahalanobis_distances(Ymean, Ycov, ind_global, parallel=True)
print "mh ok"
#mh = mahalanobis_distances(Ymean, Ycov, ind_eco, parallel=True).reshape(1884,5039)

from scipy.stats import chisqprob
pmh = chisqprob(mh,9).reshape((14661,36080))
#### volver a poner dimensiones de cuando haces el mh! ####
print "pmh ok"
# quitar valores muy bajos!

kwargs = bio.meta
kwargs.update(
	dtype=rasterio.float32,
	count=1,
	compress='lzw')

#with rasterio.drivers():
total = np.zeros(dem2.shape, dtype=rasterio.float32)
total = pmh
assert total.dtype == rasterio.float32
dst = rasterio.open('results.tif', 'w', **kwargs)
dst.write_band(1, total.astype(rasterio.float32))

#### bring to R
# meter los datos mejor en un "ecoregion2@data" raster objects como vector!

#print "exporting csv" # aqui exportariamos la imagen como tif con rasterio usando como modelo la creada para el PA!
#np.savetxt("pmh.csv", pmh, delimiter=";")

#R
#mh<-read.table('mh.csv',header=F,sep=';')
#dim(mh)
#library(raster)
#mh2<-raster(as.matrix(mh))
#plot(mh2)

## otros metodos para calcular las distancias!
#mh = mahalanobis_distances(Ymean, Ycov, ind_eco, parallel=False)
#mh = mahalanobis_distances(Ymean, Ycov, ind_eco, parallel=True)
#mh = mahalanobis_distances_scipy(Ymean, Ycov, ind_eco, parallel=True)
#mh = mahalanobis_distances_scipy(Ymean, Ycov, ind_eco, parallel=False)

print "DONE"