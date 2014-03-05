import mkl
mkl.set_num_threads(1)
#except ImportError:
#pass

import numpy as np
import scipy
from scipy.linalg import cholesky, solve_triangular
from sklearn.externals.joblib import Parallel, delayed
from multiprocessing import cpu_count
# Thanks to Sturla Molden for the Mahalanobis functions

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
###
print "scripts loaded"

import csv
import os
import grass.pygrass as pygrass

#os.system('g.region rast=global_eco3 zoom=global_eco3')
#os.system('r.mask global_eco3 maskc=1')

print "open global variables"
dem = pygrass.raster.RasterNumpy('dem')
dem.open('r')
dem22 = dem.flatten()

print "dem 1"
#np.isnan(dem22)

tree = pygrass.raster.RasterNumpy('tree')
tree.open('r')
tree22 = tree.flatten()

print "tree 2"
#np.isnan(tree22)

epr = pygrass.raster.RasterNumpy('epr2')
epr.open('r')
epr22 = epr.flatten()

print "epr 3"
#np.isnan(epr22)

pre = pygrass.raster.RasterNumpy('pre2')
pre.open('r')
pre22 = pre.flatten()

print "pre 4"
#np.isnan(pre22)

bio = pygrass.raster.RasterNumpy('bio2')
bio.open('r')
bio22 = bio.flatten()

print "bio 5"
#np.isnan(bio22)

slope = pygrass.raster.RasterNumpy('slope2')
slope.open('r')
slope22 = slope.flatten()

print "slope 6"
#np.isnan(slope22)

ndwi = pygrass.raster.RasterNumpy('ndwi')
ndwi.open('r')
ndwi22 = ndwi.flatten()

print "ndwi 7"
#np.isnan(ndwi22)

ndvi = pygrass.raster.RasterNumpy('ndvi')
ndvi.open('r')
ndvi22 = ndvi.flatten()

print "ndvi 8"
#np.isnan(ndvi22)

herb = pygrass.raster.RasterNumpy('herb')
herb.open('r')
herb22 = herb.flatten()

print "herb 9"
#np.isnan(herb22)

print "All global variables imported"

ind_global = np.column_stack((dem22,bio22,pre22,epr22,herb22,ndvi22,ndwi22,slope22,tree22))

print "open pa mask"
pa2 = pygrass.raster.RasterNumpy('eco_pa2') # map with ecoreg region but nulls outside park
pa2.open('r')
pa3 = pa2.flatten()

print "masking pa"

ind = np.where(pa3>0,(True),(False))

dem3 = dem22[ind]
print "pamap dem 1 ok"

bio3 = bio22[ind]
print "pamap bio 2 ok"

pre3 = pre22[ind]
print "pamap pre 3 ok"

epr3 = epr22[ind]
print "pamap epr 4 ok"

herb3 = herb22[ind]
print "pamap herb 5 ok"

#min_ndvi06_10_ave3 = 
#dem4 = dem3.flatten()

#print "pamap 6 ok"

ndvi3 = ndvi22[ind]
print "pamap ndvi 7 ok"

ndwi3 = ndwi22[ind]
print "pamap ndwi 8 ok"

slope3 = slope22[ind]
print "pamap slope 9 ok"

tree3 = tree22[ind]
print "pamap tree 10 ok"

ind_pa = np.column_stack((dem3,bio3,pre3,epr3,herb3,ndvi3,ndwi3,slope3,tree3))

print "2nd stack ok"

Ymean = np.mean(ind_pa,axis=0)
print "Ymean ok"
Ycov = np.cov(ind_pa,rowvar=False)
print "Ycov ok"

# example data!
#Y = numpy.random.randn(93476*9).reshape((93476,9))
#Ymean = numpy.mean(Y,axis=0)
#Ycov = numpy.cov(Y,rowvar=False)

mh = mahalanobis_distances(Ymean, Ycov, ind_global, parallel=True)
print "mh ok"

from scipy.stats import chisqprob
pmh = chisqprob(mh,9).reshape((1878,5046))
#### volver a poner dimensiones de cuando haces el mh! ####
print "pmh ok"
# quitar valores muy bajos!

out = pygrass.raster.RasterNumpy('results', mtype='FCELL')

new[:] = pmh

new.close()

new.open()  # re-open the closed map
new.close()  # then close
os.system('d.erase')
os.system('d.rast results')
os.system('r.out.gdal in=results output=results.tif')

print "results exported"
# export!

# out = pygrass.raster.RasterNumpy('eco_pa4') # eco_pa22 is a raster map which I created just to fill it in with new values

# out.open('w')
# #out.open('w')
# out.max()
# out.min()
# out = np.where(out >= 0,(pmh),(pmh))  # filling the map with the values from pmh. Also try "out[:] = pmh[:]"
# out.max()
# out.min()
# out.name='out' 
# out.close()

# calculate single HRI 0.5 value
hr1 = pmh >= 0.5
hr2 = sum(hr1)
hr3 = hr2/len(slope3)

with open('hri_results.csv', 'wb') as test_file:
	file_writer = csv.writer(test_file)
	# write headers, see the return array for the actual contents
	#file_writer.writerow(['wdpa_id', 'ap', 'wn', 'time1', 'time2'])
	#for elem in bp_results:
	file_writer.writerow(hri3) # inlcude wdpaid and ecoreg!

# from skimage import io

# io.imshow(pmh)
# io.show()


#print "exporting csv" # aqui exportariamos la imagen como tif con rasterio usando como modelo la creada para el PA!
#np.savetxt("pmh.csv", pmh, delimiter=";")

## otros metodos para calcular las distancias!
#mh = mahalanobis_distances(Ymean, Ycov, ind_global, parallel=False)
#mh = mahalanobis_distances(Ymean, Ycov, ind_global, parallel=True)
#mh = mahalanobis_distances_scipy(Ymean, Ycov, ind_global, parallel=True)
#mh = mahalanobis_distances_scipy(Ymean, Ycov, ind_global, parallel=False)

print "DONE"