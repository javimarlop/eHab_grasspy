from grass.pygrass.raster import RasterNumpy

out = pygrass.raster.RasterNumpy('results', mtype='FCELL')

new[:] = pmh

new.close()

new.open()  # re-open the closed map
new.close()  # then close
