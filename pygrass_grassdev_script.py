from grass.pygrass.raster import RasterNumpy

new = RasterNumpy('newraster', mtype='FCELL')

import numpy as np

new[:] = np.random.randn(*new.shape)

new.min()

new.max()

new[:3, :4]
new.close()


new.open()  # re-open the closed map

new[:3, :4]
norm = (new - new.min())/(new.max() - new.min())  # do something

norm.min()

norm.max()

norm.name  # the name is not set

norm.name = 'norm'  # set the name

norm.close()  # then close

norm.close('norm')  # give the name and close the raster,