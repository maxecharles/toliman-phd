import matplotlib.pyplot as plt
import alphacen as ac

obj = ac.TolimanJitter(detector_pixel_size=0.3)
source = obj.create_source()
psf = obj.optics.model(source)
plt.imshow(psf)
plt.show()
