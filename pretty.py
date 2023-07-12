import dLuxToliman as dlT
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
mpl.use('macosx')
plt.rcParams["image.origin"] = 'lower'

optics = dlT.TolimanOptics(wf_npixels=512, psf_npixels=5000, psf_oversample=60)
wavels = dlT.AlphaCen().wavelengths
psf = optics.propagate(wavels)
ge = dlT.gradient_energy.get_GE(psf)
dpi = 5000

# # plotting
# fig, ax = plt.subplots()
# ax.imshow(rotated_psf**.5, cmap='inferno')
# ax.set(xticks=[], yticks=[])
# plt.savefig('figs/clarissa/psf.pdf', bbox_inches='tight', dpi=350)
# plt.close()
#
fig, ax = plt.subplots()
ax.imshow(ge, cmap='gnuplot2')
ax.set(xticks=[], yticks=[])
plt.savefig('figs/clarissa/ge.png', bbox_inches='tight', dpi=dpi)
plt.close()

# # rortating stuff
# optics = dlT.TolimanOptics(wf_npixels=512, psf_npixels=8192, psf_oversample=70)
# psf = optics.propagate(wavels)
# ge = dlT.gradient_energy.get_GE(psf)
# edge_ge = scipy.ndimage.rotate(ge, 13)

# fig, ax = plt.subplots()
# ax.imshow(edge_ge, cmap='gnuplot2')
# ax.set(xticks=[], yticks=[])
# plt.savefig('figs/clarissa/edge_ge.png', bbox_inches='tight', dpi=dpi)
# plt.close()
#
# vertex_ge = scipy.ndimage.rotate(ge, 14-30)
# fig, ax = plt.subplots()
# ax.imshow(vertex_ge, cmap='gnuplot2')
# ax.set(xticks=[], yticks=[])
# plt.savefig('figs/clarissa/vertex_ge.png', bbox_inches='tight', dpi=dpi)
# plt.close()