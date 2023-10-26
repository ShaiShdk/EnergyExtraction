
#%%
import os

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
csfont = {'fontname':'Times New Roman'}
hfont  = {'fontname':'Times New Roman'}
mpl.rcParams['text.usetex'] = True
mpl.rc('font', **{'family': 'serif'})

parent_folder  = '/Users/shaish/Library/CloudStorage/Dropbox/Science'
parent_folder += '/Projects/EnergyExtraction/codes_results_resub/GitHub_NX_codes/NX_results'

os.chdir(parent_folder)

#%%

# v2even = np.load('v2phi_even.npy')
# v2odd = np.load('v2phi_odd.npy')
# v2phi = np.empty((tuple([v2odd.shape[0]+v2even.shape[0]]) + v2odd.shape[1:]))

# v2phi[::2,...] = v2even
# v2phi[1::2,...] = v2odd

v2phi = np.load('v2phi_all.npy')

Gamma0 = 1

#%%

Vaf_series = np.around(np.linspace(1.5,5.,8),3)
a1_series = np.around(np.linspace(2,5,13),3)

go_series = np.around(np.linspace(.125,.275,4),3)
ge_series = np.around(np.linspace(.1,.3,5),3)
g0_series = np.around(np.linspace(.1,.3,9),3)

g_series = g0_series

#%%

for jj in range(v2phi.shape[0]):
    vg = v2phi[jj,:,:]
    lim_g0 = g_series[jj]/Gamma0
    vg[vg > lim_g0] = 0
    v2phi[jj,:,:] = vg

bool_veff = np.ones(v2phi.shape)
bool_veff[v2phi == 0] = 0
# bool_veff = np.swapaxes(bool_veff,0,1)

def explode(data):
    size = np.array(data.shape)
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e = data
    return data_e

n_voxels = np.zeros(bool_veff.shape, dtype=bool)
facecolors = np.where(n_voxels, '#FFD65DC0', '#1A88CCC0')
edgecolors = np.where(n_voxels, '#BFAB6E', '#FDF4A6')

fcolors_2 = explode(facecolors)
ecolors_2 = explode(edgecolors)

x, y, z = np.indices(np.array(bool_veff.shape)+1)

ax = plt.figure().add_subplot(projection='3d')
ax.view_init(elev=30, azim=135, roll=0)
ax.voxels(x, y, z, bool_veff,alpha=1, facecolors=fcolors_2, edgecolors=ecolors_2)#,extent=[0,1,0,1,0,1])
ax.set_aspect('equal')

# xmin = min(data[0])
# xmax = max(data[0])
# ymin = min(data[1])
# ymax = max(data[1])
# zmin = min(data[2])
# zmax = max(data[2])
# ax.set_xlim3d(xmin, xmax)
# ax.set_ylim3d(ymin, ymax)
# ax.set_zlim3d(zmin, zmax)

ax.set_xlabel('$g_0$')
# ax.set_ylabel('$g_0$')
# ax.set_zlabel('$g_0$')

# plt.savefig('v2phi_gamma1.png')
# plt.savefig('v2phi_gamma1.pdf')
plt.show()

#%%

ax = plt.axes(projection='3d')
ax.view_init(elev=30, azim=135, roll=0)
# ax.set_box_aspect((np.ptp(g_series), np.ptp(a1_series), np.ptp(Vaf_series)))
ax.set_box_aspect((2,3,1.5))

for gg in range(len(g_series)):
    for aa in range(len(a1_series)):
        for vv in range(len(Vaf_series)):
            if bool_veff[gg,aa,vv]:
                ax.scatter(gg, aa, vv, c=255*v2phi[gg,aa,vv]/np.max(v2phi), cmap='viridis', linewidth=0.5,s=50)

# plt.savefig('v2phi_GG1.png')
# plt.savefig('v2phi_GG1.pdf')

#%%

V0 = v2phi

from numpy import array
from scipy.interpolate import RegularGridInterpolator as rgi
# my_interpolating_function = rgi((x,y,z), V)
# Vi = my_interpolating_function(array([xi,yi,zi]).T)

xg, yg ,zg = np.meshgrid(g_series, a1_series, Vaf_series, indexing='ij', sparse=True)

Vint = rgi()

#%%

theta = 2 * np.pi * np.random.random(1000)
r = 6 * np.random.random(1000)
x = np.ravel(r * np.sin(theta))
y = np.ravel(r * np.cos(theta))

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

z = f(x, y)

ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z , cmap='viridis', linewidth=0.5)

#%%

# trpts = np.where(v2phi.ravel() != 0)[0]
# ax.scatter(g_series.ravel()[trpts],a1_series.ravel()[trpts],Vaf_series.ravel()[trpts])

#%%

def last_nonzero_index(arr):
    indices = np.where(arr != 0)[0]
    if len(indices) == 0:
        return None
    else:
        return indices[-1]

Zv = np.empty((len(g_series),len(a1_series)))
for gg in range(len(g_series)):
    for aa in range(len(a1_series)):
        Zv[gg,aa] = last_nonzero_index(bool_veff[gg,aa,:])

Zv[Zv != None] = np.min(Vaf_series) + .5*Zv[Zv!=None]

Z_none = np.zeros(Zv.shape)
# Z_none[Zv==None] = 0

#%%

# ax.plot_trisurf(g0grid, a1grid, Zv, linewidth=0.2, antialiased=True)
# ax.plot_surface(g0grid,a1grid,Z_none.T ,cmap=cm.bone,
#                        linewidth=1, antialiased=True)

from matplotlib import cm

g0grid, a1grid = np.meshgrid(g_series,a1_series)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.view_init(elev=30, azim=135, roll=0)
ax.set_box_aspect((2,3,1.5))

ax.plot_surface(g0grid,a1grid,Zv.T ,cmap=cm.RdBu,
                       linewidth=1, antialiased=True)

ax.set_xlabel('$g_0$')
ax.set_ylabel('$\alpha$')
ax.set_zlabel('$V_{AF}$')

# plt.savefig('v2phi_KK1_GG1.png')
# plt.savefig('v2phi_KK1_GG1.pdf')


#%%
fig, ax = plt.subplots()
ax.set_aspect('equal')
plt.imshow(np.flipud(Zv),interpolation='bicubic',cmap=mpl.colormaps['bone'],aspect=.05,extent=[np.min(g0_series),np.max(g0_series),np.min(a1_series),np.max(a1_series)])


#%%

X,Y,Z = np.meshgrid(g_series,a1_series,Vaf_series)

#%%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
# Define dimensions
# Nx, Ny, Nz = 100, 300, 500
# X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), -np.arange(Nz))

# g0_series = np.around(np.linspace(0,1,51),3)
# v2phi = np.zeros((len(g0_series),len(a1_series),len(Vaf_series)))
# for jj in range(len(g0_series)):
#     v2phi[jj,...] = v2phi_kk[0,...]

X,Y,Z = np.meshgrid(g0_series,a1_series,Vaf_series)
# Create fake data
# data = (((X+100)**2 + (Y-20)**2 + 2*Z)/1000+1)

data = v2phi
data = np.swapaxes(data , 0, 1)

kw = {
    'vmin': data.min(),
    'vmax': data.max(),
    'levels': np.around(np.linspace(data.min(), data.max(), 20), 2),    
    'cmap': mpl.colormaps['bone']
}

# Create a figure with 3D ax
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')

# Plot contour surfaces
_ = ax.contourf(
    X[:, :, 0], Y[:, :, 0], data[:, :, 0],
    zdir='z', offset=Z.min(), **kw
)
_ = ax.contourf(
    X[0, :, :], data[0, :, :], Z[0, :, :],
    zdir='y', offset=Y.min(), **kw
)
C = ax.contourf(
    data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
    zdir='x', offset=X.max(), **kw
)
# --

# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# Plot edges
edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
ax.plot([xmax, xmax], [ymin, ymax], [zmin, zmin], **edges_kw)
ax.plot([xmin, xmax], [ymin, ymin], [zmin, zmin], **edges_kw)
ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)

# Set labels and zticks
# ax.set(
#     xlabel='$g_0$',
#     ylabel='$\alpha_1$',
#     zlabel='$V_{AF}$',
#     #zticks=[0, -150, -300, -450],
# )

# Set zoom and angle view
ax.view_init(elev=30, azim=135, roll=0)
ax.set_box_aspect([2,3,2], zoom=0.9)

# Colorbar
fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='$V_{p}$')

plt.savefig('v2phi_KK1_GG1_contour.png')
plt.savefig('v2phi_KK1_GG1_contour.pdf')

# Show Figure
plt.show()
