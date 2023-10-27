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

v2phi = np.load('v2phi_all.npy')
Gamma0 = 1

Vaf_series = np.around(np.linspace(1.5,5.,8),3)
a1_series = np.around(np.linspace(2,5,13),3)

go_series = np.around(np.linspace(.125,.275,4),3)
ge_series = np.around(np.linspace(.1,.3,5),3)
g0_series = np.around(np.linspace(.1,.3,9),3)

g_series = g0_series

for jj in range(v2phi.shape[0]):
    vg = v2phi[jj,:,:]
    lim_g0 = g_series[jj]/Gamma0
    vg[vg > lim_g0] = 0
    v2phi[jj,:,:] = vg

bool_veff = np.ones(v2phi.shape)
bool_veff[v2phi == 0] = 0

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

ax.set_xlabel('$g_0$')
plt.show()

ax = plt.axes(projection='3d')
ax.view_init(elev=30, azim=135, roll=0)
# ax.set_box_aspect((np.ptp(g_series), np.ptp(a1_series), np.ptp(Vaf_series)))
ax.set_box_aspect((2,3,1.5))

for gg in range(len(g_series)):
    for aa in range(len(a1_series)):
        for vv in range(len(Vaf_series)):
            if bool_veff[gg,aa,vv]:
                ax.scatter(gg, aa, vv, c=255*v2phi[gg,aa,vv]/np.max(v2phi), cmap='viridis', linewidth=0.5,s=50)
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

Vgrid, a1grid = np.meshgrid(Vaf_series,a1_series)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.view_init(elev=30, azim=135, roll=0)
# ax.set_box_aspect((2,3,1.5))

# V2D = Zmap
V2D = vb
ax.plot_surface(Vgrid,a1grid,V2D ,cmap=cm.RdBu, linewidth=1, antialiased=True)

ax.set_xlabel('$g_0$')
ax.set_ylabel('$\alpha$')
ax.set_zlabel('$V_{AF}$')

# plt.savefig('v2phi_KK1_GG1.png')
# plt.savefig('v2phi_KK1_GG1.pdf')


#%%

ax = plt.axes(projection='3d')
ax.view_init(elev=30, azim=215, roll=0)
# ax.set_box_aspect((np.ptp(g_series), np.ptp(a1_series), np.ptp(Vaf_series)))
ax.set_box_aspect((2,2,1))

vb = deepcopy(v_bnd_infty[0,0,...]/phi_bnd_infty[0,0,...])
# vb = deepcopy(v2phi_ratio[0,0,...])
V2D = vb

for aa in range(len(a1_series)):
    for vv in range(len(Vaf_series)):
        ax.scatter(aa, vv, V2D[aa,vv], c=V2D[aa,vv], cmap='coolwarm', linewidth=0.5,s=50)

plt.savefig('v2phi_bnd_GG1_surface.png')
plt.savefig('v2phi_bnd_GG1_surface.pdf')

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

Gamma0 = 1
g0_series = np.around(np.linspace(0,.25,26),3)
v2b = np.zeros((len(g0_series),len(a1_series),len(Vaf_series)))
v2phi = deepcopy(v2phi_kk[0,...])

# for jj in range(len(g0_series)):
#     v2phi[jj,...] = v2phi_kk[0,...]

for jj in range(len(g0_series)):
    v2b[jj,...] = v2phi

for jj in range(len(g0_series)):
    vg = v2b[jj,:,:]
    lim_g0 = g0_series[jj]/Gamma0
    vg[vg > lim_g0] = 0
    v2b[jj,:,:] = vg

X,Y,Z = np.meshgrid(g0_series,a1_series,Vaf_series)


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

data = v2b
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

plt.savefig('v2phi_bnd_KK0_GG0_contour.png')
plt.savefig('v2phi_bnd_KK0_GG0_contour.pdf')

# Show Figure
plt.show()

#%%

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# surf = ax.plot_surface(gg, alpha, Vaf, cmap=cm.viridis,
                    #    linewidth=0,alpha=1)#, antialiased=False)

ax.view_init(elev=30, azim=250)
ax.set_box_aspect((2,1,1))


#%%

g0_series = np.around(np.linspace(0,.5,26),3)
v2phi = np.zeros((len(g0_series),len(a1_series),len(Vaf_series)))
for jj in range(len(g0_series)):
    v2phi[jj,...] = v2phi_kk[0,...]

for jj in range(len(g0_series)):
    vg = v2phi[jj,:,:]
    lim_g0 = g0_series[jj]/Gamma0
    vg[vg > lim_g0] = 0
    v2phi[jj,:,:] = vg


# v2phi_kk = deepcopy(v2phi_ratio[0,...])
# np.save(v2phi_name,v2phi_ratio)

# v2phi = deepcopy(v2phi_kk)

#%%

# v2phi = deepcopy(v2phi_kk)

for jj in range(len(g0_series)):
    vg = v2phi[jj,:,:]
    lim_g0 = g0_series[jj]/Gamma0
    vg[vg > lim_g0] = 0
    v2phi[jj,:,:] = vg

bool_veff = np.ones(v2phi.shape)
bool_veff[v2phi == 0] = 0

bool_veff = np.swapaxes(bool_veff,0,1)

def explode(data):
    size = np.array(data.shape)
    data_e = np.zeros(size - 1, dtype=data.dtype)
    # data_e[::2, ::2, ::2] = data
    data_e = data
    return data_e

n_voxels = np.zeros(bool_veff.shape, dtype=bool)
facecolors = np.where(n_voxels, '#FFD65DC0', '#1A88CCC0')
edgecolors = np.where(n_voxels, '#BFAB6E', '#FDF4A6')

fcolors_2 = explode(facecolors)
ecolors_2 = explode(edgecolors)

x, y, z = np.indices(np.array(bool_veff.shape)+1)

ax = plt.figure().add_subplot(projection='3d')
ax.voxels(x, y, z, bool_veff,alpha=1, facecolors=fcolors_2, edgecolors=ecolors_2)
ax.set_aspect('equal')

plt.savefig('v2phi_gamma1.png')
plt.savefig('v2phi_gamma1.pdf')
plt.show()

#%%

Vtable = v2phi
fig = plt.figure(figsize=(10,7),dpi=100)
plt.imshow(np.flipud(Vtable),interpolation='bicubic',cmap=mpl.colormaps['bone'])
ax = plt.gca()
ax.set_autoscale_on(True)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel(r'$\alpha_1$',fontsize=30)
plt.xlabel(r'$\tau_K$',fontsize=30)

#%%

Gamma0 = 1
g0_series = np.around(np.linspace(0,.25,26),3)
# v2phi = deepcopy(v2phi_kk[0,...])

v2b = np.zeros((len(g0_series),len(a1_series),len(Vaf_series)))

Vmap_bnd = deepcopy(v_bnd_infty[0,0,...])

for jj in range(len(g0_series)):
    v2b[jj,...] = Vmap_bnd

for jj in range(len(g0_series)):
    vg = Vmap
    lim_g0 = g0_series[jj]/Gamma0
    vg[vg > lim_g0] = 0#np.max(vg)
    v2b[jj,:,:] = vg

X,Y,Z = np.meshgrid(g0_series,a1_series,Vaf_series)

data = v2b
data = np.swapaxes(data , 0, 1)

kw = {
    'vmin': data.min(),
    'vmax': data.max(),
    'levels': np.around(np.linspace(data.min(), data.max(), 20), 2),    
    'cmap': mpl.colormaps['YlGnBu']
}

# Create a figure with 3D ax
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')

# Plot contour back surfaces
_ = ax.contourf(
    X[:, :, 0], Y[:, :, 0], data[:, :, 0],
    zdir='z', offset=Z.min(), alpha=.5, **kw
)
_ = ax.contourf(
    X[0, :, :], data[0, :, :], Z[0, :, :],
    zdir='y', offset=Y.min(), alpha=.5, **kw
)
C = ax.contourf(
    data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
    zdir='x', offset=X.max(), alpha=.5, **kw
)

# Plot contour front surfaces
_ = ax.contourf(
    X[:, :, -1], Y[:, :, -1], data[:, :, -1],
    zdir='z', offset=Z.max(), alpha=.5, **kw
)
_ = ax.contourf(
    X[-1, :, :], data[-1, :, :], Z[-1, :, :],
    zdir='y', offset=Y.max(), alpha=.5, **kw
)
C = ax.contourf(
    data[:, 0, :], Y[:, 0, :], Z[:, 0, :],
    zdir='x', offset=X.min(), alpha=.5, **kw
)
# --

# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
ax.plot([xmax, xmax], [ymin, ymax], [zmin, zmin], **edges_kw)
ax.plot([xmin, xmax], [ymin, ymin], [zmin, zmin], **edges_kw)
ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
ax.view_init(elev=30, azim=135, roll=0)
# ax.set_title(f'$\kappa =$ {KK} ; $\gamma =$ {GG}')
ax.set_box_aspect([2,3,2], zoom=0.9)

fig.colorbar(C, ax=ax, fraction=0.02, pad=0.05, label='$V_{p}$')

# plt.savefig(f'v2phi_bnd_KK{KK}_GG{GG}_contour.png')
# plt.savefig(f'v2phi_bnd_KK{KK}_GG{GG}_contour.pdf')
plt.show()

#%%



Gamma0 = 1
g0_series = np.around(np.linspace(0,.25,26),3)
# v2phi = deepcopy(v2phi_kk[0,...])

Vmap = deepcopy(v_bnd_infty[0,0,...]/phi_bnd_infty[0,0,...])
Vmap = np.flipud(Vmap)

Vmap_bnd = deepcopy(v_bnd_infty[0,0,...])
Vmap_bnd = np.flipud(Vmap_bnd)

v2b = np.zeros((len(g0_series),len(a1_series),len(Vaf_series)))

for jj in range(len(g0_series)):
    v2b[jj,...] = Vmap#_bnd

for jj in range(len(g0_series)):
    vg = Vmap#v2b[jj,:,:]
    vg_bnd = Vmap#_bnd
    lim_g0 = g0_series[jj]/Gamma0
    vg[vg > lim_g0] = 0#np.max(vg)
    v2b[jj,:,:] = vg_bnd

#%%    

X,Y,Z = np.meshgrid(g0_series,a1_series,Vaf_series)

data = v2b
data = np.swapaxes(data , 0, 1)

kw = {
    'vmin': data.min(),
    'vmax': data.max(),
    'levels': np.around(np.linspace(data.min(), data.max(), 20), 2),    
    'cmap': mpl.colormaps['YlGnBu']
}

# Create a figure with 3D ax
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')

# Plot contour back surfaces
_ = ax.contourf(
    X[:, :, 0], Y[:, :, 0], data[:, :, 0],
    zdir='z', offset=Z.min(), alpha=.5, **kw
)
_ = ax.contourf(
    X[0, :, :], data[0, :, :], Z[0, :, :],
    zdir='y', offset=Y.min(), alpha=.5, **kw
)
C = ax.contourf(
    data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
    zdir='x', offset=X.max(), alpha=.5, **kw
)

# Plot contour front surfaces
_ = ax.contourf(
    X[:, :, -1], Y[:, :, -1], data[:, :, -1],
    zdir='z', offset=Z.max(), alpha=.5, **kw
)
_ = ax.contourf(
    X[-1, :, :], data[-1, :, :], Z[-1, :, :],
    zdir='y', offset=Y.max(), alpha=.5, **kw
)
C = ax.contourf(
    data[:, 0, :], Y[:, 0, :], Z[:, 0, :],
    zdir='x', offset=X.min(), alpha=.5, **kw
)
# --

# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
ax.plot([xmax, xmax], [ymin, ymax], [zmin, zmin], **edges_kw)
ax.plot([xmin, xmax], [ymin, ymin], [zmin, zmin], **edges_kw)
ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
ax.view_init(elev=30, azim=135, roll=0)
# ax.set_title(f'$\kappa =$ {KK} ; $\gamma =$ {GG}')
ax.set_box_aspect([2,3,2], zoom=0.9)

fig.colorbar(C, ax=ax, fraction=0.02, pad=0.05, label='$V_{p}$')

# plt.savefig(f'v2phi_bnd_KK{KK}_GG{GG}_contour.png')
# plt.savefig(f'v2phi_bnd_KK{KK}_GG{GG}_contour.pdf')
plt.show()


#%% ################### FINAL SHAPE #######################


Gamma0 = 1
g0_series = np.around(np.linspace(0,.25,51),3)
# v2phi = deepcopy(v2phi_kk[0,...])

Vmap = deepcopy(v_bnd_infty[0,0,...]/phi_bnd_infty[0,0,...])
Vmap = np.fliplr(Vmap)

gauss_sigma = 2
Vmap = gaussian_filter(Vmap, gauss_sigma)

# Vmap_bnd = deepcopy(v_bnd_infty[0,0,...])
# Vmap_bnd = np.flipud(Vmap_bnd)

# data = np.zeros((len(g0_series),len(a1_series),len(Vaf_series)))
v2b = np.zeros((len(g0_series),len(a1_series),len(Vaf_series)))

for jj in range(len(g0_series)):
    # data[jj,...] = Vmap_bnd
    v2b[jj,...] = Vmap

# v2b = (v2b - np.min(v2b))/(np.max(v2b) - np.min(v2b))

# for jj in range(len(g0_series)):
#     vg = vb[jj,:,:]
#     vg_bnd = data[jj,...]
#     lim_g0 = g0_series[jj]/Gamma0
#     vg_bnd[vg > lim_g0] = 0#np.max(vg)
#     data[jj,:,:] = vg_bnd.astype(bool)

for jj in range(len(g0_series)):
    vg = v2b[jj,:,:]
    lim_g0 = g0_series[jj]/Gamma0
    vg[vg > lim_g0] = 0#np.max(vg)
    v2b[jj,:,:] = vg.astype(bool)

# X,Y,Z = np.meshgrid(g0_series,a1_series,Vaf_series)

v2b = np.swapaxes(v2b,1,2)

X,Y,Z = np.meshgrid(g0_series,Vaf_series,a1_series)

data = v2b
data = np.swapaxes(data , 0, 1)

kw = {
    'vmin': data.min(),
    'vmax': data.max(),
    'levels': np.around(np.linspace(data.min(), data.max(), 20), 2),    
    # 'cmap': mpl.colormaps['bone']
    'cmap': mpl.colormaps['YlGnBu']
}

# Create a figure with 3D ax
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')

# Plot contour back surfaces
_ = ax.contourf(
    X[:, :, 0], Y[:, :, 0], data[:, :, 0],
    zdir='z', offset=Z.min(), alpha=.5, **kw
)
_ = ax.contourf(
    X[0, :, :], data[0, :, :], Z[0, :, :],
    zdir='y', offset=Y.min(), alpha=.5, **kw
)
C = ax.contourf(
    data[:, -1, :], Y[:, -1, :], Z[:, -1, :],
    zdir='x', offset=X.max(), alpha=.5, **kw
)

# Plot contour front surfaces
_ = ax.contourf(
    X[:, :, -1], Y[:, :, -1], data[:, :, -1],
    zdir='z', offset=Z.max(), alpha=.5, **kw
)
_ = ax.contourf(
    X[-1, :, :], data[-1, :, :], Z[-1, :, :],
    zdir='y', offset=Y.max(), alpha=.5, **kw
)
C = ax.contourf(
    data[:, 0, :], Y[:, 0, :], Z[:, 0, :],
    zdir='x', offset=X.min(), alpha=.5, **kw
)
# --

# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
ax.plot([xmax, xmax], [ymin, ymax], [zmin, zmin], **edges_kw)
ax.plot([xmin, xmax], [ymin, ymin], [zmin, zmin], **edges_kw)
ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
ax.view_init(elev=30, azim=150, roll=0)
# ax.view_init(elev=30, azim=225, roll=0)
# ax.set_title(f'$\kappa =$ {KK} ; $\gamma =$ {GG}')
ax.set_box_aspect([1.5,2,2], zoom=0.9)
# ax.set_xlabel('$g_0$')
# ax.set_ylabel('$\alpha_1$')

fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='$V_{p}$')

# plt.savefig(f'v2phi_bnd_KK{KK}_GG{GG}_contour.png')
# plt.savefig(f'v2phi_bnd_KK{KK}_GG{GG}_contour.pdf')
plt.show()


