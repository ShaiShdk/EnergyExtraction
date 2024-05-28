#%%
########### Active Viscoelastic String w/ Overdamped Particle ###########
"""
    Created on Oct 2023
    @author: Shahriar Shadkhoo -- Caltech
    -------------------
    This code simulates a 1D chain of active viscoelastic substance,
    which pulls on a particle agains the drag force.     
"""

parent_folder  = 'parent_directory'

from copy import deepcopy
import os
from time import time

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib import rc
csfont = {'fontname':'Times New Roman'}
hfont  = {'fontname':'Times New Roman'}
mpl.rcParams['text.usetex'] = True
mpl.rc('font', **{'family': 'serif'})

t_start = time()

saveimg = True
savedat = True
pltshow = True

GG = 0
KK = None

local_folder = f'/GG{GG}'
os.chdir(parent_folder + local_folder)

v2phi_name = f'v2phi_GG{GG}'
v_bnd_name = f'v_bnd_GG{GG}'
phi_bnd_name = f'phi_GG{GG}'

KK_series  = np.around(np.linspace(.5,2,7),3)  
g0_series  = [None]      
a1_series  = np.around(np.linspace(2,10,9),3)  
Vaf_series = np.around(np.linspace(.5,10.,20),3) 

np.save('KK_series',KK_series)
np.save('g0_series',g0_series)
np.save('a1_series',a1_series)
np.save('Vaf_series',Vaf_series)

v2phi_ratio = np.zeros((len(KK_series),len(g0_series),len(a1_series),len(Vaf_series)))
v_bnd_infty = np.zeros((len(KK_series),len(g0_series),len(a1_series),len(Vaf_series)))
phi_bnd_infty = np.zeros((len(KK_series),len(g0_series),len(a1_series),len(Vaf_series)))

NUMRUNS = np.prod(v2phi_ratio.shape)

Ttot, dt = 5 , .0001
Nt = int(Ttot/dt - 1)
tseries = np.linspace(0,Ttot,Nt)

n_plot = np.min((10,Nt))
tplot = np.arange(Nt)[::int(Nt/n_plot)]
if not(pltshow):
    tplot = []

############# Field Parameters ############
phi_i = 1
eta = 1
a2 = 1
########### Particle Parameters ###########
Gamma0 = 1

xmax = 50#np.max(Vaf_series) * Ttot
x_res = 100
dx = xmax/x_res

xs = xmax/2 + np.arange(-xmax/2,xmax/2,dx)
Nx = len(xs)
xs = xs.reshape((Nx,1))

inc0 = np.zeros((Nx-1 , Nx))
for ii in range(Nx-1):
    inc0[ii,ii] = 1
    inc0[ii,ii+1] = -1

def time_avg(F , T , dt=dt):
    F_avg = (dt/T) * np.convolve(F.reshape(len(F),),np.ones((int(T/dt),)))
    return F_avg[:len(F)]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

total_cnt = 0

k_ind = 0
for KK in KK_series:
    g_ind = 0
    for g0 in g0_series:
        a_ind = 0
        for a1 in a1_series:
            v_ind = 0
            for Vaf in Vaf_series:

                if (savedat and saveimg):    
                    saveFolder = parent_folder + local_folder + f'/{KK:.3f}_g0{g0}_{a1:.3f}_{Vaf:.3f}'
                    if not(os.path.exists(saveFolder)):
                        os.mkdir(saveFolder)
                    os.chdir(saveFolder)

                if savedat:
                    with open('a1_Vaf_metadata.txt','a') as f:
                        f.write('-------------------------- PARAMETERS -------------------------' + '\n')
                        f.write('---------------------------------------------------------------' + '\n')
                        f.write('g0 = '     + 'None' + '\n')
                        f.write('alpha = '  + str(a1) + '\n')
                        f.write('Vaf = '    + str(Vaf) + '\n')
                        f.write('eta = '    + str(eta) + '\n')
                        f.write('Gamma0 = ' + str(Gamma0) + '\n')
                        f.write('---------------------------------------------------------------' + '\n')
                        f.write('---------------------------------------------------------------' + '\n')
                    f.close()

                t_update = np.arange(0,Nt,int(dx/Vaf/dt)+1)

                tinit = time()
                total_cnt += 1

                if pltshow:
                    fig = plt.figure(figsize=(10,5),dpi=100,\
                                    facecolor='w',edgecolor='w')

                u_field = .0*deepcopy(xs)
                v_field = np.zeros((Nx,1))
                Xfield  = xs + u_field

                phi_bnd = np.zeros((Nt,1))
                v_bnd   = np.zeros((Nt,1))

                Xaf     = np.min(Xfield)

                F_particle = 0
                plt_cnt = -1

                for tt in tqdm(range(Nt-1)):
                    Xaf += Vaf * dt
                    if tt in t_update:
                        Xed = (Xfield[1:] + Xfield[:-1])/2
                        inactive_eds = np.where(Xed > Xaf)[0]
                        inc = deepcopy(inc0)
                        inc[inactive_eds,:] = np.zeros((len(inactive_eds),Nx))
                        # active_eds = list(set(range(Nx)) - set(inactive_eds))

                    eps = -inc.dot(u_field)/dx
                    phi = phi_i/(1+eps)

                    sigma1 = -a1 * phi
                    sigma2 = +a2 * phi**2
                    sigmaT = sigma1 + sigma2

                    adj = inc.T.dot(inc)
                    F_field = - eta * adj.dot(v_field).reshape((Nx,1))/dx**2 \
                              - KK * adj.dot(u_field).reshape((Nx,1))/dx**2 \
                              - inc.T.dot(sigmaT)/dx \
                              - GG * v_field.reshape((Nx,1)) 

                    axel = F_field/phi_i
                    du_field = + v_field.reshape((Nx,1)) * dt \
                               + .5 * axel.reshape((Nx,1)) * dt**2
                    u_field += du_field.reshape((Nx,1))
                    v_field += axel.reshape((Nx,1)) * dt

                    Xfield = xs + u_field

                    v_bnd[tt,0] = v_field[0][0]
                    phi_bnd[tt,0] = phi[0][0]

                    #####################################################################

                    if tt in tplot:
                        plt_cnt += 1
                        
                        Xact = Xfield[Xfield<Xaf]
                        Xact_avg = (Xact[:-1] + Xact[1:])/2
                        plt.plot(Xfield,tt*dt*np.ones((len(Xfield),1)),\
                                '.',markersize=3,alpha=.5,\
                                color=[0,133/255,212/255])
                        plt.plot(Xact[:],tt*dt*np.ones((len(Xact[:]),1)),\
                                '--',markersize=3,alpha=1,\
                                color=[151/255,14/255,83/255])
                        plt.plot(Xact_avg,tt*dt*np.ones((len(Xact_avg),1))\
                                +Ttot*phi[:len(Xact_avg)]/(2*a1*n_plot),\
                                linewidth=3,alpha=1,solid_capstyle='round',\
                                    color=[151/255,14/255,83/255])

                if pltshow:
                    plt.plot(np.min(xs)+Vaf*tseries[:tt],tseries[:tt],'-',\
                            linewidth=5,solid_capstyle='round',\
                            color=[1/255,76/255,128/255])
                    plt.xlabel(r'$X$',fontsize=30)
                    plt.ylabel(r'Time',fontsize=30,fontname='Times',\
                            fontweight = 'bold')
                    plt.xticks(fontsize=20) ; plt.yticks(fontsize=20)
                    if saveimg:
                        imgname =  r'Vaf%r.'%round(Vaf,3) + r'_a1%r'%round(a1,3)\
                                 + r'_eta%r'%round(eta,3) + r'_kappa%r'%round(KK,3)\
                                 + r'_g0None'
                        plt.savefig(imgname + '.png')
                        plt.savefig(imgname + '.pdf')
                    plt.show()

                v_bnd = time_avg(v_bnd,eta/(a2*phi_i**2),dt)
                phi_bnd = time_avg(phi_bnd,eta/(a2*phi_i**2),dt)

                v_bnd[-1] = v_bnd[-2]
                phi_bnd[-1] = phi_bnd[-2]
                v2phi_ratio[k_ind,g_ind,a_ind,v_ind] = np.max(v_bnd/phi_bnd , axis=0)

                v_bnd_infty[k_ind,g_ind,a_ind,v_ind] = v_bnd[-1]
                phi_bnd_infty[k_ind,g_ind,a_ind,v_ind] = phi_bnd[-1]

                if pltshow:
                    fig = plt.figure(figsize=(10,5),dpi=100,\
                                    facecolor='w',edgecolor='w',linewidth=5)
                    plt.plot(tseries,v_bnd/Vaf,linewidth=5,solid_capstyle='round')
                    plt.xlabel(r'Time',fontsize=30)
                    plt.ylabel(r'$v_b/\phi_b$',fontsize=30,fontname='Times',\
                                fontweight = 'bold')
                    plt.xticks(fontsize=20) ; plt.yticks(fontsize=20)
                    if saveimg:
                        imgname = r'v2phi' +  r'Vaf%r.'%round(Vaf,3) + r'_a1%r'%round(a1,3)\
                                 + r'_eta%r'%round(eta,3) + r'_kappa%r'%round(KK,3)\
                                 + r'_g0None'
                        plt.savefig(imgname + '.png')
                        plt.savefig(imgname + '.pdf')
                    plt.show()

                v_ind += 1

                tfinish = time()

                print('a1 =',a1,'; a2 =',a2,'; Vaf =',Vaf)
                print('eta =',eta,'; Kappa =',KK,'; gamma =',GG)
                print('Gamma =',Gamma0, '; g0 = ' , ' None')
                print('Phi_bnd =',np.around(phi_bnd[-1],2))
                print('V_bnd/Vaf =',np.around(v_bnd[-1]/Vaf,2))
                print('RunTime =',int(tfinish - tinit),'sec')
                print('Total Time =',int((tfinish - t_start)//60),'min',\
                      int((tfinish - t_start)%60),'sec')
                print('Remaining Runs =', NUMRUNS - total_cnt)
                print('Average RunTime =',int((tfinish - t_start)/total_cnt),'sec')
                print('-------------------------------------','\n')

                np.save('v_boundary',v_bnd)
                np.save('phi_boundary',phi_bnd)
                np.save('v2phi',v2phi_ratio)

                if savedat:
                    with open('a1_Vaf_phase_diagram.txt','a') as f:
                        f.write('--------------------' + '\n')
                        f.write('g0   ; a1   ; Vaf  ; V_bnd/Vaf  ;  phi_bnd' + '\n')
                        f.write('--------------------' + '\n')
                    f.close()    
                    with open('a1_Vaf_metadata.txt','a') as f:
                        f.write('----------------------------------' + '\n')
                    f.close

                if (savedat and saveimg):
                    os.chdir(saveFolder)                        
                    with open('a1_Vaf_phase_diagram.txt','a') as f:
                        f.write( 'None' + '  ; ' + str(a1) + '  ; ' + str(Vaf) \
                                + '  ; ' + str(round(v_bnd[-1],3)) + '  ; ' + str(round(phi_bnd[-1],3)) + '\n' + '\n')
                    f.close()    
                    with open('a1_Vaf_metadata.txt','a') as f:
                        f.write('g0 = ' 'None' + ' ; a1 = ' + str(a1) + ' ; Vaf = ' + str(Vaf) + '\n')
                        f.write('----------------------------------' + '\n')
                        f.write('Phi_bnd = ' + str(np.around(phi_bnd[-1],2)) + '\n')
                        f.write('V_bnd/Vaf = ' + str(np.around(v_bnd[-1]/Vaf,2)) + '\n')
                        f.write('RunTime = ' + str(int(tfinish - tinit)) + ' sec' + '\n')
                        f.write('Total Time = ' + str(int((tfinish - t_start)//60)) + ' min '\
                                + str(int((tfinish - t_start)%60)) + ' sec' + '\n')
                        f.write('Remaining Runs = ' + str(NUMRUNS - total_cnt) + '\n')
                        f.write('Average RunTime = ' + str(int((tfinish - t_start)/total_cnt)) + ' sec' + '\n')            
                        f.write('----------------------------------' + '\n')
                    f.close()    

                os.chdir(parent_folder)

            a_ind += 1
        g_ind += 1
    k_ind += 1

t_finish = time()

vb = deepcopy(v_bnd_infty[0,0,:,0]/Vaf)
plt.plot(a1_series,vb)

v2phi_kk = deepcopy(v2phi_ratio[0,...])

if savedat:
    np.save(v2phi_name,v2phi_ratio)
    np.save(v_bnd_name,v_bnd_infty)
    np.save(phi_bnd_name,phi_bnd_infty)

#%%

from scipy.ndimage.filters import gaussian_filter

parent_folder  = '/Users/shaish/Library/CloudStorage/Dropbox/Science'
parent_folder += '/Projects/EnergyExtraction/codes_results_resub/GitHub_NX_codes/NX_results'

KK , GG = 0 , 0

local_folder = f'/KK{KK}_GG{GG}_highRes'
os.chdir(parent_folder + local_folder)

v2phi_name  = f'v2phi_KK{KK}_GG{GG}.npy'
vbnd_name   = f'v_bnd_KK{KK}_GG{GG}.npy'
pbnd_name   = f'phi_bnd_KK{KK}_GG{GG}.npy'


Vaf_series = np.around(np.linspace(1.,10,37),3)
a1_series = np.around(np.linspace(1.5,10,35),3)

vbnd = np.load(vbnd_name)
pbnd = np.load(pbnd_name)

Vmap = vbnd[0,0,...]/pbnd[0,0,...]#deepcopy(v_bnd_infty[0,0,...]/phi_bnd_infty[0,0,...])
Vmap = np.flipud(Vmap)

vxp = vbnd[0,0,...] * pbnd[0,0,...]
vxp = np.flipud(vxp)

gauss_sigma = 2
Vmap = gaussian_filter(Vmap, gauss_sigma)

plt.imshow(Vmap,interpolation='bicubic',cmap=mpl.colormaps['YlGnBu']\
           ,extent=[np.min(a1_series),np.max(a1_series),np.min(Vaf_series),np.max(Vaf_series)])
plt.colorbar()
plt.contour(np.flipud(Vmap),levels=np.arange(0,np.max(Vmap),.05),colors=[(0,.5,.7)]\
            ,extent=[np.min(a1_series),np.max(a1_series),np.min(Vaf_series),np.max(Vaf_series)])

if saveimg:
    plt.savefig(f'v2phi_bnd_KK{KK}_GG{GG}_heatmap.png')
    plt.savefig(f'v2phi_bnd_KK{KK}_GG{GG}_heatmap.pdf')

#%%

saveimg = False

Gamma0 = 1
g0_series = np.around(np.linspace(0,.25,51),3)

v2pb = np.zeros((len(g0_series),len(a1_series),len(Vaf_series)))
vXpb = np.zeros((len(g0_series),len(a1_series),len(Vaf_series)))

for jj in range(len(g0_series)):
    v2pb[jj,...] = Vmap

for jj in range(len(g0_series)):
    vg1 = v2pb[jj,:,:]
    lim_g0 = g0_series[jj]/Gamma0
    vg1[vg1 > lim_g0] = 0
    v2pb[jj,:,:] = vg1.astype(bool)

X,Y,Z = np.meshgrid(g0_series,a1_series,Vaf_series)

data = np.swapaxes(v2pb , 0 , 1)

kw = {
    'vmin': data.min(),
    'vmax': data.max(),
    'levels': np.around(np.linspace(data.min(), data.max(), 20), 2),    
    'cmap': mpl.colormaps['YlGnBu']
}

# Create a figure with 3D ax
fig = plt.figure(figsize=(5, 4),dpi=100)
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

edges_kw = dict(color=[.1,.3,.6], linewidth=.5, zorder=1e3)
ax.plot([xmax, xmax], [ymin, ymax], [zmin, zmin], **edges_kw)
ax.plot([xmin, xmax], [ymin, ymin], [zmin, zmin], **edges_kw)
ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
ax.plot([xmax, xmin], [ymin, ymin], [zmax, zmax], **edges_kw)
ax.plot([xmin, xmin], [ymin, ymax], [zmax, zmax], **edges_kw)
ax.plot([xmin, xmin], [ymin, ymin], [zmax, zmin], **edges_kw)
ax.plot([xmin, xmax], [ymax, ymax], [zmin, zmin], **edges_kw)
ax.plot([xmax, xmax], [ymin, ymax], [zmax, zmax], **edges_kw)
ax.plot([xmin, xmin], [ymin, ymax], [zmin, zmin], **edges_kw)
ax.plot([xmax, xmax], [ymax, ymax], [zmin, zmax], **edges_kw)
ax.plot([xmin, xmax], [ymax, ymax], [zmax, zmax], **edges_kw)
ax.plot([xmin, xmin], [ymax, ymax], [zmin, zmax], **edges_kw)

ax.view_init(elev=30, azim=150, roll=0)
ax.set_box_aspect([2,2,2], zoom=0.9)

if saveimg:
    plt.savefig(f'v2phi_bnd_KK{KK}_GG{GG}_3Dmanifolds.png')
    plt.savefig(f'v2phi_bnd_KK{KK}_GG{GG}_3Dmanifolds.pdf')
plt.show()

