#%%
########### Active Viscoelastic String w/ Overdamped Particle ###########
"""
    Created on Oct 2023
    @author: Shai
    -------------------
    This code simulates a 1D chain of active viscoelastic substance,
    which pulls on a particle agains the drag force.     
"""

parent_folder  = '/Users/shaish/Library/CloudStorage/Dropbox/Science'
parent_folder += '/Projects/EnergyExtraction/codes_results_resub/GitHub_NX_codes/NX_results/KK1_GG0_highRes'

v2phi_name = 'v2phi_KK1_GG1'
v_bnd_name = 'v_bnd_KK1_GG1'
phi_bnd_name = 'phi_bnd_KK1_GG1'

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

os.chdir(parent_folder)

KK_series  = [1]
Vaf_series = np.around(np.linspace(1.,10,37),3)
a1_series  = np.around(np.linspace(1.5,10,35),3)
g0_series  = [None]#np.around(np.linspace(.1,.3,9),3)

np.save('KK_series',KK_series)
np.save('g0_series',g0_series)
np.save('a1_series',a1_series)
np.save('Vaf_series',Vaf_series)

v2phi_ratio = np.zeros((len(KK_series),len(g0_series),len(a1_series),len(Vaf_series)))
v_bnd_infty = np.zeros((len(KK_series),len(g0_series),len(a1_series),len(Vaf_series)))
phi_bnd_infty = np.zeros((len(KK_series),len(g0_series),len(a1_series),len(Vaf_series)))

NUMRUNS    = np.prod(v2phi_ratio.shape)

Ttot, dt = 5 , .0001
Nt = int(Ttot/dt - 1)
tseries = np.linspace(0,Ttot,Nt)

n_plot = np.min((10,Nt))
tplot = np.arange(Nt)[::int(Nt/n_plot)]
if not(pltshow):
    tplot = []

############# Field Parameters ############
phi_i = 1
GG = 0
eta = 1
a2 = 1
########### Particle Parameters ###########
Gamma0 = 1

xmax = np.max(Vaf_series) * Ttot
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
                    saveFolder = parent_folder + f'/{KK:.3f}_g0{g0}_{a1:.3f}_{Vaf:.3f}'
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

                phi_x = phi_i * np.ones((Nx,1))
                u_field = .0*deepcopy(xs)

                v_field = np.zeros((Nx,1))
                Xfield  = xs + u_field

                X_field = np.zeros((Nt,Nx))

                phi_bnd = np.zeros((Nt,1))
                v_bnd   = np.zeros((Nt,1))

                Xaf     = np.min(Xfield)

                phi_xt  = np.zeros((len(tplot),Nx-1))
                Xfield_Time = np.zeros((len(tplot),Nx))
                F_particle = 0
                plt_cnt = -1

                for tt in tqdm(range(Nt-1)):
                    Xaf += Vaf * dt
                    if tt in t_update:
                        Xed = (Xfield[1:] + Xfield[:-1])/2
                        inactive_eds = np.where(Xed > Xaf)[0]
                        inc = deepcopy(inc0)
                        inc[inactive_eds,:] = np.zeros((len(inactive_eds),Nx))
                        
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
                    X_field[tt,:] = (xs + u_field).reshape(Nx,)

                    v_bnd[tt,0] = v_field[0][0]
                    phi_bnd[tt,0] = phi[0][0]

                    #####################################################################

                    if tt in tplot:
                        plt_cnt += 1
                        Xfield_Time[plt_cnt,:] = Xfield.reshape(Nx,)          
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
                    plt.plot(tseries,v_bnd/phi_bnd,linewidth=5,solid_capstyle='round')
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
                print('V_bnd/Vaf =',np.around(v_bnd[-1]/Vaf/Ttot,2))
                print('RunTime =',int(tfinish - tinit),'sec')
                print('Total Time =',int((tfinish - t_start)//60),'min',\
                    int((tfinish - t_start)%60),'sec')
                print('Remaining Runs =', NUMRUNS - total_cnt)
                print('Average RunTime =',int((tfinish - t_start)/total_cnt),'sec')
                print('-------------------------------------','\n')

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
                        f.write('V_bnd/Vaf = ' + str(np.around(v_bnd[-1]/Vaf/Ttot,2)) + '\n')
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

# import beepy
# beepy.beep(sound='ping')

v2phi_kk = deepcopy(v2phi_ratio[0,...])

if savedat:
    np.save(v2phi_name,v2phi_ratio)
    np.save(v_bnd_name,v_bnd_infty)
    np.save(phi_bnd_name,phi_bnd_infty)


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

veff = v2phi/np.max(v2phi)
veff = np.swapaxes(veff,0,1)

veff = v2phi

ax_num = 0
for aa in range(veff.shape[ax_num]):
    plt.imshow(np.flipud(veff[aa,:,:]),interpolation='bicubic',cmap=mpl.colormaps['bone'],extent=[np.min(a1_series),np.max(a1_series),np.min(Vaf_series),np.max(Vaf_series)])
    plt.show()

ax_num = 1
for aa in range(veff.shape[ax_num]):
    plt.imshow(np.flipud(veff[:,aa,:]),interpolation='bicubic',cmap=mpl.colormaps['bone'],extent=[np.min(g0_series),np.max(g0_series),np.min(Vaf_series),np.max(Vaf_series)])
    plt.show()

ax_num = 2
for aa in range(veff.shape[ax_num]):
    plt.imshow(np.flipud(veff[:,:,aa]),interpolation='bicubic',cmap=mpl.colormaps['bone'],extent=[np.min(g0_series),np.max(g0_series),np.min(a1_series),np.max(a1_series)])
    plt.show()


#%%

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# surf = ax.plot_surface(gg, alpha, Vaf, cmap=cm.viridis,
                    #    linewidth=0,alpha=1)#, antialiased=False)

ax.view_init(elev=30, azim=250)
ax.set_box_aspect((2,1,1))
