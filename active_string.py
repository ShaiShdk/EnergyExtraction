#%%
########### Active Viscoelastic String ###########
"""
    Created on Oct 2023
    @author: Shai
    -------------------
    This code simulates a 1D chain of active viscoelastic substance,
    with symmetric...     
"""

parent_folder  = '/Users/shaish/Library/CloudStorage/Dropbox/Science'
parent_folder += '/Projects/EnergyExtraction/codes_resub'

import os
from copy import deepcopy

from scipy.sparse import csr_matrix
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

pltshow = True

os.chdir(parent_folder)

Ttot, dt = 2 , .00001
Nt = int(Ttot/dt - 1)
tseries = np.linspace(0,Ttot,Nt)

T_avg = int(Ttot/dt/2)

n_plot = np.min((10,Nt))
tplot = np.arange(Nt)[::int(Nt/n_plot)]

############# Field Parameters ############
phi_i = 1
GG = 1
KK = 1
a2 = 1
a1_series  = [3]
eta_series = [1]
########### Particle Parameters ###########

xmax = 1
x_res = 100
dx = xmax/x_res

xs = 0*xmax/2 + np.arange(-xmax/2,xmax/2,dx)
Nx = len(xs)
xs = xs.reshape((Nx,1))

eta = eta_series[0]
a1 = a1_series[0]

fig = plt.figure(figsize=(10,5),dpi=100,\
                        facecolor='w',edgecolor='w')

u_field = 0*deepcopy(xs)

inc0 = np.zeros((Nx-1 , Nx))
for ii in range(Nx-1):
    inc0[ii,ii] = 1
    inc0[ii,ii+1] = -1

inc0 = csr_matrix(inc0)
adj = inc0.T.dot(inc0)

v_field = np.zeros((Nx,1))
X_field = np.zeros((Nt,Nx))

F_particle = 0
plt_cnt = -1

phi = phi_i*np.ones((Nx-1,1))

KE = np.zeros((Nt,Nx))
PE = np.zeros((Nt,Nx-1))
dA = np.zeros((Nt,Nx))
dU = np.zeros((Nt,Nx))
dQ = np.zeros((Nt,Nx))

for tt in tqdm(range(Nt)):

    sigma1 = -a1 * phi
    sigma2 = +a2 * phi**2
    sigma = sigma1 + sigma2
    F_field = - eta * adj.dot(v_field).reshape((Nx,1))/dx**2 \
              - KK * adj.dot(u_field).reshape((Nx,1))/dx**2 \
              - GG * v_field.reshape((Nx,1)) - inc0.T.dot(sigma)/dx

    axel = F_field/phi_i
    du_field = + v_field.reshape((Nx,1)) * dt \
               + .5 * axel.reshape((Nx,1)) * dt**2

    u_field += du_field.reshape((Nx,1))
    v_field += axel.reshape((Nx,1)) * dt

    X_field[tt,:] = (xs + u_field).reshape(Nx,)

    eps = -inc0.dot(u_field)/dx
    phi = phi_i/(1+eps)

    #####################################################################

    KE[tt,:] = .5 * phi_i * (v_field**2).reshape(Nx,)/dx              # Kinetic Energy
    PE[tt,:] = .5 * (KK) * (eps**2).reshape(Nx-1,)/dx               # Potential Energy
    dA[tt,:] = (v_field * inc0.T.dot(sigma1)).reshape(Nx,)/dx**2
    dU[tt,:] = (v_field * inc0.T.dot(sigma2)).reshape(Nx,)/dx**2
    dQ[tt,:] = eta * (v_field * adj.dot(v_field)).reshape(len(v_field),)/dx**3\
               + GG * (v_field**2).reshape(Nx,)/dx

    if tt in tplot:
        plt_cnt += 1
        Xact = X_field[tt,:]
        X_avg = (X_field[tt,:-1] + X_field[tt,1:])/2
        plt.plot(X_avg,tt*dt*np.ones((len(X_avg),1)),\
                    '.',markersize=3,alpha=.5,\
                    color=[0,133/255,212/255])
        plt.plot(X_avg,tt*dt*np.ones((len(X_avg),1))\
                    +Ttot*phi[:len(X_avg)]/(2*a1*n_plot),\
                    linewidth=3,alpha=1,solid_capstyle='round',\
                    color=[151/255,14/255,83/255])


plt.xlabel(r'$X$',fontsize=30)
plt.ylabel(r'Time',fontsize=30,fontname='Times',\
            fontweight = 'bold')
plt.xticks(fontsize=20) ; plt.yticks(fontsize=20)
plt.show()

print('a1 =',a1,'; a2 =',a2)
print('eta =',eta,'; Kappa =',KK,'; gamma =',GG)
print('V_bnd =',np.around(X_field[tt,0]/Ttot,2))
print('-------------------------------------','\n')

KE_total = np.sum(KE, axis = 1)
PE_total = np.sum(PE, axis = 1)
dA_total = np.sum(dA, axis = 1)
dU_total = np.sum(dU, axis = 1)
dQ_total = np.sum(dQ, axis = 1)

AE_total = np.cumsum(dA_total) * dt
UE_total = np.cumsum(dU_total) * dt
QE_total = np.cumsum(dQ_total) * dt

#%%

IE_total = KE_total + PE_total

TOT_E = IE_total + QE_total + AE_total + UE_total

# plt.plot(tseries, dx*IE_total, '.' , tseries, dx*QE_total, '.' ,\
#         tseries, dx*AE_total, '.', tseries, dx*UE_total, '.',\
#             tseries, dx*TOT_E, '.', markersize= 1)#linewidth=3)
plt.plot(tseries, dx*IE_total, tseries, dx*QE_total, \
         tseries, dx*AE_total, tseries, dx*UE_total, \
            tseries, dx*TOT_E, linewidth=3)
plt.legend(['Internal Energy','Dissipated Energy',\
            'Active Energy','Pressure Energy','Total Energy'],loc='right',\
                bbox_to_anchor=(.95,.35))
plt.show()

# plt.plot(tseries[1:],KE_total[1:]/AE_total[1:])

#%%

# fig = plt.figure(figsize = (10,7),dpi=100,facecolor='w',edgecolor='w')
# # for ii in range(len(Vaf_series))
# # plt.plot(tseries,v_bnd[0,:,-1])
# plt.plot(eta_series,v_bnd[:,0,-1])

# for ii in range(len(eta_series)):
#     plt.plot(Veff_table[ii,:])

# fig = plt.figure(figsize=(10,7),dpi=100)
# plt.plot(tseries,v_bnd[0][0]/Vaf)
# plt.plot(tseries,1-phi_i/phi_bnd[0][0])
# plt.show()
# fig = plt.figure(figsize=(10,7),dpi=100)
# plt.plot(tseries,v_bnd[0][0]/phi_bnd[0][0])
# plt.show()

# if savedat:
#     os.chdir(parent_folder)
#     np.save('Veff_table' , Veff_table)

# # print((1-phi_i/phi_bnd[0][0][-1])/Veff[-1][0])
# print((1-phi_i/phi_bnd[0][0][-1]))#/v_bnd[0][0][-1])
# print(v_bnd[0][0][-1])

# print(v_bnd[0][0][-1]/Vaf/(1-phi_i/phi_bnd[0][0][-1]))

# print(np.around(t_finish - t_start , 2), 'secs')


#%%#############################################################################

FE = KE

# FE /= np.ptp(FE)

tmap = (np.linspace(0,Nt-1,Nx)).astype(int)
Ts = tseries[tmap]
Xf = X_field[tmap,:]
_ , Tf = np.meshgrid(xs, Ts)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(Tf, Xf, FE[tmap], cmap=cm.viridis,
                       linewidth=0,alpha=1)#, antialiased=False)

# surf = ax.plot_surface(Tf[:,:-2], Xf[:,:-2], PE[tmap,:-1], cmap=cm.viridis,
#                        linewidth=0,alpha=1)#, antialiased=False)
ax.view_init(elev=30, azim=250)
ax.set_box_aspect((2,1,1)) #.5*np.ptp(FE[tmap])/np.max(FE)))
# ax.set_box_aspect((2*np.ptp(Tf)/np.max(Tf), 1*np.ptp(Xf)/np.max(Xf), 1))#.5*np.ptp(FE[tmap])/np.max(FE)))


#%%

# IE = KE + PE
# tmap = (np.linspace(0,Nt-1,Nx)).astype(int)

# plt.imshow(np.flipud(KE[tmap,:]))
# plt.title('Kinetic')
# plt.colorbar()
# plt.show()
# plt.imshow(np.flipud(PE[tmap,:]))
# plt.title('Potential')
# plt.colorbar()
# plt.show()
# plt.imshow(np.flipud(IE[tmap,:]))
# plt.title('Internal')
# plt.colorbar()
# plt.show()
# plt.imshow(np.flipud(dA[tmap,:]))
# plt.title('Active')
# plt.colorbar()
# plt.show()
# plt.imshow(np.flipud(dU[tmap,:]))
# plt.title('dU term')
# plt.colorbar()
# plt.show()
# plt.imshow(np.flipud(dQ[tmap,:]))
# plt.title('Dissipated')
# plt.colorbar()
# plt.show()

