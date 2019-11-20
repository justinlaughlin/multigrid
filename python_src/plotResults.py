# plotResults.py

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fontsize = 20

#=======================
# import data
dataDir = "../build/"
X = np.loadtxt(dataDir + "X.txt")
Y = np.loadtxt(dataDir + "Y.txt")

N = X.shape[0]
# initialize with zeros so boundaries are set to 0 (dirichlet condition)
umg = np.zeros((N,N))
#ucholesky = np.zeros((N,N))
ua = np.zeros((N,N))
errmg = np.zeros((N,N))
#errcholesky = np.zeros((N,N))

# set interior nodes to proper values
umg[1:-1,1:-1] = np.loadtxt(dataDir + "umg.txt").reshape((N-2,N-2))
#ucholesky[1:-1,1:-1] = np.loadtxt(dataDir + "ucholesky.txt").reshape((N-2,N-2))
ua[1:-1,1:-1] = np.loadtxt(dataDir + "ua.txt").reshape((N-2,N-2))
errmg[1:-1,1:-1] = np.loadtxt(dataDir + "errorMultigrid.txt").reshape((N-2,N-2))
#errcholesky[1:-1,1:-1] = np.loadtxt(dataDir + "errorCholesky.txt").reshape((N-2,N-2))



max_abs = np.max(np.abs(ua))
err_max_abs = np.max(np.abs(errmg))

# plot
fig,ax = plt.subplots(1,3)
#ax = fig.add_subplot(111, projection='3d')
ax[0].set_title("u [multigrid]", size=fontsize)
#ax[1].title.set_text("u [cholesky]")
ax[1].set_title("u [analytic]", size=fontsize)
c = ax[0].pcolormesh(X,Y,umg,cmap='RdBu',vmin=-max_abs,vmax=max_abs)
#ax[1].pcolormesh(X,Y,ucholesky,cmap='RdBu',vmin=-max_abs,vmax=max_abs)
ax[1].pcolormesh(X,Y,ua,cmap='RdBu',vmin=-max_abs,vmax=max_abs)


ax[2].set_title("error [multigrid]", size=fontsize)

ax[0].tick_params(labelsize=fontsize)
ax[1].tick_params(labelsize=fontsize)
ax[2].tick_params(labelsize=fontsize)

#ax[1,1].title.set_text("error [cholesky]")
cerr = ax[2].pcolormesh(X,Y,errmg,cmap='BrBG',vmin=-err_max_abs,vmax=err_max_abs)
#ax[1,1].pcolormesh(X,Y,errcholesky,cmap='BrBG',vmin=-err_max_abs,vmax=err_max_abs)
#ax[3].set_visible(False)

fig.subplots_adjust(right=0.76) # make room for colorbars

#surf = ax.plot_surface(X,Y,ucholesky,cmap=cm.coolwarm)
#fig.colorbar(surf, shrink=0.5, aspect=5)
cbar_ax = fig.add_axes([0.8,0.15,0.04,0.7])
cbar_ax2 = fig.add_axes([0.9,0.15,0.04,0.7])
fig.colorbar(c,cax=cbar_ax)#,pad=0.2)
fig.colorbar(cerr,cax=cbar_ax2)#,pad=0.2)

cbar_ax.tick_params(labelsize=fontsize)
cbar_ax2.tick_params(labelsize=fontsize)

plt.show()
