#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:50:39 2018

@author: tenecarter
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np;
from matplotlib import cm

def F(yn,a,dt,T,yo):

    yTest = model(a,yo,dt,T)[0].tolist();
    out = np.linalg.norm(np.subtract(yn,yTest))**2;
    return out;

def model(a,yo,dt,T):
    t = np.arange(0,T,dt)
    N = t.size;
    y = np.zeros(N);
    y[0] = yo;
    for kk in range(1 , N):
        y[kk] = (1+a*dt)*y[kk-1]-dt*a;
    return y,t

a = -.9;
T = 10;
dt = 0.01;
yo = 0;
[y,t] = model(a,yo,dt,T);
y= y.tolist();
t= t.tolist();
yn = [];
for k in range(0,len(y)):
    yn.append(y[k] + 0.5*np.random.normal());
plt.plot(t,yn);
plt.plot(t,y);


aTest = np.arange(-1.2,-.5,.01).tolist();
yoTest =np.arange(-.3,.5, .05)
Fs = np.zeros((len(yoTest),len(aTest)));
for kk in range( 0,len(aTest)):
    for jj in range(0 , len(yoTest)):
        Fs[jj,kk] = F(yn,aTest[kk],dt,T,yoTest[jj]);

#ind = Fs.index(min(Fs));
#aOpt  = aTest[ind];
#[yOpt,t] = model(aOpt,yo,dt,T);
#plt.plot(t,yOpt,'r',2)
minimumVal = np.min(Fs)
indecies = np.argwhere(Fs == minimumVal)

#fig = plt.figure(2)
#ax = fig.gca(projection='3d')
#aTest, yoTest= np.meshgrid(aTest, yoTest)
#surf = ax.plot_surface(yoTest, aTest, Fs, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()

print(aTest[indecies[0][1]])
print(yoTest[indecies[0][0]])