# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:34:06 2020

@author: raly
"""

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot(P):
    f, ax = plt.subplots(1, 3, figsize=(15,5))
    alpha_pts = 0.5
    
    i = 0
    #ax[i].plot(P_gen[:,0], P_gen[:,1], 'y-', lw=3, label='Generating circle')
    ax[i].scatter(P[0,:], P[1,:], alpha=alpha_pts, label='Cluster points P')
    ax[i].set_title('View X-Y')
    ax[i].set_xlabel('x'); ax[i].set_ylabel('y');
    ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1)
    ax[i].grid()
    i = 1
    #ax[i].plot(P_gen[:,0], P_gen[:,2], 'y-', lw=3, label='Generating circle')
    ax[i].scatter(P[0,:], P[2,:], alpha=alpha_pts, label='Cluster points P')
    ax[i].set_title('View X-Z')
    ax[i].set_xlabel('x'); ax[i].set_ylabel('z'); 
    ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1)
    ax[i].grid()
    i = 2
    #ax[i].plot(P_gen[:,1], P_gen[:,2], 'y-', lw=3, label='Generating circle')
    ax[i].scatter(P[1,:], P[2,:], alpha=alpha_pts, label='Cluster points P')
    ax[i].set_title('View Y-Z')
    ax[i].set_xlabel('y'); ax[i].set_ylabel('z'); 
    ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1)
    ax[i].legend()
    ax[i].grid()
    


# # Create dummy transformation
# mu = np.array([[4.0], [3.0], [2.0]])
# theta = (np.pi/2.0) * (0.9)
# phi = np.deg2rad(50)
# k = np.array([-np.sin(phi), np.cos(phi), 0.0])
# n = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
# m = np.cross(n, k)
# R = np.c_[k, m, n]

# T = np.r_[np.c_[R, mu], np.array([[0, 0, 0, 1]])]

# # Sample circle points
# no = 20
# r = 2.0
# theta = np.linspace(-np.pi, np.pi, no)
# pf = np.r_[[r*np.cos(theta)], [r*np.sin(theta)], np.zeros((1,no))]

# fig = plt.figure(1)
# plt.scatter(pf[0,:], pf[1,:])
# ax = plt.gca()
# ax.set_aspect('equal', 'datalim')
# plt.show()

# # Project into 3D
# P_data = R.dot(pf) + mu
# plot(P_data)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(p_data[0,:], p_data[1,:], p_data[2,:])
# plt.show()


P_data = np.array([[86.5757, -6.04848, 1026.08],[82.9267, -34.7781, 1018.61],[74.3755, -68.1183, 1001.44],[65.5652, -88.5409, 983.91],[56.7572, -102.455, 966.433],[42.4992, -116.171, 938.348],[32.4782, -120.81, 918.667],[21.782, -121.833, 897.717],[8.39108, -117.553, 871.691],[-0.402378, -111.173, 854.646],[-9.6755, -100.975, 836.893]], dtype=np.float32).T


# Compute axis of rotation
mu_est = P_data.dot(np.ones((P_data.shape[1],1), dtype=np.float32)) / P_data.shape[1]
U,S,VT = np.linalg.svd((P_data - mu_est).T)
n_est = VT[2,:]
if n_est.dot(np.array([1,0,0], dtype=np.float32)) < 0.0:
    n_est *= -1.0
m_est = np.cross(n_est, np.array([0.0, 0.0, -1.0], dtype=np.float32))
m_est /= np.linalg.norm(m_est)
k_est = np.cross(m_est, n_est)
R_est = np.c_[k_est, m_est, n_est].T
t_est = -R_est.dot(mu_est)

T_est = np.r_[np.c_[R_est, t_est], np.array([[0, 0, 0, 1]], dtype=np.float32)]
P_proj = T_est.dot(np.r_[P_data, np.ones((1,P_data.shape[1]))])

A = np.c_[P_proj[0,:], P_proj[1,:], np.ones((P_data.shape[1],))]
b = np.power(P_proj[0,:], 2.0) + np.power(P_proj[1,:], 2.0)
c,res,rank,s = np.linalg.lstsq(A, b, rcond=None)

xc = c[0] / 2.0
yc = c[1] / 2.0
rc = np.sqrt(c[2] + np.power(xc,2.0) + np.power(yc,2.0))

X = np.linalg.inv(R_est).dot(np.array([[xc], [yc], [0.0]])) + mu_est
print("Estimated radius:")
print(rc)
print("Center point:")
print(X)
print("Axis of rotation:")
print(n_est)

# Compute residuals



