#
#   Date created: 17 Dec 2020
#
#   Description: Given the extracted (x,y) coordinates, calculate angle and
#                   it's first and second derivatives using finite differences.
#                   This will be our training (+ testing) data.
#

# %% Imports
import pickle
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.signal import savgol_filter

# Setting up few global parameters
data_path = pathlib.Path("data")
DPI = 100
plt.style.use('dark_background')

# Loading red and blue end's xy coordinates
with open(data_path/'bluexy.pkl', 'rb') as fh:
    bluexy = pickle.load(fh)

with open(data_path/'redxy.pkl', 'rb') as fh:
    redxy = pickle.load(fh)

# converting list of tuples (x,y) -> numpy 2D array
np_bluexy = np.asarray(bluexy)
np_redxy = np.asarray(redxy)

print(f'Shape of blue points : {np_bluexy.shape}')
print(f'Shape of red points : {np_redxy.shape}')


# %%
data = {}

# converting (x1, y1, x2, y2) -> {\theta1, \theta2}
theta = np.arctan2(np_redxy[:, 1]-np_bluexy[:, 1],
                   np_redxy[:, 0] - np_bluexy[:, 0])

data['t'] = np.linspace(0, 100, len(np_bluexy))
data['theta'] = theta
# detrending \theta
pf = PolynomialFeatures(degree=3)
xp = pf.fit_transform(data['t'].reshape(-1, 1))
model_p = LinearRegression()
model_p.fit(xp, data['theta'])
trendp = model_p.predict(xp)

fig = plt.figure(figsize=(10, 6), dpi=DPI)
ax = fig.add_subplot(211)
ax.plot(data['theta'])
ax.plot(trendp)
ax.legend(['data', 'legend'])

ax2 = fig.add_subplot(212)
ax2.plot(data['theta']-trendp)


# %% Calculating first derivative of extracted (and detrended) angle using finite difference
data['theta'] = data['theta']-trendp

data['dtheta'] = (data['theta'][1:]-data['theta']
                  [:-1]) / (100.0/len(np_bluexy))

# smoothing the first derivative using savgol filter and calculating second derivative using finite difference
dtheta = savgol_filter(data['dtheta'], 23, polyorder=3)
data['dtheta'] = dtheta
data['d2theta'] = (data['dtheta'][1:]-data['dtheta']
                   [:-1]) / (100.0/len(np_bluexy))
dtheta2 = savgol_filter(data['d2theta'], 23, polyorder=3)
data['d2theta'] = dtheta2

data['dtheta'] = data['dtheta'][:-1]
data['theta'] = data['theta'][:-2]


d1 = np.concatenate(
    (data['t'][:-2].reshape(-1, 1), data['theta'].reshape(-1, 1)), axis=1)
d2 = np.concatenate(
    (data['dtheta'].reshape(-1, 1), data['d2theta'].reshape(-1, 1)), axis=1)
data_dump = np.concatenate((d1, d2), axis=1)
print(f'Shape of dumped data : {data_dump.shape}')
np.savetxt(data_path/'extracted_angles.csv',
           data_dump, delimiter=',', fmt='%.4f')

# Plot the trajectories
fig = plt.figure(figsize=(8, 6), dpi=150)
ax1 = fig.add_subplot(211)
ax1.set_title('Phase potrait')
#ax1.plot(data['theta'], data['dtheta'], '-b', label=r'$\theta$', alpha=1.0)
ax1.plot(data['theta'], dtheta[:-1], '-b', label=r'$\theta$', alpha=1.0)
ax1.set_ylim(-0.5, 0.5)

ax2 = fig.add_subplot(212)
ax2.plot(data['theta'], '-b', label=r'$\theta$', alpha=1.0, lw=1.0)
ax2.set_xlabel('Time')
ax2.set_ylabel('angle')
ax2.plot(dtheta[:-1], '-r', label=r'$\dot{\theta}$', alpha=0.8, lw=0.8)
ax2.plot(data['d2theta'], '-', label=r'$\ddot{\theta}$', alpha=0.7)
ax2.set_ylim(-3, 3)
ax2.legend(loc='best')

plt.tight_layout()

plt.show()
