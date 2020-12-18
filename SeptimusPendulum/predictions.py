# %% Imports
from functools import reduce
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ihnn import iHNN, train, transformVectorField
from baseline_nn import BNN, trainBS
from hnn import HNN, trainHNN
from scipy.integrate import solve_ivp

# %% Global constants
TRAIN_TEST_RATIO = 0.8
DPI = 150
plt.style.use('dark_background')

# %%  loading test data
file_path = Path('data')
dataset = np.loadtxt(file_path/'extracted_angles.csv',
                     dtype=float, delimiter=',')
train_ds = dataset[:int(TRAIN_TEST_RATIO*len(dataset))]
test_ds = dataset[int(TRAIN_TEST_RATIO*len(dataset)):]
test_x = torch.tensor(test_ds[:, 1:3], requires_grad=True, dtype=torch.float32)
# %% Let's predict trajectories


def model_update(t, state, model, model_name='gNNN'):
    '''
        helper function that takes in the trained model and returns it's forward pass. This will be used as a function that returns derivatives in ODE integrators
    '''
    state = state.reshape(-1, 2)
    x = torch.tensor(state, dtype=torch.float32, requires_grad=True)
    deriv = np.zeros_like(state)
    if model_name == 'gHNN':
        H, can_cords = model.forward(x)
        can_cords_dot = model.time_derivative(can_cords, H)
        dxdt_hat = transformVectorField(model, x, can_cords_dot)
    else:
        dxdt_hat = model.forward(x)

    deriv = dxdt_hat.detach().numpy()
    return deriv


# %% load the trained model
save_path = Path('trained_models')
learned_ihnn = iHNN(d_in=2, activation_fn='Tanh', device='cpu')
learned_ihnn.load_state_dict(torch.load(
    save_path/'SP_EXP_gHNN_seed_957_lr_0.001_bs_16_epochs_50.pkl'), strict=False)
learned_baseline = BNN(d_in=2, activation_fn='Tanh', device='cpu')
learned_baseline.load_state_dict(torch.load(
    save_path/'SP_EXP_BNN_seed_957_lr_0.001_bs_16_epochs_50.pkl'), strict=False)


# Making predictions on the test data
data_inference = test_x.detach().numpy()
in_state = data_inference[0]
t_obs = test_ds[:1000, 0]-test_ds[0, 0]
true_traj = data_inference[0:1000]


# gHNN inference
def update_fn(t, y0):
    return model_update(t, y0, learned_ihnn, model_name='gHNN')


gHNN_traj = solve_ivp(update_fn, t_span=[
    0, 20], y0=in_state, t_eval=t_obs, rtol=1e-12)


# baseline inference
def update_fn(t, y0): return model_update(
    t, y0, learned_baseline, model_name='baseline')


BNN_traj = solve_ivp(update_fn, t_span=[
    0, 20], y0=in_state, t_eval=t_obs, rtol=1e-12)

# %% Visualizing the accuracy of the prediction
fig = plt.figure(figsize=(6, 6), dpi=DPI)
ax1 = fig.add_subplot(211)
ax1.plot(t_obs, true_traj[:, 0], label='True', color='white')
ax1.plot(t_obs, gHNN_traj.y[0, :], label='gHNN', color='yellow')
ax1.plot(t_obs, BNN_traj.y[0, :], label='Baseline', color='red')
ax1.legend(loc='best')

ax2 = fig.add_subplot(212)
ax2.plot(true_traj[:, 0], true_traj[:, 1], label='True', color='white')
ax2.plot(gHNN_traj.y[0, :], gHNN_traj.y[1, :], label='gHNN', color='yellow')
ax2.plot(BNN_traj.y[0, :], BNN_traj.y[1, :], label='baseline', color='red')
ax2.legend(loc='best')

plt.show()

# %% Saving the data to file
np.savetxt(file_path/'gHNN_traj.csv', np.concatenate(
    (gHNN_traj.t.reshape(-1, 1), gHNN_traj.y.T), axis=1), fmt='%.6f', delimiter=',')
np.savetxt(file_path/'true_traj.csv', np.concatenate(
    (gHNN_traj.t.reshape(-1, 1), true_traj), axis=1), fmt='%.6f', delimiter=',')
np.savetxt(file_path/'Conventional_traj.csv', np.concatenate(
    (BNN_traj.t.reshape(-1, 1), BNN_traj.y.T), axis=1), fmt='%.6f', delimiter=',')
