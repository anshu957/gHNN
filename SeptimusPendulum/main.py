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
import pathlib
from scipy.integrate import solve_ivp

# %% Global constants
TRAIN_TEST_RATIO = 0.8
DPI = 150

plt.style.use('dark_background')

# model can take three values ['gHNN', 'HNN', 'BNN'] for (generalized HNN, conventional HNN, baseline NN (vanilla MLP))
args = {'LR': 1e-03, 'dev': 'cpu', 'epochs': 50,
        'verbose': True, 'batch_size': 16, 'SEED': 957, 'model': 'gHNN'}

# %% loading full dataset
file_path = pathlib.Path('data')
dataset = np.loadtxt(file_path/'extracted_angles.csv',
                     dtype=float, delimiter=',')
print(f'Shape of loaded dataset : {dataset.shape}')


# %% Making train - test datasets
train_ds = dataset[:int(TRAIN_TEST_RATIO*len(dataset))]
test_ds = dataset[int(TRAIN_TEST_RATIO*len(dataset)):]

# input tensor for training data
train_x = torch.tensor(
    train_ds[:, 1:3], requires_grad=True, dtype=torch.float32)
# label for training data
train_Y = torch.tensor(train_ds[:, 2:], dtype=torch.float32)
# input tensor for test data
test_x = torch.tensor(test_ds[:, 1:3], requires_grad=True, dtype=torch.float32)
# label for test data
test_Y = torch.tensor(test_ds[:, 2:], dtype=torch.float32)

print('Shape of training inputs: {}'.format(train_x.shape))
print('Shape of training targets: {}'.format(train_Y.shape))
print('Shape of testing inputs: {}'.format(test_x.shape))
print('Shape of testing targets: {}'.format(test_Y.shape))


# %% Visualizing training and testing dataset
fig = plt.figure(figsize=(8, 6), dpi=DPI)
ax1 = fig.add_subplot(211)

theta = train_x[:, 0].detach().numpy()
theta_dot = train_x[:, 1].detach().numpy()
theta_ddot = train_Y[:, 1].detach().numpy()

ax1.plot(theta, 'b', label=r'$\theta$')
ax1.plot(theta_dot, 'r', label=r'$\dot{\theta}$')
ax1.set_xlabel('Time')
ax1.legend(loc='best')

ax2 = fig.add_subplot(212)
ax2.plot(theta, 'y', label=r'$\theta$')
ax2.plot(theta_dot, 'c', label=r'$\dot{\theta}$', alpha=0.8)
ax2.plot(theta_ddot, 'r', label=r'$\ddot{\theta}$', alpha=0.3)
ax2.set_xlabel('Time')
ax2.legend(loc='best')

plt.tight_layout()
plt.show()

# %% Let's train!

np.random.seed(args['SEED'])
torch.manual_seed(args['SEED'])

# select the model and uncomment the others [ HNN, BNN (conventional), iHNN (for gHNN)]
if args['model'] == 'gHNN':
    model = iHNN(d_in=2, activation_fn='Tanh', device='cpu')
elif args['model'] == 'HNN':
    model = HNN(d_in=2, activation_fn='Tanh', device='cpu')
else:
    model = BNN(d_in=2, activation_fn='Tanh', device='cpu')
optim = torch.optim.Adam(model.parameters(), args['LR'], weight_decay=1e-04)
stats = train(model, args, train_x, train_Y,
              test_x, test_Y, optim)

save_path = Path('trained_models')
filename = f"SP_EXP_{args['model']}_seed_{args['SEED']}_lr_{args['LR']}_bs_{args['batch_size']}_epochs_{args['epochs']}.pkl"
torch.save(model.state_dict(), save_path/filename)
