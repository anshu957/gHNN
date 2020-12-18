import torch
import numpy as np
from tqdm import tqdm


class HNN(torch.nn.Module):
    def __init__(self, d_in,  activation_fn, device):
        super(HNN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.M = self.permutation_tensor(d_in).to(device)

        if activation_fn == "Tanh":
            print("Using Tanh()...")
            self.nonlinear_fn = torch.nn.Tanh()
        elif activation_fn == "ReLU":
            print("Using ReLU()...")
            self.nonlinear_fn = torch.nn.ReLU()
        elif activation_fn == 'Sigmoid':
            print('Using Sigmoid ...')
            self.nonlinear_fn = torch.nn.Sigmoid()

        # HNN Block
        self.layer1 = torch.nn.Linear(d_in, 100)
        self.layers.append(self.layer1)

        self.layer2 = torch.nn.Linear(100, 100)
        self.layers.append(self.layer2)

        self.layer3 = torch.nn.Linear(100, 1, bias=None)
        self.layers.append(self.layer3)

    def forward(self, x):
        y = self.nonlinear_fn(self.layer1(x))
        y1 = self.nonlinear_fn(self.layer2(y))

        return self.layer3(y1)

    def time_derivative(self, x):
        H = self.forward(x)
        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

        return dH @ self.M.t(), H

    def permutation_tensor(self, n):
        M = None
        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[: n // 2]])

        return M


def trainHNN(model, args, train_x, train_Y, test_x, test_Y, optim):

    x = train_x
    x_dot = train_Y
    test_x = test_x
    test_x_dot = test_Y

    L2_loss = torch.nn.MSELoss()

    stats = {"train_loss": [], "test_loss": []}

    no_batches = int(x.shape[0]/args["batch_size"])

    print('No. of batches : {}'.format(no_batches))

    for epoch in tqdm(range(args["epochs"])):

        loss_epoch = 0.0
        for batch in range(no_batches):

            optim.zero_grad()

            ixs = torch.randperm(x.shape[0])[: args["batch_size"]]
            x_dot_hat, H = model.time_derivative(x[ixs])

            loss = L2_loss(x_dot[ixs], x_dot_hat)

            loss.backward()

            grad = torch.cat([p.grad.flatten()
                              for p in model.parameters()]).clone()

            optim.step()

            loss_epoch += loss.item()

        # run test data
        test_ixs = torch.randperm(test_x.shape[0])[: args["batch_size"]]

        test_x_dot_hat, H_test = model.time_derivative(test_x[test_ixs])

        test_loss = L2_loss(test_x_dot[test_ixs], test_x_dot_hat)

        # logging
        stats["train_loss"].append(loss_epoch/no_batches)
        stats["test_loss"].append(test_loss.item())

        if args["verbose"]:
            print(
                "epoch {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}, H {:.4e}".format(
                    epoch+1,
                    loss_epoch/no_batches,
                    test_loss.item(),
                    grad @ grad,
                    grad.std(),
                    H[0][0],
                )
            )

    return stats
