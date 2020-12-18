import torch
from tqdm import tqdm


class BNN(torch.nn.Module):
    def __init__(self, d_in, activation_fn, device):
        super(BNN, self).__init__()
        self.layers = torch.nn.ModuleList()

        if activation_fn == "Tanh":
            print("Using Tanh()...")
            self.nonlinear_fn = torch.nn.Tanh()
        elif activation_fn == "ReLU":
            print("Using ReLU()...")
            self.nonlinear_fn = torch.nn.ReLU()
        elif activation_fn == 'Sigmoid':
            print('Using Sigmoid ...')
            self.nonlinear_fn = torch.nn.Sigmoid()

        # Classic MLP
        self.layer1 = torch.nn.Linear(d_in, 100)
        self.layers.append(self.layer1)

        self.layer2 = torch.nn.Linear(100, 100)
        self.layers.append(self.layer2)

        self.layer3 = torch.nn.Linear(100, d_in, bias=None)
        self.layers.append(self.layer3)

        print(self.layers)

        for i in range(len(self.layers)):
            torch.nn.init.orthogonal_(self.layers[i].weight)

    def forward(self, x):
        y1 = self.nonlinear_fn(self.layer1(x))
        y2 = self.nonlinear_fn(self.layer2(y1))

        out = self.layer3(y2)
        return out


def trainBS(model, args, train_x, train_Y, test_x, test_Y, optim):
    x = train_x.to(args['dev'])
    x_dot = train_Y.to(args['dev'])
    test_x = test_x.to(args['dev'])
    test_x_dot = test_Y.to(args['dev'])
    L2_loss = torch.nn.MSELoss()

    stats = {"train_loss": [], "test_loss": []}
    no_batches = int(x.shape[0]/args['batch_size'])

    for epoch in tqdm(range(args['epochs'])):
        loss_epoch = 0.0
        test_loss_epoch = 0.0
        for batch in range(no_batches):

            optim.zero_grad()
            # net2 = copy.deepcopy(model)
            ixs = torch.randperm(x.shape[0])[: args['batch_size']]
            x_dot_hat = model.forward(x[ixs])
            loss = L2_loss(x_dot_hat, x_dot[ixs])

            #loss = loss_traj + loss_jacobian  + 1e-03*(1.0/torch.abs(H.mean())) #+ 0.1/(det_jac) #+ loss_T #

            loss.backward()

            grad = torch.cat([p.grad.flatten()
                              for p in model.parameters()]).clone()
            #print(f'Batch Gradient = {grad}')
            optim.step()
            loss_epoch += loss.item()

            # run test data
            test_ixs = torch.randperm(test_x.shape[0])[: args['batch_size']]
            x_test_dot_hat = model.forward(test_x[test_ixs])

            test_loss = L2_loss(x_test_dot_hat, test_x_dot[test_ixs])
            test_loss_epoch += test_loss.item()

        # logging
        stats["train_loss"].append(loss_epoch/no_batches)
        stats["test_loss"].append(test_loss_epoch/no_batches)
        if args['verbose']:
            print(
                "epoch {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}".format(
                    epoch,
                    loss_epoch/no_batches,
                    test_loss_epoch/no_batches,
                    grad @ grad,
                    grad.std()
                )
            )

    return stats
