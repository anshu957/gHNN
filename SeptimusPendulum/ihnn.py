import torch
import sys
from tqdm import tqdm


class iHNN(torch.nn.Module):
    def __init__(self, d_in, activation_fn, device):
        super(iHNN, self).__init__()
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
        elif activation_fn == 'Softplus':
            print('Using Softplus ...')
            self.nonlinear_fn = torch.nn.Softplus()

        # Block for converting provided coordinates to canonical coordinates
        self.layer1 = torch.nn.Linear(d_in, 20)
        self.layers.append(self.layer1)

        self.layer2 = torch.nn.Linear(20, 20)
        self.layers.append(self.layer2)

        self.layer3 = torch.nn.Linear(20, 1)
        self.layers.append(self.layer3)
        # -------------------------------------------------

        # HNN block
        self.layer4 = torch.nn.Linear(d_in, 100)
        self.layers.append(self.layer4)

        self.layer5 = torch.nn.Linear(100, 100)
        self.layers.append(self.layer5)

        self.layer6 = torch.nn.Linear(100, 1, bias=None)
        self.layers.append(self.layer6)
        # -------------------------------------------------

        print(self.layers)

        # for i in range(len(self.layers)):
        #    torch.nn.init.orthogonal_(self.layers[i].weight)

    def to_canonical(self, x):
        y1 = self.nonlinear_fn(self.layer1(x))
        y2 = self.nonlinear_fn(self.layer2(y1))
        out = self.layer3(y2)
        return out

    def hnn(self, x):
        y1 = self.nonlinear_fn(self.layer4(x))
        y2 = self.nonlinear_fn(self.layer5(y1))
        # nn_fn = torch.nn.LeakyReLU(0.9)
        out = self.layer6(y2)
        return out

    def time_derivative(self, x, H):

        dH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

        return dH @ self.M.t()

    def forward(self, x):
        # set_trace()
        can_momentum = self.to_canonical(x)  # convert to canonical momentum
        # concatenate input canonical coordinates with learning canonical momenta
        can_coords = torch.cat((x[:, 0:1], can_momentum), dim=1)

        H = self.hnn(can_coords)
        return H, can_coords

    def permutation_tensor(self, n):
        M = None
        M = torch.eye(n)
        M = torch.cat([M[n // 2:], -M[: n // 2]])

        return M


def getInverseJacobian(net2, x):
    '''
        Take input point x and forward it through 1st block to calculate jacobian of transformation
    '''

    jac = torch.zeros(size=(2, 2))

    x = x.reshape(1, 2)
    y = net2.to_canonical(x)

    jac[0, :] = torch.tensor([1.0, 0.0], dtype=torch.float32)
    jac[1, :] = torch.autograd.grad(y[0], x, create_graph=True)[0]
    # print(x)
    # print('----------')
    # print('Jacobian : {}'.format(self.jac))
    # print(net2.layer1.weight)

    # Getting inverse of jacobian using Penrose pseudo-inverse
    try:
        jac_inverse = torch.pinverse(jac)
    except RuntimeError:
        print(jac)
        # print(net2.layer2.weight)
    #    torch.save(net2.state_dict(), 'tmp_saved_model.tar')

    if torch.isnan(jac_inverse).any():
        print('Nan encountered in Jacobian !')
        sys.exit(0)

    return jac_inverse


def transformVectorField(model, x, can_vector_field):
    '''
        Transforms the incoming vector field coming out of the network to vector field in original coordinates
        using inverse of jacobian matrix from 1st block of neural network.

        Input: Tensor coming from output of the network (vector filed) -- (n_batches X 4)
        Output: Tensor with transformed vector field -- (n_batches X 4)

    '''
    transformed_vec_field = torch.zeros_like(x)
    for i in range(can_vector_field.shape[0]):
        jac_inverse = getInverseJacobian(model, x[i])
        # jac_inverse = torch.tensor([ [x[i][0], 0.0], [0.0, x[i][1]] ])
        trans_vf = jac_inverse @ can_vector_field[i]
        # trans_vf = torch.mul(torch.exp(can_coords[i]),can_vector_field[i])
        transformed_vec_field[i, :] = trans_vf

    return transformed_vec_field


def train(model, args, train_x, train_Y, test_x, test_Y, optim):
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
            H, can_coords = model.forward(x[ixs])
            can_coords_dot = model.time_derivative(can_coords, H)

            x_dot_hat = transformVectorField(
                model, x[ixs], can_coords_dot)

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
            H_test, can_coords_test = model.forward(test_x[test_ixs])
            can_coords_dot_test = model.time_derivative(
                can_coords_test, H_test)
            x_test_dot_hat = transformVectorField(
                model, test_x[test_ixs], can_coords_dot_test)

            test_loss = L2_loss(x_test_dot_hat, test_x_dot[test_ixs])
            test_loss_epoch += test_loss.item()

        # logging
        stats["train_loss"].append(loss_epoch/no_batches)
        stats["test_loss"].append(test_loss_epoch/no_batches)
        if args['verbose']:
            print(
                "epoch {}, train_loss {:.4e}, test_loss {:.4e}, grad norm {:.4e}, grad std {:.4e}, H {:.4e}".format(
                    epoch,
                    loss_epoch/no_batches,
                    test_loss_epoch/no_batches,
                    grad @ grad,
                    grad.std(),
                    H[0][0],
                )
            )

    return stats
