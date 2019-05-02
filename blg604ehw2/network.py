""" Neural Networks for agents """

import torch


class FcNet(torch.nn.Module):
    """ Fully connected feature network """

    def __init__(self, nobs, n_neuron=128):
        super().__init__()
        self.fc_1 = torch.nn.Linear(nobs, n_neuron)
        self.fc_2 = torch.nn.Linear(n_neuron, n_neuron)
        self._init_weights()

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc_1(x))
        x = torch.nn.functional.relu(self.fc_2(x))
        return x

    # Optional
    def _init_weights(self):
        """Parameter initialization"""
        pass


class Cnn(torch.nn.Module):
    def __init__(self, in_channels=4, out_feature=512):
        """ Convolutional feature network similar to the
        DeepMind's Nature paper.
        """
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, out_feature)

    def forward(self, x):
        #import pdb;pdb.set_trace()
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.fc4(x.view(x.size(0), -1)))
        return x

    # Optional
    def _init_weights(self):
        """Parameter initialization"""
        pass

class SimpleHead(torch.nn.Module):
    """ Linear output head """

    def __init__(self, nact, n_in):
        super().__init__()
        self.head = torch.nn.Linear(n_in, nact)
        self._init_weights()

    def forward(self, x):
        return self.head(x)

    # Optional
    def _init_weights(self):
        """Parameter initialization"""
        pass

class DuelingHead(torch.nn.Module):
    """ Dueling output head
        Advantage and value functions are used for the
        final q values.

            Q(s, a) = V(s) + A(s, a) - sum_i(A(s, a_i))/n(A)

        As the number of actions is increased, dueling
        head performs better than the linear one.
    """

    def __init__(self, nact, n_in):
        super().__init__()
        ### YOUR CODE HERE ###
        self.vhead = torch.nn.Linear(n_in, 1)
        self.ahead = torch.nn.Linear(n_in, nact)
        self._init_weights()
        ###       END      ###

    def forward(self, x):
        ### YOUR CODE HERE ###
        advantage = self.ahead(x)
        return self.vhead(x) + self.ahead(x) - advantage.mean()
        ###       END      ###

    # Optional
    def _init_weights(self):
        """Parameter initialization"""
        pass

class Network(torch.nn.Module):
    """ Merges feature and head networks """
    def __init__(self, feature_net, head_net):
        super().__init__()
        self.feature_net = feature_net
        self.head_net = head_net

    def forward(self, x, *args):
        #import pdb;pdb.set_trace()
        x = self.feature_net(x)
        x = self.head_net(x, *args)
        return x

# --------- A3C Networks ----------

# You may discard below and implement heads without sequential
#  elements if you want to.


class DiscreteDistHead(torch.nn.Module):
    """ Discrete Distribution generating sequential head """
    def __init__(self, in_feature, n_out):
        super().__init__()
        self.dist_gru = torch.nn.GRUCell(in_feature, 128)
        self.dist_head = torch.nn.Linear(128, n_out)

        self.value_gru = torch.nn.GRUCell(in_feature, 128)
        self.value_head = torch.nn.Linear(128, 1)

    def forward(self, x, h):
        h_a, h_c = h
        h_a = self.dist_gru(x, h_a)
        logits = self.dist_head(h_a)
        dist = torch.distributions.Categorical(logits=logits)

        h_c = self.value_gru(x, h_c)
        value = self.value_head(h_c)
        return dist, value, (h_a, h_c)


class ContinuousDistHead(torch.nn.Module):
    """ Continuous Distribution generating sequential head """
    def __init__(self, in_feature, n_out):
        super().__init__()
        self.dist_gru = torch.nn.GRUCell(in_feature, 128)
        self.dist_mu = torch.nn.Linear(128, n_out)
        self.dist_sigma = torch.nn.Linear(128, n_out)

        self.value_gru = torch.nn.GRUCell(in_feature, 128)
        self.value_head = torch.nn.GRUCell(128, 1)

    def forward(self, x, h):
        #import pdb;pdb.set_trace()
        h_a, h_c = h
        h_a = self.dist_gru(x, h_a)
        mu = self.dist_mu(h_a)
        sigma = torch.nn.functional.softplus(self.dist_sigma(h_a))
        dist = torch.distributions.Normal(loc=mu, scale=sigma)

        h_c = self.value_gru(x, h_c)
        value = self.value_head(h_c)
        return dist, value, (h_a, h_c)