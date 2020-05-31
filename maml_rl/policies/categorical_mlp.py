import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torchbnn as bnn
from collections import OrderedDict
from maml_rl.policies.policy import Policy, weight_init

import arguments

class CategoricalMLPPolicy(Policy):
    """Policy network based on a multi-layer perceptron (MLP), with a
    `Categorical` distribution output. This policy network can be used on tasks
    with discrete action spaces (eg. `TabularMDPEnv`). The code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/policies/maml_minimal_categorical_mlp_policy.py
    """
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=(),
                 nonlinearity=F.relu):
        super(CategoricalMLPPolicy, self).__init__(input_size=input_size,
                                                   output_size=output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1

        layer_sizes = (input_size,) + hidden_sizes + (output_size,)
        '''
        for i in range(1, self.num_layers + 1):
            self.add_module('layer{0}'.format(i),
                            nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
        '''
        for i in range(1, self.num_layers + 1):
            self.add_module('layer{0}'.format(i),
                            bnn.BayesLinear(prior_mu=0, prior_sigma=arguments.args.prior_sigma,in_features=layer_sizes[i - 1], out_features=layer_sizes[i]))

                            

        #self.apply(weight_init)
    def get_weight(self, weight_mu, weight_log_sigma, bias_mu, bias_log_sigma):
        # global args

        if arguments.args.sigma_trans=='exp':
            transform = torch.exp
        if arguments.args.sigma_trans=='softplus':
            transform = nn.Softplus()


        if arguments.args.determ_forward:
            weight = weight_mu + transform(weight_log_sigma) * 1e-9
            bias = bias_mu + transform(bias_log_sigma) * 1e-9
        else:
            weight = weight_mu + transform(weight_log_sigma) * torch.randn_like(weight_log_sigma)
            bias = bias_mu + transform(bias_log_sigma) * torch.randn_like(bias_log_sigma)

        return weight, bias

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        '''
        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output,
                              weight=params['layer{0}.weight'.format(i)],
                              bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)

        logits = F.linear(output,
                          weight=params['layer{0}.weight'.format(self.num_layers)],
                          bias=params['layer{0}.bias'.format(self.num_layers)])
        '''

        ###########
        output = input
        for i in range(1, self.num_layers):
            weight, bias = self.get_weight(params['layer{0}.weight_mu'.format(i)], params['layer{0}.weight_log_sigma'.format(i)], params['layer{0}.bias_mu'.format(i)], params['layer{0}.bias_log_sigma'.format(i)])
            output = F.linear(output,
                              weight=weight,
                              bias=bias)
            output = self.nonlinearity(output)

        i = self.num_layers
        weight, bias = self.get_weight(params['layer{0}.weight_mu'.format(i)], params['layer{0}.weight_log_sigma'.format(i)], params['layer{0}.bias_mu'.format(i)], params['layer{0}.bias_log_sigma'.format(i)])
        logits = F.linear(output,
                      weight=weight,
                      bias=bias)

        ###########

        return Categorical(logits=logits)
