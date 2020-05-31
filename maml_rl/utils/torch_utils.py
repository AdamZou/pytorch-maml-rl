import torch
import numpy as np

from torch.distributions import Categorical, Independent, Normal
from torch.nn.utils.convert_parameters import _check_param_device
import torch.nn as nn

from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence
from collections import OrderedDict
import math
import arguments




def deter_sigma(params):
    updated_params = OrderedDict()
    for (name, param) in params.items():
        if 'log_sigma' in name:
            updated_params[name] = torch.log(torch.exp(param) * 1e-9)
            #updated_params[name] = math.log( 1e-9)
        else:
            updated_params[name] = param

    return updated_params


def stopgrad_params(params):
    updated_params = OrderedDict()
    for (name, param) in params.items():
        updated_params[name] = param.detach()

    return updated_params


def get_dist(weight_mu, weight_log_sigma, bias_mu, bias_log_sigma):

    #weight = Independent(MultivariateNormal(loc=weight_mu, scale_tril=torch.diag_embed(torch.exp(weight_log_sigma))) ,1)
    #bias = Independent(MultivariateNormal(loc=bias_mu, scale_tril=torch.diag(torch.exp(bias_log_sigma))) ,1)
    if arguments.args.sigma_trans=='exp':
        transform = torch.exp
    if arguments.args.sigma_trans=='softplus':
        transform = nn.Softplus()

    weight = MultivariateNormal(loc=weight_mu, scale_tril=torch.diag_embed(transform(weight_log_sigma)))
    bias = MultivariateNormal(loc=bias_mu, scale_tril=torch.diag(transform(bias_log_sigma)))

    return weight, bias


def KL(params_a,params_b,num_layers):

    kl = 0

    if not arguments.args.continuous:
        num_layers += 1
    for i in range(1, num_layers):
        params = params_a
        weight_a, bias_a = get_dist(params['layer{0}.weight_mu'.format(i)], params['layer{0}.weight_log_sigma'.format(i)], params['layer{0}.bias_mu'.format(i)], params['layer{0}.bias_log_sigma'.format(i)])
        params = params_b
        weight_b, bias_b = get_dist(params['layer{0}.weight_mu'.format(i)], params['layer{0}.weight_log_sigma'.format(i)], params['layer{0}.bias_mu'.format(i)], params['layer{0}.bias_log_sigma'.format(i)])
        kl += torch.sum(kl_divergence(weight_a,weight_b)) + torch.sum(kl_divergence(bias_a,bias_b))


    #print(arguments.args.contin)
    if arguments.args.continuous:
        params = params_a
        weight_a, bias_a = get_dist(params['mu.weight_mu'], params['mu.weight_log_sigma'], params['mu.bias_mu'], params['mu.bias_log_sigma'])
        if arguments.args.fix_sigma < 0:
            sigma_a = params['sigma']
        params = params_b
        weight_b, bias_b = get_dist(params['mu.weight_mu'], params['mu.weight_log_sigma'], params['mu.bias_mu'], params['mu.bias_log_sigma'])
        if arguments.args.fix_sigma < 0:
            sigma_b = params['sigma']
        kl += torch.sum(kl_divergence(weight_a,weight_b)) + torch.sum(kl_divergence(bias_a,bias_b))
        if arguments.args.fix_sigma < 0:
            kl += torch.sum((sigma_a - sigma_b) ** 2) * arguments.args.sigma_lr


    return kl


def L2(params_a,params_b,num_layers):

    l2 = 0
    for (name, param) in params_a.items():
        l2 += torch.sum((params_a[name] - params_b[name]) ** 2)

    return l2


def weighted_mean(tensor, lengths=None):
    if lengths is None:
        return torch.mean(tensor)
    if tensor.dim() < 2:
        raise ValueError('Expected tensor with at least 2 dimensions '
                         '(trajectory_length x batch_size), got {0}D '
                         'tensor.'.format(tensor.dim()))
    for i, length in enumerate(lengths):
        tensor[length:, i].fill_(0.)

    extra_dims = (1,) * (tensor.dim() - 2)
    lengths = torch.as_tensor(lengths, dtype=torch.float32)

    out = torch.sum(tensor, dim=0)
    out.div_(lengths.view(-1, *extra_dims))

    return out

def weighted_normalize(tensor, lengths=None, epsilon=1e-8):
    mean = weighted_mean(tensor, lengths=lengths)
    out = tensor - mean.mean()
    for i, length in enumerate(lengths):
        out[length:, i].fill_(0.)

    std = torch.sqrt(weighted_mean(out ** 2, lengths=lengths).mean())
    out.div_(std + epsilon)

    return out

def detach_distribution(pi):
    if isinstance(pi, Independent):
        distribution = Independent(detach_distribution(pi.base_dist),
                                   pi.reinterpreted_batch_ndims)
    elif isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    elif isinstance(pi, Normal):
        distribution = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
    else:
        raise NotImplementedError('Only `Categorical`, `Independent` and '
                                  '`Normal` policies are valid policies. Got '
                                  '`{0}`.'.format(type(pi)))
    return distribution

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (tuple, list)):
        return np.stack([to_numpy(t) for t in tensor], axis=0)
    else:
        raise NotImplementedError()

def vector_to_parameters(vector, parameters):
    param_device = None

    pointer = 0
    for param in parameters:
        param_device = _check_param_device(param, param_device)

        num_param = param.numel()
        param.data.copy_(vector[pointer:pointer + num_param]
                         .view_as(param).data)

        pointer += num_param
