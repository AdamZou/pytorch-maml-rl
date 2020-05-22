import numpy as np
import torch
from maml_rl.utils.torch_utils import weighted_mean, to_numpy
from maml_rl.utils.optimization import conjugate_gradient
import arguments
from torch.nn.utils.convert_parameters import parameters_to_vector
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       to_numpy, vector_to_parameters, KL, stopgrad_params, deter_sigma, L2)
from collections import OrderedDict
from torch.distributions.kl import kl_divergence

def value_iteration(transitions, rewards, gamma=0.95, theta=1e-5):
    rewards = np.expand_dims(rewards, axis=2)
    values = np.zeros(transitions.shape[0], dtype=np.float32)
    delta = np.inf
    while delta >= theta:
        q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
        new_values = np.max(q_values, axis=1)
        delta = np.max(np.abs(new_values - values))
        values = new_values

    return values

def value_iteration_finite_horizon(transitions, rewards, horizon=10, gamma=0.95):
    rewards = np.expand_dims(rewards, axis=2)
    values = np.zeros(transitions.shape[0], dtype=np.float32)
    for k in range(horizon):
        q_values = np.sum(transitions * (rewards + gamma * values), axis=2)
        values = np.max(q_values, axis=1)

    return values

def get_returns(episodes):
    return to_numpy([episode.rewards.sum(dim=0) for episode in episodes])

def reinforce_loss(policy, episodes, params=None):
    pi = policy(episodes.observations.view((-1, *episodes.observation_shape)),
                params=params)

    log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape)))
    log_probs = log_probs.view(len(episodes), episodes.batch_size)

    losses = -weighted_mean(log_probs * episodes.advantages,
                            lengths=episodes.lengths)

    return losses.mean()

#################trpo

def hessian_vector_product_fn(params, kl, damping=1e-2):
    grads = torch.autograd.grad(kl,
                                params.values(),
                                create_graph=True)
    flat_grad_kl = parameters_to_vector(grads)

    def _product(vector, retain_graph=True):
        grad_kl_v = torch.dot(flat_grad_kl, vector)
        grad2s = torch.autograd.grad(grad_kl_v,
                                     params.values(),
                                     retain_graph=retain_graph)
        flat_grad2_kl = parameters_to_vector(grad2s)

        return flat_grad2_kl + damping * vector
    return _product

# may need modification when adaptation steps are more than one. but we stick to one-step adaptation for now.
def surrogate_loss(policy,params ,episodes, old_pi=None):

    with torch.set_grad_enabled(old_pi is None):
        pi = policy(episodes.observations, params=params)

        if old_pi is None:
            old_pi = detach_distribution(pi)

        log_ratio = (pi.log_prob(episodes.actions)
                     - old_pi.log_prob(episodes.actions))
        ratio = torch.exp(log_ratio)

        kls = weighted_mean(kl_divergence(pi, old_pi),
                            lengths=episodes.lengths)

        losses = -weighted_mean(ratio * episodes.advantages,lengths=episodes.lengths)

    return losses.mean(), kls.mean(), old_pi

# update for one meta batch
def trpo_update(policy, old_params,future, old_pi=None,max_kl=1e-3,
                cg_iters=10,
                cg_damping=1e-2,
                ls_max_steps=10,
                ls_backtrack_ratio=0.5):

    #num_tasks = len(future)
    #logs = {}

    params = OrderedDict()
    for (name, param) in old_params.items():
        params[name] = param.clone()

    # Compute the surrogate loss
    old_loss, old_kl, old_pi =  surrogate_loss(policy, params,future, old_pi=None)

    #logs['loss_before'] = to_numpy(old_losses)
    #logs['kl_before'] = to_numpy(old_kls)

    #old_loss = sum(old_losses) / num_tasks
    grads = torch.autograd.grad(old_loss,
                                params.values(),
                                retain_graph=True)
    grads = parameters_to_vector(grads)

    # Compute the step direction with Conjugate Gradient

    #old_kl = sum(old_kls) / num_tasks
    hessian_vector_product = hessian_vector_product_fn(params, old_kl,
                                                         damping=cg_damping)
    stepdir = conjugate_gradient(hessian_vector_product,
                                 grads,
                                 cg_iters=cg_iters)

    # Compute the Lagrange multiplier
    shs = 0.5 * torch.dot(stepdir,
                          hessian_vector_product(stepdir, retain_graph=False))
    lagrange_multiplier = torch.sqrt(shs / max_kl)

    step = stepdir / lagrange_multiplier



    # Save the old parameters
    old_params = parameters_to_vector(params.values())

    # Line search
    step_size = 1.0
    for _ in range(ls_max_steps):
        vector_to_parameters(old_params - step_size * step,
                             params.values())

        loss, kl, _ = surrogate_loss(policy, params,future, old_pi=old_pi)

        improve = loss - old_loss
        #kl = sum(kls) / num_tasks
        if (improve.item() < 0.0) and (kl.item() < max_kl):
            #logs['loss_after'] = to_numpy(losses)
            #logs['kl_after'] = to_numpy(kls)
            break
        step_size *= ls_backtrack_ratio
    else:
        vector_to_parameters(old_params, params.values())



    return params
