import maml_rl.envs
import gym
import torch
import json
import numpy as np
import pickle
from tqdm import trange

from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns

from collections import OrderedDict

def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config['env-kwargs'])
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    with open(args.policy, 'rb') as f:
        state_dict = torch.load(f, map_location=torch.device(args.device))
        policy.load_state_dict(state_dict)
    policy.share_memory()

    #print policy params
    params = OrderedDict(policy.named_parameters())
    for i in range(1, policy.num_layers):
        print('layer',i)
        print('weight_mu=',params['layer{0}.weight_mu'.format(i)], 'weight_log_sigma=',params['layer{0}.weight_log_sigma'.format(i)], 'bias_mu=',params['layer{0}.bias_mu'.format(i)], 'bias_log_sigma=',params['layer{0}.bias_log_sigma'.format(i)])
    print('layer_last')
    print('weight_mu=',params['mu.weight_mu'], 'weight_log_sigma=',params['mu.weight_log_sigma'],'bias_mu=', params['mu.bias_mu'], 'bias_log_sigma=',params['mu.bias_log_sigma'])
    if not arguments.args.fix_sigma:
        print('sigma=',params['sigma'])
    # Baseline
    baseline = LinearFeatureBaseline(get_input_size(env))

    # Sampler
    sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs=config['env-kwargs'],
                               batch_size=config['fast-batch-size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=args.seed,
                               num_workers=args.num_workers)

    logs = {'tasks': []}
    train_returns, valid_returns = [], []
    for batch in trange(args.num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        train_episodes, valid_episodes = sampler.sample(tasks,
                                                        num_steps=args.num_steps, #num_steps=config['num-steps'],
                                                        fast_lr=config['fast-lr'],
                                                        gamma=config['gamma'],
                                                        gae_lambda=config['gae-lambda'],
                                                        device=args.device)

        logs['tasks'].extend(tasks)
        train_returns.append(get_returns(train_episodes[0]))
        valid_returns.append(get_returns(valid_episodes))

        print('batch=',batch)
        #print('tasks=',tasks)
        print('train_returns=',np.mean(train_returns[-1],axis=0), np.mean(train_returns[-1]))
        print('valid_returns=',np.mean(valid_returns[-1],axis=0), np.mean(valid_returns[-1]))

    logs['train_returns'] = np.concatenate(train_returns, axis=0)
    logs['valid_returns'] = np.concatenate(valid_returns, axis=0)

    with open(args.output, 'wb') as f:
        np.savez(f, **logs)


if __name__ == '__main__':
    import arguments
    arguments.test_init()
    main(arguments.args)
