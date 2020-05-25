import gym
import torch
import json
import os
import yaml
from tqdm import trange
import numpy as np
import pickle

import maml_rl.envs
from maml_rl.metalearners import MAMLTRPO
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.samplers import MultiTaskSampler
from maml_rl.utils.helpers import get_policy_for_env, get_input_size
from maml_rl.utils.reinforcement_learning import get_returns

from collections import OrderedDict

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        policy_filename = os.path.join(args.output_folder, 'policy.th')
        config_filename = os.path.join(args.output_folder, 'config.json')

        with open(config_filename, 'w') as f:
            config.update(vars(args))
            json.dump(config, f, indent=2)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    env = gym.make(config['env-name'], **config['env-kwargs'])
    env.close()

    # Policy
    policy = get_policy_for_env(env,
                                hidden_sizes=config['hidden-sizes'],
                                nonlinearity=config['nonlinearity'])
    policy.share_memory()

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

    metalearner = MAMLTRPO(policy,
                           fast_lr=config['fast-lr'],
                           first_order=config['first-order'],
                           device=args.device)

    num_iterations = 0
    result={'train_returns':[],'valid_returns':[]}

    for batch in trange(args.num_batches):
    #for batch in trange(config['num-batches']):
    #for batch in range(config['num-batches']):
        tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
        # inner-update
        futures = sampler.sample_async(tasks,
                                       num_steps=args.num_steps, #num_steps=config['num-steps'],
                                       fast_lr=config['fast-lr'],
                                       gamma=config['gamma'],
                                       gae_lambda=config['gae-lambda'],
                                       device=args.device)
        # meta-update
        logs = metalearner.step(*futures,
                                max_kl=config['max-kl'],
                                cg_iters=config['cg-iters'],
                                cg_damping=config['cg-damping'],
                                ls_max_steps=config['ls-max-steps'],
                                ls_backtrack_ratio=config['ls-backtrack-ratio'])

        train_episodes, valid_episodes = sampler.sample_wait(futures)
        num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
        num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)

        train_returns=get_returns(train_episodes[0])
        valid_returns=get_returns(valid_episodes)
        logs.update(tasks=tasks,
                    num_iterations=num_iterations,
                    train_returns=train_returns,
                    valid_returns=valid_returns)

        print('batch=',batch)
        #print('tasks=',tasks)
        print('train_returns=',np.mean(train_returns,axis=0), np.mean(train_returns))
        print('valid_returns=',np.mean(valid_returns,axis=0), np.mean(valid_returns))
        result['train_returns'].append(np.mean(train_returns))
        result['valid_returns'].append(np.mean(valid_returns))
        #print policy params sigma
        if arguments.args.fix_sigma < 0:
            params = OrderedDict(policy.named_parameters())
            print('sigma=',params['sigma'])

        #print(logs)
        # Save policy
        if args.output_folder is not None:
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)

    # save logs
    with open(os.path.join(args.output_folder,args.output), 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #print policy params
    params = OrderedDict(policy.named_parameters())
    for i in range(1, policy.num_layers):
        print('layer',i)
        print('weight_mu=',params['layer{0}.weight_mu'.format(i)], 'weight_log_sigma=',params['layer{0}.weight_log_sigma'.format(i)], 'bias_mu=',params['layer{0}.bias_mu'.format(i)], 'bias_log_sigma=',params['layer{0}.bias_log_sigma'.format(i)])
    print('layer_last')
    print('weight_mu=',params['mu.weight_mu'], 'weight_log_sigma=',params['mu.weight_log_sigma'],'bias_mu=', params['mu.bias_mu'], 'bias_log_sigma=',params['mu.bias_log_sigma'])
    if arguments.args.fix_sigma < 0:
        print('sigma=',params['sigma'])


    # metatest after metatrain
    test_result = []

    for batch in trange(args.test_num_batches):
        tasks = sampler.sample_tasks(num_tasks=args.test_meta_batch_size)
        batch_result=[]

        for step in range(args.test_num_steps):
            train_episodes, valid_episodes = sampler.sample(tasks,
                                                            num_steps=(step+1), #num_steps=config['num-steps'],
                                                            fast_lr=config['fast-lr'],
                                                            gamma=config['gamma'],
                                                            gae_lambda=config['gae-lambda'],
                                                            device=args.device)
            if step==0:
                batch_result.append(np.mean(get_returns(train_episodes[0])))
            batch_result.append(np.mean(get_returns(valid_episodes)))
        test_result.append(batch_result)

    with open(os.path.join(args.output_folder,args.test_output), 'wb') as handle:
        pickle.dump(test_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    test_plot=np.mean(test_result,axis=0)
    with open(os.path.join(args.output_folder,'test_plot.pkl'), 'wb') as handle:
        pickle.dump(test_plot, handle, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':
    import arguments
    arguments.init()
    main(arguments.args)
