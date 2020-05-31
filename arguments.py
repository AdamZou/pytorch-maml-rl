import argparse
import multiprocessing as mp
import torch
import os
args = None

def init():
    global args

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Train')

    parser.add_argument('--config', type=str, required=True,
        help='path to the configuration file.')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output-folder', type=str,
        help='name of the output folder')
    misc.add_argument('--seed', type=int, default=None,
        help='random seed')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')
    misc.add_argument('--determ_forward', type=bool, default=False,
        help='determ_forward')
    misc.add_argument('--rp_inner', type=bool, default=False,
        help='if true, use reparameterization mc in inner-update instead of vanilla')
    misc.add_argument('--prior_sigma', type=float, default=0.01,
        help='prior_sigma')
    misc.add_argument('--fix_sigma', type=float, default=-1.0,
        help='if negative, no fix')
    misc.add_argument('--meta_lr', type=float, default=0.001,
        help='meta learning rate')
    misc.add_argument('--meta_loss', type=str, default='maml',
        help='meta loss type')
    misc.add_argument('--meta_op', type=str, default='sgd',
        help='meta update algorithm type')
    misc.add_argument('--deter_b', type=bool, default=False,
        help='if true, set sigma_b to 0')
    misc.add_argument('--stop_grad', type=bool, default=False,
        help='if true, stop further grads when updating')
    misc.add_argument('--continuous', type=bool, default=True,
        help='if true, ')
    misc.add_argument('--sigma_lr', type=float, default=10.0,
        help='learning rate of sigma')
    misc.add_argument('--num_steps', type=int, default=1,
        help='number of inner steps')
    misc.add_argument('--sigma_trans', type=str, default='exp',
        help='log sigma transform function. exp or softplus')
    misc.add_argument('--output', type=str, default='train_results.pkl',
        help='name of the output folder')
    misc.add_argument('--num_batches', type=int, default=500,
        help='number of meta steps')
    misc.add_argument('--test_num_batches', type=int, default=10,
        help='number of test batches')
    misc.add_argument('--test_num_steps', type=int, default=3,
        help='number of test steps')
    misc.add_argument('--test_meta_batch_size', type=int, default=20,
        help='number of test meta batches')
    misc.add_argument('--test_output', type=str, default='test_results.pkl',
        help='name of the test output folder')


    args = parser.parse_args()
    print(args)
    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')


def test_init():
    global args

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML) - Test')


    parser.add_argument('--dir', type=str, required=True,
        help='path to the files')
    parser.add_argument('--config', type=str, default='config.json',
        help='path to the configuration file')
    parser.add_argument('--policy', type=str, default='policy.th',
        help='path to the policy checkpoint')

    # Evaluation
    evaluation = parser.add_argument_group('Evaluation')
    evaluation.add_argument('--num-batches', type=int, default=10,
        help='number of batches (default: 10)')
    evaluation.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch (default: 40)')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--output', type=str, default='results.npz',
        help='name of the output folder (default: maml)')
    misc.add_argument('--seed', type=int, default=1,
        help='random seed (default: 1)')
    misc.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: '
             '{0})'.format(mp.cpu_count() - 1))
    misc.add_argument('--use-cuda', action='store_true',
        help='use cuda (default: false, use cpu). WARNING: Full upport for cuda '
        'is not guaranteed. Using CPU is encouraged.')
    misc.add_argument('--determ_forward', type=bool, default=False,
        help='determ_forward')
    misc.add_argument('--prior_sigma', type=float, default=0.01,
        help='prior_sigma')
    misc.add_argument('--fix_sigma', type=float, default=-1.0,
        help='if negative, no fix')
    misc.add_argument('--num_steps', type=int, default=1,
        help='number of inner steps')
    misc.add_argument('--stop_grad', type=bool, default=False,
        help='if true, stop further grads when updating')
    misc.add_argument('--sigma_trans', type=str, default='exp',
        help='log sigma transform function. exp or softplus')


    args = parser.parse_args()
    args.config = os.path.join(args.dir , args.config)
    args.policy = os.path.join(args.dir , args.policy)
    args.output = os.path.join(args.dir , args.output)

    args.device = ('cuda' if (torch.cuda.is_available()
                   and args.use_cuda) else 'cpu')
