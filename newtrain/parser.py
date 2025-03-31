from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import tools.data_manager as data_manager

def defParser():
    parser = argparse.ArgumentParser(description='Train AP3D')
    # Datasets
    parser.add_argument('--root', type=str, default='/home/guxinqian/data/')
    parser.add_argument('-d', '--dataset', type=str, default='mars',
                        choices=data_manager.get_names())
    parser.add_argument('-j', '--workers', default=4, type=int)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    # Augment
    parser.add_argument('--seq_len', type=int, default=4, 
                        help="number of images to sample in a tracklet")
    parser.add_argument('--sample_stride', type=int, default=8, 
                        help="stride of images to sample in a tracklet")
    # Optimization options
    parser.add_argument('--max_epoch', default=240, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--train_batch', default=32, type=int)
    parser.add_argument('--test_batch', default=32, type=int)
    parser.add_argument('--lr', default=0.0003, type=float)
    parser.add_argument('--stepsize', default=[60, 120, 180], nargs='+', type=int,
                        help="stepsize to decay learning rate")
    parser.add_argument('--gamma', default=0.1, type=float,
                        help="learning rate decay")
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    parser.add_argument('--margin', type=float, default=0.3, 
                        help="margin for triplet loss")
    parser.add_argument('--distance', type=str, default='cosine', 
                        help="euclidean or cosine")
    parser.add_argument('--num_instances', type=int, default=4, 
                        help="number of instances per identity")
    # Architecture
    parser.add_argument('-a', '--arch', type=str, default='ap3dres50', 
                        help="ap3dres50, ap3dnlres50")
    # Miscs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--eval_step', type=int, default=10)
    parser.add_argument('--start_eval', type=int, default=0, 
                        help="start to evaluate after specific epoch")
    parser.add_argument('--save_dir', type=str, default='log-mars-ap3d')
    parser.add_argument('--use_cpu', action='store_true', help="use cpu")
    parser.add_argument('--gpu', default='0, 1', type=str, 
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    return parser.parse_args()
