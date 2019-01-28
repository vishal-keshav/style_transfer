"""
Project Name: Style-Transfer
Author: Vishal Keshav
"""

import argparse

from core import dataset_preprocessing as dp
from core import hyperparameter_search as hp
from core import model_trainer as mt
from core import evaluator as e
from core import generate_visualization as gv

program_name = "style_transfer"
program_details = "Implementing all style-transfer algorithms"

def argument_parser():
    parser = argparse.ArgumentParser(description=program_name + " : " + program_details)
    parser.add_argument('--phase', default='all', type=str, help='Phase')
    parser.add_argument('--dataset', default='MNIST', type=str, help='Dataset')
    parser.add_argument('--download', default=False, type=bool, help='Download dataset?')
    parser.add_argument('--preprocess', default=False, type=bool, help='Preprocess dataset?')
    parser.add_argument('--create_model', default=False, type=bool, help='Create Model from trained data?')
    parser.add_argument('--view', default=False, type=bool, help='View Sample dataset?')
    parser.add_argument('--viewtype', default='model', type=str, help='Type of visualization?')
    parser.add_argument('--notify', default=False, type=bool, help='Training notification?')
    parser.add_argument('--model', default=1, type=int, help='Model Version')
    parser.add_argument('--param', default=1, type=int, help='Hyper-parameter Version')
    args = parser.parse_args()
    return args

def main():
    args = argument_parser()
    if args.phase == 'dataset' or args.phase == 'all':
        dp.execute(args)
    if args.phase == 'param_search':
        hp.execute(args)
    if args.phase == 'train' or args.phase == 'all':
        mt.execute(args)
    if args.phase == 'evaluate' or args.phase == 'all':
        e.execute(args)
    if args.phase == 'visualize' or args.phase == 'all':
        gv.execute(args)

if __name__ == "__main__":
    main()
