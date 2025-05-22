import json
import argparse
import os
from trainer import train
import wandb


def main():

    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) # Converting argparse Namespace to a dict.
    args.update(param) # Add parameters from json

    wandb.login()
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'Summary/Average Accuracy (CNN)',
            'goal': 'maximize'
        },
        'parameters': {
            'init_lr': {
                'values': [0.001, 0.003]
            }
        },
    }
    sweep_id = wandb.sweep(sweep_config, project="Propmt-E")

    def sweep_train():
        wandb.init(name=f'{args["model_name"]}')
        args.update(wandb.config)
        train(args)

    wandb.agent(sweep_id, function=sweep_train, count=2)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/clip4l2p_inr.json',
                        help='Json file of settings.')
    return parser

if __name__ == '__main__':
    main()
