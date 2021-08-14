#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merges all observations dumped inside data/simulations
"""
import os
import glob
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Save observations.')
parser.add_argument('--exp', help='Experiment ID.', type=str)
parser.add_argument('--path', default='/Users/naveenmysore/Documents/data/csdi_data')
args = parser.parse_args()

observations = None
flag = False
for name in glob.glob(os.path.join(os.getcwd(), 'simulations', 'data', 'observations_*.csv')):
    print(f'*** Reading {name}')
    if not flag:
        observations = pd.read_csv(name)
        flag = True
    else:
        observations = observations.append(pd.read_csv(name))
    print(f'*** Deleting {name}')
    os.remove(name)
print(f"*** Saving: observations {observations.shape}")
df = pd.DataFrame(observations).set_index('trajectory_step')
_dir = os.path.split(os.getcwd())[0]
df.to_csv(os.path.join(args.path, f'observations_{args.exp}.csv'))
print(f"*** Saved: data/observations.csv")

springs = None
flag = False
for name in glob.glob(os.path.join(os.getcwd(), 'simulations', 'data', 'springs_*.csv')):
    print(f'*** Reading {name}')
    if not flag:
        springs = pd.read_csv(name)
        flag = True
    else:
        springs = springs.append(pd.read_csv(name))
    print(f'*** Deleting {name}')
    os.remove(name)
print(f"*** Saving: springs {springs.shape}")
df = pd.DataFrame(springs).set_index('trajectory_step')
_dir = os.path.split(os.getcwd())[0]
df.to_csv(os.path.join(args.path, f'springs_{args.exp}.csv'))
print(f"*** Saved: data/springs.csv")
