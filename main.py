#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import os
import glob
import time
import json
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests import ParCorr
from tigramite.pcmci import PCMCI
import seaborn as sns
import sklearn.metrics
from multiprocessing import Pool

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(description='Control variables for PC Algorithm.')
parser.add_argument('--pv', default=0.02, help='Threshold p value', type=float)
parser.add_argument('--tau', default=100, help='Max Tau', type=int)


# Variables of interest
variables_dim_1 = ['p_0_x_position', 'p_1_x_position', 'p_2_x_position', 'p_3_x_position']
variables_dim_2 = ['p_0_y_position', 'p_1_y_position', 'p_2_x_position', 'p_3_x_position']

data_observations_path = os.path.join(os.getcwd(), 'simulations', 'data', 'observations.csv')
springs_observations_path = os.path.join(os.getcwd(), 'simulations', 'data', 'springs.csv')

def get_experiment_id():
    import datetime
    import pytz
    timezone = pytz.timezone("America/Los_Angeles")
    dt = timezone.localize(datetime.datetime.now())
    _time = f'{dt.time().hour}:{dt.time().minute}:{dt.time().second}:{dt.time().microsecond}'
    _day = f'{dt.date().month}:{dt.date().day}:{dt.date().year}'
    name = f'exp_{_day}-{_time}'
    return name

def load_observations(path, _variables):
    data = pd.read_csv(path)
    # reversing the data so that the least valued
    # 0 index represents most reset observation
    # collect variables of interest
    if _variables:
        data = data[_variables]
    print(data.head())
    var_names = data.columns.values
    dataframe = pp.DataFrame(data.values,
                             datatime=np.arange(len(data)),
                             var_names=var_names)
    return dataframe


# Positions of particles
observations_dim_1 = load_observations(data_observations_path, variables_dim_1)
observations_dim_2 = load_observations(data_observations_path, variables_dim_2)

# Spring constants between particles
springs = load_observations(springs_observations_path, [])


def setup_pcmci(data_frame):
    pcmci = PCMCI(dataframe=data_frame,
                  cond_ind_test=ParCorr(),
                  verbosity=0)
    # Auto correlations are helpful to determine tau
    # correlations = pcmci.get_lagged_dependencies(tau_max=100, val_only=True)['val_matrix']
    return pcmci


def construct_causal_graph(time_step, p_values_dim_1, p_values_dim_2, p_threshold):
    _vars = [f'particle_{i}' for i in range(len(variables_dim_1))]
    graph = nx.complete_graph(_vars)
    for p_a in range(len(_vars)):
        for p_b in range(len(_vars)):
            if p_a != p_b:
                avg_p_val = (p_values_dim_1[p_a][p_b][time_step] + p_values_dim_2[p_a][p_b][time_step])/2.0
                if graph.has_edge(f'particle_{p_a}', f'particle_{p_b}') and (np.abs(avg_p_val) > p_threshold):
                    graph.remove_edge(f'particle_{p_a}', f'particle_{p_b}')

    # variables_dim_1 is ok
    area_under_curve = save_graph(time_step, graph, variables_dim_1)
    return area_under_curve, graph


def save_graph(time_step, causal_graph, _variables):
    # observations -> positions
    # springs -> spring constants
    # causal graph from predictions
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))

    # ----- Plotting Particle positions
    axes[0][0].set_title('Particle position')
    entries = []
    _observations = pd.read_csv(data_observations_path)
    for particle_id in range(0, len(variables_dim_1)):
        data = {'particle': particle_id,
                'x_cordinate': _observations.iloc[time_step][f'p_{particle_id}_x_position'],
                'y_cordinate': _observations.iloc[time_step][f'p_{particle_id}_y_position']}
        entries.append(data)
    pdframe = pd.DataFrame(entries)
    pl = sns.scatterplot(data=pdframe,
                         x='x_cordinate',
                         y='y_cordinate',
                         hue='particle',
                         ax=axes[0][0])
    pl.set_ylim(-5.0, 5.0)
    pl.set_xlim(-5.0, 5.0)

    # ----- Plotting spring constants
    _springs = pd.read_csv(springs_observations_path)
    axes[0][1].set_title(f'Spring connections')
    columns = [f'particle_{i}' for i in range(len(_variables))]
    s_mat = []
    for p_a in range(len(_variables)):
        for p_b in range(len(_variables)):
            s_mat.append(_springs.iloc[time_step][f's_{p_a}_{p_b}'])
    s_mat = np.reshape(s_mat, (len(_variables), len(_variables)))
    sns.heatmap(pd.DataFrame(s_mat, columns=columns, index=columns),
                vmin=0.0, vmax=2.0, ax=axes[0][1])

    # ----- Plotting Ground Truth Causal graph
    axes[1][0].set_title(f'Ground truth causal graph (Springs)')
    _vars = [f'particle_{i}' for i in range(len(_variables))]
    graph = nx.complete_graph(_vars)
    for p_a in range(len(_vars)):
        for p_b in range(len(_vars)):
            if np.abs(_springs.iloc[time_step][f's_{p_a}_{p_b}']) == 0.0 and graph.has_edge(f'particle_{p_a}', f'particle_{p_b}'):
                graph.remove_edge(f'particle_{p_a}', f'particle_{p_b}')
    nx.draw(graph,
            pos=nx.circular_layout(graph),
            with_labels=True,
            ax=axes[1][0],
            node_size=500)

    # ----- Plotting Predicted Causal graph
    axes[1][1].set_title(f'Predicted causal graph (Springs)')
    nx.draw(causal_graph,
            pos=nx.circular_layout(causal_graph),
            with_labels=True,
            ax=axes[1][1],
            node_size=500)

    true_labels = []
    pred_labels = []

    for p_a in range(len(_vars)):
        for p_b in range(len(_vars)):
            if np.abs(_springs.iloc[time_step][f's_{p_a}_{p_b}']) == 0.0:
                true_labels.append(0)
                if causal_graph.has_edge(f'particle_{p_a}', f'particle_{p_b}'):
                    # false positive
                    pred_labels.append(1)
                else:
                    # True negative
                    pred_labels.append(0)
            else:
                true_labels.append(1)
                if causal_graph.has_edge(f'particle_{p_a}', f'particle_{p_b}'):
                    # True positive
                    pred_labels.append(1)
                else:
                    # False negative
                    pred_labels.append(0)
    #positive class is 1; negative class is 0
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=true_labels, y_score=pred_labels, pos_label=1)
    auroc = sklearn.metrics.auc(fpr, tpr)

    args = parser.parse_args()

    details = f'p_threshold: {args.pv}, tau: {args.tau}, fpr: {round(np.mean(fpr), 2)}, tpr: {round(np.mean(tpr), 2)}, auroc:{round(auroc, 2)}'
    fig.suptitle(f'Time step {time_step} - {details}')

    #plt.show()
    fig.savefig(os.path.join(os.getcwd(), 'tmp', f'graph_{time_step}.png'))
    plt.clf()
    plt.close(fig)

    return auroc


def get_parents(tau_max, tau_min):
    _vars = list(range(len(variables_dim_1)))
    _lags = list(range(-(tau_max), -tau_min + 1, 1))
    # Set the default as all combinations of the selected variables
    _int_sel_links = {}
    for j in _vars:
        _int_sel_links[j] = [(var, -lag) for var in _vars
                             for lag in range(tau_min, tau_max + 1)
                             if not (var == j and lag == 0)]
    # Remove contemporary links
    for j in _int_sel_links.keys():
        _int_sel_links[j] = [link for link in _int_sel_links[j]
                             if link[1] != 0]
    # Remove self links
    for j in _int_sel_links.keys():
        _int_sel_links[j] = [link for link in _int_sel_links[j]
                             if link[0] != j]

    return _int_sel_links

def run_pc(dim):
    args = parser.parse_args()
    print(f'Running pcmci on dim {dim}')
    parents = get_parents(tau_min=1, tau_max=args.tau)
    pcmci = setup_pcmci(observations_dim_1)
    pcmci.verbosity = 0
    results = pcmci.run_pcmci(tau_max=args.tau,
                              selected_links=parents)
    p_values = results['p_matrix'].round(3)
    logging.info(f'Saving pcmci {dim}')
    np.save(os.path.join(os.getcwd(), 'data', f'p_values_dim_{dim}'), p_values)
    logging.info(f'Saved pcmci {dim}')

def main():
    # First they estimate all parents for last layer.
    # Using the same kin relationship as parent sets
    # The same set of parents are used for momemtary ci test backwards in time.
    # Running pcmci on dim 1

    exp_id = get_experiment_id()

    # *** Control Variables ***
    args = parser.parse_args()
    tau_max = args.tau
    p_threshold = args.pv

    _springs = pd.read_csv(springs_observations_path)

    # Clean
    if os.path.exists('data/p_values_dim_1.npy'):
        os.remove('data/p_values_dim_1.npy')
    if os.path.exists('data/p_values_dim_2.npy'):
        os.remove('data/p_values_dim_2.npy')

    start_time = time.time()
    dims = [1, 2]
    with Pool(4) as p:
        p.map(run_pc, dims)

    with open('data/p_values_dim_1.npy', 'rb') as f1:
        p_values_dim_1 = np.load(f1)

    with open('data/p_values_dim_2.npy', 'rb') as f2:
        p_values_dim_2 = np.load(f2)

    logging.info('Constructing causal graph')
    links_distribution = dict()
    links_distribution['links'] = []
    _vars = [f'particle_{i}' for i in range(len(variables_dim_1))]
    time_step = tau_max-1
    aurocs = []
    while time_step != 0:
        auroc, graph = construct_causal_graph(time_step, p_values_dim_1, p_values_dim_2, p_threshold)
        aurocs.append(auroc)
        for p_a in range(len(_vars)):
            for p_b in range(len(_vars)):
                if (p_a != p_b) and (graph.has_edge(f'particle_{p_a}', f'particle_{p_b}')):
                    links_distribution['links'].append(f'{p_a}-{p_b}')
        time_step -= 1

    end_time = time.time()

    print('Done.')
    print(f'Average AUROC {np.mean(aurocs)}')

    # Publish results
    # Read simulation settings
    json_files = glob.glob('simulations/data/*.json')
    with open(json_files[0]) as json_file:
        results = json.load(json_file)
    results['tau'] = tau_max
    results['p_threshold'] = p_threshold
    results['auroc'] = np.mean(aurocs)
    with open(f'result/simulation_details_{exp_id}.json', 'w') as f:
        json.dump(results, f)

    df = pd.DataFrame(links_distribution)
    _title = f'LinkDistribution - {exp_id} - tau:{tau_max} - p_threshold: {p_threshold}'
    dist_plt = sns.histplot(df, x='links').set_title(_title)
    dist_plt = dist_plt.get_figure()
    dist_plt.savefig(f'result/links_dist_{exp_id}.png')

    print(f'Total time taken {end_time - start_time}')

# delete all png files.
fp_in = f"{os.getcwd()}/tmp/timestep_*.png"
for f in glob.glob(fp_in):
    os.remove(f)
logging.info('trajectory gif stores in media')


if __name__ == "__main__":
    main()