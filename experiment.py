# !/usr/bin/env python3.8
# -*- coding: utf-8 -*-
import os
import json

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from Utils import Log
import pandas as pd
import datetime
import pytz

timezone = pytz.timezone("America/Los_Angeles")
dt = timezone.localize(datetime.datetime.now())


class Observations(object):
    def __init__(self):
        self.column_names = []
        self.observations = dict()

    def set_column_names(self, columns):
        self.column_names = columns
        self.observations = {label: [] for label in columns}

    def add_an_observation(self, observation):
        for col_name in observation.keys():
            self.observations[col_name].append(observation[col_name])

    def get_observations(self):
        df = pd.DataFrame(self.observations)
        return df

    def save_observations(self, path, name):
        Log.debug("Simulation", "Experiment", "saving observations")
        df = pd.DataFrame(self.observations)
        # index represents most reset observation
        df.to_csv(os.path.join(path, f'{name}.csv'), index=False)
        Log.debug("Simulation", "Experiment", f"saved to  {os.path.join(path, f'{name}.csv')}")


class Experiment(object):
    def __init__(self, _id=None):
        self.experiment_path = os.path.join(os.getcwd(), '../meta', 'experiment.json')
        self._id = _id if _id else self._get_experiment_id()
        self.auroc = []
        self.meta_path = os.path.join(os.getcwd(), '../meta', 'meta.json')

    def get_id(self):
        return self._id

    def set_id(self, id):
        self._id = id

    def fmt_id(self, _id):
        #_time = f'{dt.time().hour}:{dt.time().minute}:{dt.time().second}:{dt.time().microsecond}'
        _day = f'{dt.date().month}:{dt.date().day}:{dt.date().year}'
        name = f'exp_{_day}-{_id}'
        return name

    def _get_experiment_id(self):
        _id = self.fmt_id(_id=1)
        #TODO: Experiment id number just increments by one doesnt take day into logic
        if os.path.exists(self.experiment_path):
            with open(self.experiment_path) as json_file:
                exp_data = json.load(json_file)
                _id = self.fmt_id(_id=len(exp_data.keys())+1)
        return _id

    def load_recent(self):
        meta_data = dict()
        if os.path.exists(self.meta_path):
            with open(self.meta_path) as json_file:
                meta_data = json.load(json_file)
        exp_id = meta_data['recent']
        self.set_id(exp_id)
        #self.load_settings()

    def create(self):
        Log.info("Simulation", "Experiment", f"Creating experiment {self._id}")
        exp_data = dict()
        if os.path.exists(self.experiment_path):
            with open(self.experiment_path) as json_file:
                exp_data = json.load(json_file)
        exp_data[self._id] = {
            'settings': {
                'springs': {
                    'min_step': None,
                    'max_step': None,
                    'tau_min': None,
                    'tau_max': None,
                    'traj_length': None,
                    'sample_freq': None,
                    'period': None,
                    'num_sim': None,
                    'number_of_particles': None,
                    'initial_velocity': None,
                    'particle_variables': [],
                    'edge_variables': []
                },
                'netsim': {
                    'length': None,
                    'num_channels': None,
                    'num_subjects': None,
                    'channel_variables': [],
                    'edges_variables': []
                },
                'kuramoto': {
                    'num_channels': None
                }
            },
            'results': {
                'conducted': False,
                'acd': {
                    'springs': {
                        'tau_min': None,
                        'tau_max': None,
                        'p_threshold': None,
                        'auroc': {
                            'mean': None,
                            'max': None,
                            'min': None,
                            'std': None
                        }
                    },
                    'netsim': {
                        'auroc': {
                            'mean': None,
                            'max': None,
                            'min': None,
                            'std': None
                        }
                    },
                    'kuramoto': {
                        'auroc': {
                            'mean': None,
                            'max': None,
                            'min': None,
                            'std': None
                        }
                    }
                }
            }
        }
        with open(self.experiment_path, 'w') as f:
            json.dump(exp_data, f, indent=4)
        Log.info('Simulations', 'Experiment', f"Saved experiment {self._id} settings to {self.experiment_path}")

        meta_data = {}
        if os.path.exists(self.meta_path):
            with open(self.meta_path) as j_file:
                meta_data = json.load(j_file)
        meta_data["recent"] = self.get_id()
        meta_data["date"] = f'{dt.date().month}:{dt.date().day}:{dt.date().year}'
        meta_data["time"] = f'{dt.time().hour}:{dt.time().minute}:{dt.time().second}:{dt.time().microsecond}'
        with open(self.meta_path, 'w') as f:
            json.dump(meta_data, f, indent=4)
        Log.info('Simulations', 'Experiment', f"meta data for experiment {self._id} logged to {self.meta_path}")


class KuromotoSystem(Experiment):
    def __init__(self):
        super().__init__()
        self.num_channels = 0
        self.length = 0

    def get_num_of_channels(self):
        return self.num_channels

    def set_num_of_channels(self, value):
        self.num_channels = value

    def set_length(self, length):
        self.length = length

    def get_length(self):
        return self.length

    def _get_channel_vars(self):
        _nc = self.get_num_of_channels()
        column_names = []
        column_names.extend([f'channel_{_id}' for _id in range(_nc)])
        return column_names

    def get_channel_observational_record(self):
        channel_observations = Observations()
        _vars = self._get_channel_vars()
        _vars.append('subject')
        channel_observations.set_column_names(columns=_vars)
        return channel_observations
