#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Simulates a chosen system
(Spring Particle, Charge Particle or Gravity Particles)
writes observational data and schema to /data
"""
import os
import time
import logging
import argparse
import pandas as pd
import numpy as np
import json
from particle_system import SpringSystem
from multiprocessing import Pool

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
parser = argparse.ArgumentParser(description='Control variables for simulations.')
parser.add_argument('--np', default=4, help='Number of particles.', type=int)
parser.add_argument('--tl', default=5000, help='Trajectory length of individual simulation.', type=int)
parser.add_argument('--sf', default=50, help='Sample frequency for individual simulation', type=int)
parser.add_argument('--ns', default=1, help='Total number of simulations.', type=int)
parser.add_argument('--vel', default=0.0, help='Initial mean velocity of particles.', type=float)
parser.add_argument('--per', default=0, help='Spring period.', type=int)


def get_experiment_id():
    import datetime
    import pytz
    timezone = pytz.timezone("America/Los_Angeles")
    dt = timezone.localize(datetime.datetime.now())
    _time = f'{dt.time().hour}:{dt.time().minute}:{dt.time().second}:{dt.time().microsecond}'
    _day = f'{dt.date().month}:{dt.date().day}:{dt.date().year}'
    name = f'exp_{_day}-{_time}'
    return name


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

    def save_observations(self, name):
        logging.info("*** Saving: observations")
        df = pd.DataFrame(self.observations).set_index('trajectory_step')
        # index represents most reset observation
        df = df[::-1]
        df.to_csv(os.path.join(os.getcwd(), 'data', f'{name}.csv'))
        logging.info(f"*** Saved: observations {name}.csv")


def run_spring_particle_simulation(_id=0):
    args = parser.parse_args()
    # *** Control Variables ***
    num_of_particles = 4
    trajectory_length = args.tl
    sample_freq = args.sf
    # Zero implies static edges
    period = args.per
    initial_velocity = args.vel
    # ********

    # Create Observation records
    particle_observations = Observations()
    spring_observations = Observations()

    logging.info(f'*** Simulation: Each simulation will have {trajectory_length/sample_freq} snapshots.')

    # Configure the observations for recording
    column_names = ['trajectory_step']
    column_names.extend([f'p_{particle_id}_x_position' for particle_id in range(num_of_particles)])
    column_names.extend([f'p_{particle_id}_y_position' for particle_id in range(num_of_particles)])
    column_names.extend([f'p_{particle_id}_x_velocity' for particle_id in range(num_of_particles)])
    column_names.extend([f'p_{particle_id}_y_velocity' for particle_id in range(num_of_particles)])
    for i in range(num_of_particles):
        for j in range(num_of_particles):
            column_names.append(f'p_{i}_{j}_distance')
    particle_observations.set_column_names(columns=column_names)

    spring_observation_columns = ['trajectory_step']
    for i in range(num_of_particles):
        for j in range(num_of_particles):
            spring_observation_columns.append(f's_{i}_{j}')
    spring_observations.set_column_names(columns=spring_observation_columns)

    # Run simulation
    # Configure the particle system
    sp = SpringSystem()
    sp.add_particles(num_of_particles)
    sp.set_initial_velocity_mean_sd((initial_velocity, 0.5))
    logging.info(f'*** Running: Simulation {_id}')

    # *** Control Variable ***
    sp.add_a_spring(particle_a=0,
                    particle_b=1,
                    spring_constant=np.random.normal(2, 0.5, 1))
    # ********

    # total_time_steps: run simulation with the current configuration for total_time_steps
    # sample_freq : make an observation for every sample_freq step.
    # For a good trajectory longer time_steps recommended
    sp.simulate(total_time_steps=trajectory_length,
                period=period,
                observations=particle_observations,
                spring_observations=spring_observations,
                sample_freq=sample_freq,
                traj_id=_id)
    logging.info(f'*** Complete: Simulation')

    # Save observations to a csv file
    particle_observations.save_observations(name=f'observations_{_id}')
    spring_observations.save_observations(name=f'springs_{_id}')


def main():
    start = time.time()
    args = parser.parse_args()
    number_of_simulations = list(range(args.ns))
    with Pool(4) as p:
        p.map(run_spring_particle_simulation, number_of_simulations)

    _data = pd.read_csv('data/observations.csv')

    # Write simulation details
    sdata = {'trajectory_length': args.tl,
             'number_of_simulations': args.ns,
             'sample_frequency': args.sf,
             'num_of_particles': args.np,
             'data_size': _data.shape[0]}
    with open(f'data/simulation_details_{get_experiment_id()}.json', 'w') as f:
        json.dump(sdata, f)

    print(f'Total time taken: {time.time() - start}')


if __name__ == "__main__":
    main()
