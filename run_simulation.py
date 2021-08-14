#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""
Simulates a chosen system
(Spring Particle, Charge Particle or Gravity Particles)
writes observational data and schema to /data
"""
import time
import argparse
import numpy as np
from experiment import Experiment
import logging
from multiprocessing import Pool
from simulations.particle_system import SpringSystem

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Control variables for simulations.')
parser.add_argument('--np', default=4, help='Number of particles.', type=int)
parser.add_argument('--tl', default=5000, help='Trajectory length of individual simulation.', type=int)
parser.add_argument('--sf', default=50, help='Sample frequency for individual simulation', type=int)
parser.add_argument('--ns', default=1, help='Total number of simulations.', type=int)
parser.add_argument('--vel', default=0.0, help='Initial mean velocity of particles.', type=float)
# Zero implies static edges
parser.add_argument('--per', default=0, help='Spring period.', type=int)

args = parser.parse_args()
experiment = Experiment()

# *** Control Variables ***
experiment.set_numb_of_particles(num_of_particles=args.np)
experiment.set_traj_length(traj_length=args.tl)
experiment.set_sample_freq(sample_freq=args.sf)
experiment.set_period(period=args.per)
experiment.set_initial_velocity(vel=args.vel)
# ********


def run_spring_particle_simulation(_id=0):
    logging.info(f'*** Simulation: Each simulation will have {args.tl / args.sf} snapshots.')

    # Configure the particle system
    sp = SpringSystem()
    sp.add_particles(args.np)
    sp.set_initial_velocity_mean_sd((args.vel, 0.5))
    logging.info(f'*** Running: Simulation {_id}')

    # *** Control Variable ***
    sp.add_a_spring(particle_a=0,
                    particle_b=1,
                    spring_constant=np.random.normal(3, 0.5, 1))
    # ********

    # total_time_steps: run simulation with the current configuration for total_time_steps
    # sample_freq : make an observation for every sample_freq step.
    # For a good trajectory longer time_steps recommended
    particle_observations = experiment.get_particle_observational_record()
    spring_observations = experiment.get_springs_observational_record()
    sp.simulate(total_time_steps=args.tl,
                period=args.per,
                observations=particle_observations,
                spring_observations=spring_observations,
                sample_freq=args.sf,
                traj_id=_id)
    logging.info(f'*** Complete: Simulation')

    # Save observations to a csv file
    particle_observations.save_observations(name=f'observations_{_id}')
    spring_observations.save_observations(name=f'springs_{_id}')


def main():
    # Run simulations followed by merge simulations to consolidate them
    args = parser.parse_args()
    logging.info(f'Running {args.ns} simulations with {args.np} particles of trajectory length {args.tl} with sample frequency {args.sf}')
    start = time.time()
    args = parser.parse_args()
    number_of_simulations = list(range(args.ns))

    #run_spring_particle_simulation(_id=0)
    with Pool(4) as p:
        p.map(run_spring_particle_simulation, number_of_simulations)

    experiment.save()
    print('*** Merge all observations')
    print(f'*** Run : python3 simulations/merge_simulations.py --exp {experiment.get_id()}')
    print(f'Total time taken: {time.time() - start}')


if __name__ == "__main__":
    main()
