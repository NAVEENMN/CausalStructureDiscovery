#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import logging
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
data_observations_path = os.path.join(os.getcwd(), 'data', 'observations.csv')
data_springs_path = os.path.join(os.getcwd(), 'data', 'springs.csv')

logging.info('*** Loading observations and springs data')
observations = pd.read_csv(data_observations_path)
springs = pd.read_csv(data_springs_path)
particle_count = 0
for col in observations.columns:
    if 'position' in col:
        particle_count += 1
particle_count /= 2
particle_count = int(particle_count)
logging.info(f'*** Generating snapshots for {particle_count} particles')


def snapshot(time_step):
    logging.info(f'Saving snapshot {time_step}')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('Position')
    axes[1].set_title('Spring')
    fig.suptitle(f'Time step {time_step}')

    columns = [f'particle_{i}' for i in range(particle_count)]
    springs_matrix = []
    for p_a in range(particle_count):
        for p_b in range(particle_count):
            springs_matrix.append(springs.iloc[time_step][f's_{p_a}_{p_b}'])
    springs_matrix = np.reshape(springs_matrix, (particle_count, particle_count))
    springs_matrix = pd.DataFrame(springs_matrix, columns=columns, index=columns)

    positions = []
    for particle_id in range(0, particle_count):
        data = {'particle': particle_id,
                'x_cordinate': observations.iloc[time_step][f'p_{particle_id}_x_position'],
                'y_cordinate': observations.iloc[time_step][f'p_{particle_id}_y_position']}
        positions.append(data)
    pl = sns.scatterplot(data=pd.DataFrame(positions),
                         x='x_cordinate',
                         y='y_cordinate',
                         hue='particle',
                         ax=axes[0])
    sns.heatmap(springs_matrix, vmin=0.0, vmax=1.0, ax=axes[1])
    pl.set_ylim(-5.0, 5.0)
    pl.set_xlim(-5.0, 5.0)
    plt.savefig(f"{os.getcwd()}/media/timestep_{time_step}.png")
    plt.clf()
    logging.info(f"plot saved to {os.getcwd()}/media/timestep_{time_step}.png")


def create_gif(loc='particle_simulation'):
    """
    Read png files from a location and compose a gif
    :param
    """
    import os
    import glob
    from PIL import Image

    fcont = len(glob.glob(f"{os.getcwd()}/media/timestep_*.png"))
    print(f'Creating gif with {fcont} images')
    # ref: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img, *imgs = [Image.open(f"{os.getcwd()}/media/timestep_{i}.png") for i in range(1, fcont)]
    img.save(fp=f"{os.getcwd()}/media/{loc}.gif",
             format='GIF',
             append_images=imgs,
             save_all=True,
             duration=10,
             loop=0)

    # delete all png files.
    fp_in = f"{os.getcwd()}/media/timestep_*.png"
    for f in glob.glob(fp_in):
        os.remove(f)
    logging.info('trajectory gif stores in media')


def main():
    start = time.time()

    # delete all png files.
    fp_in = f"{os.getcwd()}/media/timestep_*.png"
    for f in glob.glob(fp_in):
        os.remove(f)

    # Generate png for all time steps
    observation_ids = range(len(observations))
    with Pool(8) as p:
        p.map(snapshot, observation_ids)
    print(f'Total time taken: {time.time() - start}')

    # Merge all png to a gif
    create_gif()



if __name__ == "__main__":
    main()