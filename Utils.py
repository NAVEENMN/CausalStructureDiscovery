import os
import json
import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


class Utils(object):

    @classmethod
    def create_gif(cls, loc='pc_simulation'):
        import os
        import glob
        from PIL import Image

        fcont = len(glob.glob(f"{os.getcwd()}/tmp/graph_*.png"))
        # ref: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img, *imgs = [Image.open(f"{os.getcwd()}/tmp/graph_{i}.png") for i in range(1, fcont)]
        img.save(fp=f"{os.getcwd()}/media/{loc}.gif",
                 format='GIF',
                 append_images=imgs,
                 save_all=True,
                 duration=10,
                 loop=0)

        # delete all png files.
        fp_in = f"{os.getcwd()}/tmp/graph_*.png"
        for f in glob.glob(fp_in):
            os.remove(f)

    @classmethod
    def save_pair_plot(cls, observations):
        import os
        import seaborn as sns
        sns.pairplot(observations).savefig(f"{os.getcwd()}/media/relationship.png")


#ut = Utils()
#ut.create_gif()
#print('Done.')