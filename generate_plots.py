def get_experiment_id():
    import datetime
    import pytz
    timezone = pytz.timezone("America/Los_Angeles")
    dt = timezone.localize(datetime.datetime.now())
    _time = f'{dt.time().hour}:{dt.time().minute}:{dt.time().second}'
    _day = f'{dt.date().month}/{dt.date().day}/{dt.date().year}'
    name = f'exp_{_day}-{_time}'
    return name


def create_gif():
    import os
    import glob
    from PIL import Image

    loc = get_experiment_id()
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


create_gif()
