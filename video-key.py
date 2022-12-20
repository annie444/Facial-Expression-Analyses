# %%
import glob
import logging
import os
import platform
import re
import sys
import traceback

import av
import numpy as np
import pandas as pd
import pims
import progressbar
from skimage import draw

# %% Cell [1]
logging.basicConfig(filename='video-processing.log',
                    encoding='utf-8',
                    level=logging.INFO)

if platform.system() == 'Darwin':
    data_path = f"/Volumes/specialk_cs/2p/raw/"
else:
    data_path = f"/nadata/snlkt/specialk_cs/2p/raw/"

mice = ['CSC009', 'CSE008', 'CSC013', 'CSE020']

logging.info(
    "\n\n\n\n\n ------------------------------------ Cell [1] initialized ------------------------------------ \n\n\n\n\n"
)

# %% Cell [2]

with pd.HDFStore(
        "/home/aehler/Projects/Jupyter_Notebooks/python_outputs/data.h5"
) as data_file:
    data = data_file.get('all_trial_data')

data.index.set_names(["mouse", "week", "trial", "frame"], inplace=True)

logging.info(
    "\n\n\n\n\n ------------------------------------ Cell [2] initialized ------------------------------------ \n\n\n\n\n"
)

# %% Cell [3]

data = data.dropna(
    axis=0,
    thresh=4,
    subset=[
        "upper_eye_x", "upper_eye_y", "lower_eye_x", "lower_eye_y",
        "upper_ear_x", "upper_ear_y", "lower_ear_x", "lower_ear_y",
        "outer_ear_x", "outer_ear_y", "upper_whisker_x", "upper_whisker_y",
        "outer_whisker_x", "outer_whisker_y", "lower_whisker_x",
        "lower_whisker_y", "upper_mouth_x", "upper_mouth_y", "outer_mouth_x",
        "outer_mouth_y", "lower_mouth_x", "lower_mouth_y", "inner_nostril_x",
        "inner_nostril_y", "outer_nostril_x", "outer_nostril_y"
    ])

weeks = [0, 1, 5, 6]

trials = ["Sucrose", "Airpuff"]

new_data = data.loc[((data.week_list == 0) | (data.week_list == 1)
                     | (data.week_list == 5) | (data.week_list == 6))
                    & ((data.trial_type == "Sucrose")
                       | (data.trial_type == "Airpuff"))]

frame_trial_idx = new_data.groupby(
    ["mouse_list", "week_list", "trial_type", "trial_num"],
    group_keys=False).apply(lambda x: (x.frame_list - x.frame_list.min()))

frame_trial_idx.name = "frame_trial_idx"

new_data = new_data.join(frame_trial_idx)

data = data.join(frame_trial_idx)

logging.info(
    "\n\n\n\n\n ------------------------------------ Cell [3] initialized ------------------------------------ \n\n\n\n\n"
)

# %% Cell [4]

cse_array = ['CSE008', 'CSE020']
csc_array = ['CSC009', 'CSC013']

cse = new_data.mouse_list.isin(cse_array)
csc = new_data.mouse_list.isin(csc_array)

data_cse = new_data[cse]
data_csc = new_data[csc]

logging.info(
    "\n\n\n\n\n ------------------------------------ Cell [4] initialized ------------------------------------ \n\n\n\n\n"
)

# %% Cell [5]

dfs = [data_csc, data_cse]
df_coords = [[], []]

raw_coords = pd.concat(dfs, keys=['CSC', 'CSE'])

for d, df in enumerate(dfs):
    for point in ["eye", "ear", "whisker", "mouth", "nostril"]:
        for coord in ["_x", "_y"]:
            df_coord = df.loc[:, "upper_eye_x":"outer_nostril_y"].filter(
                like=point).filter(like=coord).groupby(level=[0, 1],
                                                       group_keys=False).apply(
                                                           np.mean, axis=1)
            df_coord.name = f"{point}{coord}"
            df_coords[d].append(df_coord)
    if d == 0:
        data_csc = data_csc.join(df_coords[d])
    elif d == 1:
        data_cse = data_cse.join(df_coords[d])

for d, dfs in enumerate(df_coords):
    df_coords[d] = pd.concat(dfs, axis=1)

mean_coords = pd.concat(df_coords, keys=['CSC', 'CSE'], names=["mouse_type"])

logging.info(
    "\n\n\n\n\n ------------------------------------ Cell [5] initialized ------------------------------------ \n\n\n\n\n"
)

# %% Cell [6]

videos = []

for mouse in mice:

    _, weeks, _ = next(os.walk(data_path + mouse), ([], [], []))

    weeks.sort()

    base_path = data_path + mouse

    for w in range(len(weeks)):

        os.chdir(f"{base_path}/{weeks[w]}/")

        video = glob.glob("*.mp4")

        videos.append(f"{base_path}/{weeks[w]}/{video[0]}")

logging.info(
    "\n\n\n\n\n ------------------------------------ Cell [6] initialized ------------------------------------ \n\n\n\n\n"
)

#%% Cell [7]

jobs = [
    # Each job takes between 1 and 10 steps to complete
    [0, 4 * 4],
    [0, len(mean_coords)],
    #[0, len(mean_coords) * 5],
    #[0, len(mean_coords)],
    #[0, len(mean_coords) * 13],
    #[0, len(mean_coords)],
    [0, len(videos)],
    [0, len(mean_coords) * 5]
]

markers = [
    '\x1b[31m▁\x1b[39m', '\x1b[31m▂\x1b[39m', '\x1b[33m▃\x1b[39m',
    '\x1b[33m▄\x1b[39m', '\x1b[32m▅\x1b[39m', '\x1b[32m▆\x1b[39m',
    '\x1b[34m▇\x1b[39m', '\x1b[34m█\x1b[39m'
]

widgets = [
    progressbar.Percentage(),
    progressbar.MultiProgressBar('jobs', markers=markers, fill_left=True),
    progressbar.AdaptiveETA()
]

max_value = sum([total for progress, total in jobs])

colors = np.array([[213, 94, 0], [204, 121, 167], [0, 114, 178],
                   [240, 228, 66], [0, 158, 115]],
                  dtype=np.int8)

logging.info(
    "\n\n\n\n\n ------------------------------------ Cell [7] initialized ------------------------------------ \n\n\n\n\n"
)

#%% Cell [8]

logging.info(
    "\n\n\n\n\n ------------------------------------ Cell [8] initialized ------------------------------------ \n\n\n\n\n"
)

with progressbar.ProgressBar(widgets=widgets, max_value=max_value) as bar:

    bar.is_ansi_terminal = True
    bar.enable_colors = True

    weeks = {}

    for i in range(len(videos)):

        mouse_type = re.search(r'\/CS[A-Z]+', videos[i]).group().strip("/")
        mouse = re.search(r'\/CS[A-Z0-9]+', videos[i]).group().strip("/")

        if mouse not in weeks.keys():
            weeks[f"{mouse}"] = {
                "weeks":
                mean_coords.xs(
                    (mouse_type, mouse), level=[
                        "mouse_type",
                        "mouse",
                    ]).index.get_level_values("week").unique().to_list(),
                "videos":
                [[
                    video for video in videos if re.search(mouse, video)
                ][v] for v in range(
                    len([video for video in videos
                         if re.search(mouse, video)]))
                 if v in mean_coords.xs(
                     (mouse_type, mouse), level=[
                         "mouse_type",
                         "mouse",
                     ]).index.get_level_values("week").unique().to_list()]
            }

    logging.info(f"\n\n\n\tweeks:\n\t\t{weeks}")

    for mouse in weeks.keys():
        for i in range(len(weeks[f"{mouse}"]["videos"])):

            mouse_type = re.search(r'CS[A-Z]+', mouse).group()

            same_trial = 500

            logging.info(f"\n\n\n\tmouse_type:\n\t\t{mouse_type}")
            logging.info(f"\n\n\n\tmouse:\n\t\t{mouse}")
            logging.info(
                f'\n\n\n\tvideo:\n\t\t{weeks[f"{mouse}"]["videos"][i]}')

            try:

                # Creating a Video object to read the video
                video = pims.PyAVVideoReader(weeks[f"{mouse}"]["videos"][i])

                logging.info(f"\n\n\n\tvideo initialized\n")

                jobs[0][0] += 1
                progress = sum([progress for progress, total in jobs])
                bar.update(progress, jobs=jobs, force=True)

                logging.info(
                    "\n\n\n\n\n ----- begin frame processing ----- \n\n\n\n\n")

                for frame in mean_coords.xs(
                    (mouse_type, mouse, weeks[f"{mouse}"]["weeks"][i]),
                        level=["mouse_type", "mouse", "week"
                               ]).index.get_level_values("frame").to_list():

                    full_buf = video.get_frame(frame)

                    frameHeight, frameWidth, colorSpace = full_buf.shape

                    jobs[1][0] += 1
                    progress = sum([progress for progress, total in jobs])
                    bar.update(progress, jobs=jobs, force=True)

                    trial = mean_coords.xs(
                        (mouse_type, mouse, weeks[f"{mouse}"]["weeks"][i],
                         frame),
                        level=["mouse_type", "mouse", "week", "frame"
                               ]).index.get_level_values("trial").to_list()[0]

                    if trial != same_trial:

                        same_trial = trial

                        # raw_video = av.open(
                        #     f'/home/aehler/Projects/Jupyter_Notebooks/python_outputs/videos/initial-points_{trial}-{weeks[f"{mouse}"]["videos"][i].replace(f"{data_path}", "")[16:]}.mov',
                        #     mode="w")
                        # out_raw = raw_video.add_stream("h264_videotoolbox",
                        #                                rate=30)
                        # out_raw.width = frameWidth
                        # out_raw.height = frameHeight
                        # out_raw.pix_fmt = "yuv420p"

                        # mean_video = av.open(
                        #     f'/home/aehler/Projects/Jupyter_Notebooks/python_outputs/videos/mean-points_{trial}-{weeks[f"{mouse}"]["videos"][i].replace(f"{data_path}", "")[16:]}.mov',
                        #     mode="w")
                        # out_mean = mean_video.add_stream(
                        #     "h264_videotoolbox", 30)
                        # out_mean.width = frameWidth
                        # out_mean.height = frameHeight
                        # out_mean.pix_fmt = "yuv420p"

                        animated_video = av.open(
                            f'/home/aehler/Projects/Jupyter_Notebooks/python_outputs/videos/animated_fully_{trial}-{weeks[f"{mouse}"]["videos"][i].replace(f"{data_path}", "")[16:]}',
                            mode="w")
                        animated_out = animated_video.add_stream(
                            "h264_videotoolbox", 30)
                        animated_out.width = frameWidth
                        animated_out.height = frameHeight
                        animated_out.pix_fmt = "yuv420p"

                    # mean_buf = buf

                    # for c, point in enumerate(
                    #     ["eye", "ear", "whisker", "mouth", "nostril"]):

                    #     rr, cc = draw.disk(
                    #         (mean_coords.xs(
                    #             (mouse_type, mouse,
                    #              weeks[f"{mouse}"]["weeks"][i], trial, frame),
                    #             level=[
                    #                 "mouse_type", "mouse", "week", "trial",
                    #                 "frame"
                    #             ])[f"{point}_y"].to_list()[0],
                    #          mean_coords.xs(
                    #              (mouse_type, mouse,
                    #               weeks[f"{mouse}"]["weeks"][i], trial, frame),
                    #              level=[
                    #                  "mouse_type", "mouse", "week", "trial",
                    #                  "frame"
                    #              ])[f"{point}_x"].to_list()[0]),
                    #         radius=12,
                    #         shape=mean_buf.shape)

                    #     mean_buf[rr, cc] = colors[c]

                    #     jobs[2][0] += 1
                    #     progress = sum([progress for progress, total in jobs])
                    #     bar.update(progress, jobs=jobs, force=True)

                    # new_frame = av.VideoFrame.from_ndarray(mean_buf,
                    #                                        format="rgb24")

                    # for packet in out_mean.encode(new_frame):

                    #     mean_video.mux(packet)

                    # jobs[3][0] += 1
                    # progress = sum([progress for progress, total in jobs])
                    # bar.update(progress, jobs=jobs, force=True)

                    # raw_buf = buf

                    # for point in [
                    #         "upper_eye", "lower_eye", "upper_ear", "lower_ear",
                    #         "outer_ear", "upper_whisker", "outer_whisker",
                    #         "lower_whisker", "upper_mouth", "outer_mouth",
                    #         "lower_mouth", "inner_nostril", "outer_nostril"
                    # ]:

                    #     rr, cc = draw.disk(
                    #         (data.xs((mouse, weeks[f"{mouse}"]["weeks"][i],
                    #                   trial, frame),
                    #                  level=["mouse", "week", "trial", "frame"
                    #                         ])[f"{point}_y"].to_list()[0],
                    #          data.xs((mouse, weeks[f"{mouse}"]["weeks"][i],
                    #                   trial, frame),
                    #                  level=["mouse", "week", "trial", "frame"
                    #                         ])[f"{point}_x"].to_list()[0]),
                    #         radius=12,
                    #         shape=raw_buf.shape)

                    #     raw_buf[rr, cc] = [243, 219, 158]

                    #     jobs[4][0] += 1
                    #     progress = sum([progress for progress, total in jobs])
                    #     bar.update(progress, jobs=jobs, force=True)

                    # new_frame = av.VideoFrame.from_ndarray(raw_buf,
                    #                                        format="rgb24")

                    # for packet in out_raw.encode(new_frame):

                    #     raw_video.mux(packet)

                    # jobs[5][0] += 1
                    # progress = sum([progress for progress, total in jobs])
                    # bar.update(progress, jobs=jobs, force=True)

                    #full_buf = buf

                    for c, point in enumerate(
                        ["eye", "ear", "whisker", "mouth", "nostril"]):
                        oldies = data.xs(
                            (mouse, weeks[f"{mouse}"]["weeks"][i], trial,
                             frame),
                            level=["mouse", "week", "trial",
                                   "frame"]).filter(like=point).filter(
                                       like="_y").columns.to_list()
                        for old in oldies:

                            rr, cc = draw.disk(
                                (mean_coords.xs((mouse_type, mouse,
                                                 weeks[f"{mouse}"]["weeks"][i],
                                                 trial, frame),
                                                level=[
                                                    "mouse_type", "mouse",
                                                    "week", "trial", "frame"
                                                ])[f"{point}_y"].to_list()[0],
                                 mean_coords.xs((mouse_type, mouse,
                                                 weeks[f"{mouse}"]["weeks"][i],
                                                 trial, frame),
                                                level=[
                                                    "mouse_type", "mouse",
                                                    "week", "trial", "frame"
                                                ])[f"{point}_x"].to_list()[0]),
                                radius=12,
                                shape=full_buf.shape)
                            full_buf[rr, cc] = colors[c]

                            rr, cc = draw.disk(
                                (data.xs(
                                    (mouse, weeks[f"{mouse}"]["weeks"][i],
                                     trial, frame),
                                    level=["mouse", "week", "trial", "frame"
                                           ])[f"{old}"].to_list()[0],
                                 data.xs(
                                     (mouse, weeks[f"{mouse}"]["weeks"][i],
                                      trial, frame),
                                     level=["mouse", "week", "trial", "frame"])
                                 [f"{old.replace('_y', '_x')}"].to_list()[0]),
                                radius=12,
                                shape=full_buf.shape)
                            full_buf[rr, cc] = colors[c]

                            rr, cc, val = draw.line_aa(
                                round(
                                    mean_coords.xs(
                                        (mouse_type, mouse,
                                         weeks[f"{mouse}"]["weeks"][i], trial,
                                         frame),
                                        level=[
                                            "mouse_type", "mouse", "week",
                                            "trial", "frame"
                                        ])[f"{point}_y"].to_list()[0]),
                                round(
                                    mean_coords.xs(
                                        (mouse_type, mouse,
                                         weeks[f"{mouse}"]["weeks"][i], trial,
                                         frame),
                                        level=[
                                            "mouse_type", "mouse", "week",
                                            "trial", "frame"
                                        ])[f"{point}_x"].to_list()[0]),
                                round(
                                    data.xs(
                                        (mouse, weeks[f"{mouse}"]["weeks"][i],
                                         trial, frame),
                                        level=[
                                            "mouse", "week", "trial", "frame"
                                        ])[f"{old}"].to_list()[0]),
                                round(
                                    data.xs(
                                        (mouse, weeks[f"{mouse}"]["weeks"][i],
                                         trial, frame),
                                        level=[
                                            "mouse", "week", "trial", "frame"
                                        ])[f"{old.replace('_y', '_x')}"].
                                    to_list()[0]))

                            full_buf[rr, cc] = np.array([
                                val * colors[c, 0], val * colors[c, 1],
                                val * colors[c, 2]
                            ],
                                                        dtype=np.int8).T

                        jobs[3][0] += 1
                        progress = sum([progress for progress, total in jobs])
                        bar.update(progress, jobs=jobs, force=True)

                    new_frame = av.VideoFrame.from_ndarray(full_buf,
                                                           format="rgb24")

                    for packet in animated_out.encode(new_frame):
                        animated_video.mux(packet)

                # When everything done, release the video capture object
                video.close()

                # packet = out_mean.encode(None)
                # while packet is not None:
                #     mean_video.mux(packet)
                #     packet = out_mean.encode(None)

                # mean_video.close()

                # packet = out_raw.encode(None)
                # while packet is not None:
                #     raw_video.mux(packet)
                #     packet = out_raw.encode(None)

                # raw_video.close()

                packet = animated_out.encode(None)
                while packet is not None:
                    animated_video.mux(packet)
                    packet = animated_out.encode(None)

                animated_video.close()

                logging.info("Flushed")

            except Exception as exception:
                logging.info("ERROR")
                traceback.print_exc()
                exc_type, exc_value, tb = sys.exc_info()
                if tb is not None:
                    prev = tb
                    curr = tb.tb_next
                    while curr is not None:
                        prev = curr
                        curr = curr.tb_next
                    print(prev.tb_frame.f_locals)
        logging.info(
            "\n\n\n\n\n ------------------------------------ Next Week ------------------------------------ \n\n\n\n\n"
        )
    logging.info(
        "\n\n\n\n\n ------------------------------------ Next Mouse ------------------------------------ \n\n\n\n\n"
    )

    bar.finish()
# %%
