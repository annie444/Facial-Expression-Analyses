# %%
import glob
import json
import os
import platform
import statistics
from math import isnan

import cv2
import ffmpeg
import h5py
import hdfdict
import numpy as np
import pandas as pd
from scipy.interpolate import interp1dc
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA


# %%
def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y


def smooth_diff(node_loc, win=25, poly=3):
    """
    node_loc is a [frames, 2] array

    win defines the window to smooth over

    poly defines the order of the polynomial
    to fit with

    """
    node_loc_vel = np.zeros_like(node_loc)

    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=1)

    node_vel = np.linalg.norm(node_loc_vel, axis=1)

    return node_vel


def into_trial_format(var, trial_start_idx, trial_end_idx):
    var_trials = []
    for start, end in zip(trial_start_idx, trial_end_idx):
        var_trials.append(var[start:end])
    return var_trials


# create gaussian kernel for smoothing
def gaussian_kernel(window_size, sigma=1):
    x_vals = np.arange(window_size)
    to_ret = np.exp(-((x_vals - window_size // 2) * 2) / (2 * sigma * 2))
    to_ret[: window_size // 2] = 0
    return to_ret


def reduce_led(iterable: list) -> list[float]:
    list: list[float] = []
    j: int = 0
    list.append(iterable[j])
    for i in range(0, len(iterable)):
        if iterable[j] < (iterable[i] - 5000):
            j = i
            list.append(iterable[j])
    return list


# %%
# define the paths and the mice

if platform.system() == "Darwin":
    data_path = f"/Volumes/specialk_cs/2p/raw/"
else:
    data_path = f"/nadata/snlkt/specialk_cs/2p/raw/"

mice = ["CSC009", "CSC013", "CSE008", "CSE020"]

# %%
# create a list of dictionaries of lists to sort by
# mouse (list) then variable name (dictionary) and variable values (list)

exprs = [{} for _ in range(len(mice))]

for m, mouse in enumerate(mice):
    print(mouse, "-- starting")

    # get each week folder from mouse base directory
    _, sub_folders, _ = next(os.walk(data_path + mouse), ([], [], []))

    # complete the path for each week in order of completion
    # (i.e. week 0 is the first week, week 1 the second)
    sub_folders.sort()
    n_exprs = len(sub_folders)
    base_path = data_path + mouse
    expr_filter = np.zeros(n_exprs, dtype=bool)
    data = [[] for _ in range(n_exprs)]

    # Create behavior arrays for each varriable
    # with the length as the number of weeks
    airpuff_on = [[] for _ in range(n_exprs)]
    airpuff_off = [[] for _ in range(n_exprs)]

    licks_on = [[] for _ in range(n_exprs)]
    licks_off = [[] for _ in range(n_exprs)]

    sucrose_on = [[] for _ in range(n_exprs)]
    sucrose_off = [[] for _ in range(n_exprs)]

    LED_on = [[] for _ in range(n_exprs)]
    LED_off = [[] for _ in range(n_exprs)]

    speaker_on = [[] for _ in range(n_exprs)]
    speaker_off = [[] for _ in range(n_exprs)]

    video_metadata = [[] for _ in range(n_exprs)]
    beh_metadata = [[] for _ in range(n_exprs)]
    trialArray = [[] for _ in range(n_exprs)]
    ITIArray = [[] for _ in range(n_exprs)]

    # enumerate each week folder
    for n, folder in enumerate(sub_folders):
        os.chdir(f"{base_path}/{folder}/")

        # get data from the csv files
        for csv in glob.glob("*.csv"):
            data[n] = pd.read_csv(f"{base_path}/{folder}/{csv}")

        # get data from the json files
        for js in glob.glob("*.json"):
            with open(f"{base_path}/{folder}/{js}", "r") as js_file:
                js_file = json.load(js_file)
                beh_metadata[n] = js_file.get("beh_metadata")
                trialArray[n] = js_file.get("beh_metadata")["trialArray"]
                ITIArray[n] = js_file.get("beh_metadata")["ITIArray"]

        # get the video metadata
        for video in glob.glob("*.mp4"):
            if len(glob.glob("*.mp4")) > 1:
                continue
            else:
                video_metadata[n] = ffmpeg.probe(f"{base_path}/{folder}/{video}")[
                    "streams"
                ][
                    (
                        int(
                            ffmpeg.probe(f"{base_path}/{folder}/{video}")["format"][
                                "nb_streams"
                            ]
                        )
                        - 1
                    )
                ]

        # set the week to "True" since it exists
        # this is necessary as some weeks don't have data (yet)
        expr_filter[n] = True

        # save list of values to variable arrays omitting NaN values
        airpuff_on[n] = [x for x in data[n]["Airpuff_on"] if isnan(x) == False]
        airpuff_off[n] = [x for x in data[n]["Airpuff_off"] if isnan(x) == False]

        sucrose_on[n] = [x for x in data[n]["Sucrose_on"] if isnan(x) == False]
        sucrose_off[n] = [x for x in data[n]["Sucrose_off"] if isnan(x) == False]

        LED_on[n] = [x for x in data[n]["LED590_on"] if isnan(x) == False]
        LED_off[n] = [x for x in data[n]["LED590_off"] if isnan(x) == False]

        speaker_on[n] = [x for x in data[n]["Speaker_on"] if isnan(x) == False]
        speaker_off[n] = [x for x in data[n]["Speaker_off"] if isnan(x) == False]

        licks_on[n] = [x for x in data[n]["Lick_on"] if isnan(x) == False]
        licks_off[n] = [x for x in data[n]["Lick_off"] if isnan(x) == False]
        if expr_filter[n]:
            print(f"\tDataset: #{n} in subfolder {folder} -- done")

    # filter the variable arrays to be only the length of weeks with data
    # this makes sure that we don't have a week of empty values in our variable arrays
    weeks = np.array(sub_folders)[expr_filter]
    data = [d for i, d in enumerate(data) if expr_filter[i]]

    airpuff_on = [d for i, d in enumerate(airpuff_on) if expr_filter[i]]
    airpuff_off = [d for i, d in enumerate(airpuff_off) if expr_filter[i]]

    sucrose_on = [d for i, d in enumerate(sucrose_on) if expr_filter[i]]
    sucrose_off = [d for i, d in enumerate(sucrose_off) if expr_filter[i]]

    LED_on = [d for i, d in enumerate(LED_on) if expr_filter[i]]
    LED_off = [d for i, d in enumerate(LED_off) if expr_filter[i]]

    speaker_on = [d for i, d in enumerate(speaker_on) if expr_filter[i]]
    speaker_off = [d for i, d in enumerate(speaker_off) if expr_filter[i]]

    licks_on = [d for i, d in enumerate(licks_on) if expr_filter[i]]
    licks_off = [d for i, d in enumerate(licks_off) if expr_filter[i]]

    video_metadata = [d for i, d in enumerate(video_metadata) if expr_filter[i]]
    beh_metadata = [d for i, d in enumerate(beh_metadata) if expr_filter[i]]
    trialArray = [d for i, d in enumerate(trialArray) if expr_filter[i]]
    ITIArray = [d for i, d in enumerate(ITIArray) if expr_filter[i]]

    # save the variable arrays to a dictionary with all the values
    for v in [
        "airpuff_on",
        "airpuff_off",
        "sucrose_on",
        "sucrose_off",
        "LED_on",
        "LED_off",
        "speaker_on",
        "speaker_off",
        "licks_on",
        "licks_off",
        "video_metadata",
        "beh_metadata",
        "trialArray",
        "ITIArray",
    ]:
        exec("exprs[%s]['%s'] = %s" % (m, v, v))
        exec("del(%s)" % (v))
    print(mouse, "-- complete")
print("Done")


# %%
# change path to point to the SLEAP datasets
if platform.system() == "Darwin":
    base_path = "/Volumes/snlktdata/Team2P/facial_expr/SLEAP/Complete/"
else:
    base_path = "/nadata/snlkt/data/Team2P/facial_expr/SLEAP/Complete/"
os.chdir(base_path)

# enumerate mice
for m, mouse in enumerate(mice):
    print(mouse, "-- starting")

    # get each SLEAP analysis file for the current mouse
    weeks = glob.glob(f"*{mouse}*")

    # sort the SLEAP analysis files by date (from first occourance to last)
    weeks.sort()
    n_weeks = len(weeks)
    week_filter = np.zeros(n_weeks, dtype=bool)

    # create variable arrays with the length of weeks
    datasets = [[] for _ in range(n_weeks)]
    edge_inds = [[] for _ in range(n_weeks)]
    edge_names = [[] for _ in range(n_weeks)]
    instance_scores = [[] for _ in range(n_weeks)]
    point_scores = [[] for _ in range(n_weeks)]
    track_occupancy = [[] for _ in range(n_weeks)]
    tracking_scores = [[] for _ in range(n_weeks)]
    tracking_locations = [[] for _ in range(n_weeks)]
    node_names = [[] for _ in range(n_weeks)]
    video_metadata = [[] for _ in range(n_weeks)]

    # enumerate weeks
    for w, week in enumerate(weeks):

        # concatenate the SLEAP analysis filename to the end of the base_path
        filename = f"{base_path}{weeks[w]}"

        # set the week to "True" since it exists
        # this is necessary as some weeks don't have data (yet)
        week_filter[w] = True

        # open the analysis file
        with h5py.File(filename, "r") as f:
            # with the analysis file open...

            # get each variable and store values
            datasets[w] = list(f.keys())
            tracking_locations[w] = fill_missing(f["tracks"][:].T)
            edge_inds[w] = f["edge_inds"][:].T
            edge_names[w] = f["edge_names"][:]
            instance_scores[w] = f["instance_scores"][:].T
            point_scores[w] = f["point_scores"][:].T
            track_occupancy[w] = f["track_occupancy"][:]
            tracking_scores[w] = f["tracking_scores"][:].T
            node_names[w] = [n.decode() for n in f["node_names"][:]]
        if week_filter[w]:
            print("\tWeek:", w, ": File:", week, "-- done")

    # filter out the weeks without data
    weeks = np.array(weeks)[week_filter]
    datasets = [d for i, d in enumerate(datasets) if week_filter[i]]
    tracking_locations = [d for i, d in enumerate(tracking_locations) if week_filter[i]]
    edge_inds = [d for i, d in enumerate(edge_inds) if week_filter[i]]
    edge_names = [d for i, d in enumerate(edge_names) if week_filter[i]]
    instance_scores = [d for i, d in enumerate(instance_scores) if week_filter[i]]
    point_scores = [d for i, d in enumerate(point_scores) if week_filter[i]]
    track_occupancy = [d for i, d in enumerate(track_occupancy) if week_filter[i]]
    tracking_scores = [d for i, d in enumerate(tracking_scores) if week_filter[i]]
    node_names = [d for i, d in enumerate(node_names) if week_filter[i]]

    # save the variable arrays to a dictionary with all the values
    for v in [
        "datasets",
        "tracking_locations",
        "node_names",
        "edge_names",
        "edge_inds",
        "instance_scores",
        "point_scores",
        "track_occupancy",
        "tracking_scores",
    ]:
        exec("exprs[%s]['%s'] = %s" % (m, v, v))
        exec("del(%s)" % (v))
    print(mouse, "-- complete")

print("Done")

# %%
# set original data path
if platform.system() == "Darwin":
    data_path = f"/Volumes/specialk_cs/2p/raw/"
else:
    data_path = f"/nadata/snlkt/specialk_cs/2p/raw/"

# enumerate each mouse
for m, mouse in enumerate(mice):
    print(mouse, "-- starting")

    # get each week folder from mouse base directory
    _, weeks, _ = next(os.walk(data_path + mouse), ([], [], []))

    # complete the path for each week in order of completion
    # (i.e. week 0 is the first week, week 1 the second)
    weeks.sort()
    n_weeks = len(weeks)
    week_filter = np.zeros(n_weeks, dtype=bool)

    # enumerate each (new) variable dictionary (using exec function for reduncacy)
    for v in [
        "mouse_list",
        "week_list",
        "frame_list",
        "timestamps",
    ]:
        # create a list with the length # of weeks for each varaible
        executable = compile(
            "%s = %s" % (v, [[] for _ in range(n_weeks)]), "casting", "exec"
        )
        exec(executable, globals(), locals())

    for v in [
        "upper_eye_x",
        "upper_eye_y",
        "lower_eye_x",
        "lower_eye_y",
        "upper_ear_x",
        "upper_ear_y",
        "lower_ear_x",
        "lower_ear_y",
        "outer_ear_x",
        "outer_ear_y",
        "upper_whisker_x",
        "upper_whisker_y",
        "outer_whisker_x",
        "outer_whisker_y",
        "lower_whisker_x",
        "lower_whisker_y",
        "upper_mouth_x",
        "upper_mouth_y",
        "outer_mouth_x",
        "outer_mouth_y",
        "lower_mouth_x",
        "lower_mouth_y",
        "inner_nostril_x",
        "inner_nostril_y",
        "outer_nostril_x",
        "outer_nostril_y",
    ]:
        # create a list with the length # of weeks for each varaible
        code = compile(f"{v} = [ _ for _ in range({n_weeks})]", "assign", "exec")
        exec(code, globals(), locals())

    # iterate each week for which we have the SLEAP tracks
    for w in range(0, len(exprs[m]["tracking_locations"]), 1):

        # if the video exists and is processed
        if type(exprs[m]["video_metadata"][w]) == type(dict()):

            week_filter[w] = True

            # calculate miliseconds per frame based on the video metadata
            miliseconds_per_frame = (
                eval(exprs[m]["video_metadata"][w].get("avg_frame_rate")) / 1000
            ) ** -1

            # enumerate each node (or point) in the SLEAP tracks
            for i, name in enumerate(exprs[m].get("node_names")[w]):

                # break down the complex 4D array into 1D arrays of x and y values
                exec(
                    "%s_x[%s] = np.array(%s)"
                    % (
                        name.replace(" ", "_"),
                        w,
                        exprs[m]["tracking_locations"][w][:, i, 0, 0].tolist(),
                    ),
                    globals(),
                    locals(),
                )
                exec(
                    "%s_y[%s] = np.array(%s)"
                    % (
                        name.replace(" ", "_"),
                        w,
                        exprs[m]["tracking_locations"][w][:, i, 1, 0].tolist(),
                    ),
                    globals(),
                    locals(),
                )

                # iterate each frame
                for f in range(
                    len(exprs[m]["tracking_locations"][w][:, i, 0, 0].tolist())
                ):
                    # label frame with the specific mouse, week, frame, and timestamp
                    mouse_list[w].append(mouse)
                    week_list[w].append(w)
                    frame_list[w].append(f)
                    miliseconds = f * miliseconds_per_frame
                    timestamps[w].append(miliseconds)

			print(
				"\tWeek:", w, ": ms/frame:", miliseconds_per_frame, "-- done"
			)


    # save the variable arrays to a dictionary with all the values
    for v in [
        "mouse_list",
        "week_list",
        "frame_list",
        "timestamps",
        "upper_eye_x",
        "upper_eye_y",
        "lower_eye_x",
        "lower_eye_y",
        "upper_ear_x",
        "upper_ear_y",
        "lower_ear_x",
        "lower_ear_y",
        "outer_ear_x",
        "outer_ear_y",
        "upper_whisker_x",
        "upper_whisker_y",
        "outer_whisker_x",
        "outer_whisker_y",
        "lower_whisker_x",
        "lower_whisker_y",
        "upper_mouth_x",
        "upper_mouth_y",
        "outer_mouth_x",
        "outer_mouth_y",
        "lower_mouth_x",
        "lower_mouth_y",
        "inner_nostril_x",
        "inner_nostril_y",
        "outer_nostril_x",
        "outer_nostril_y",
    ]:
        exec("exprs[%s]['%s'] = %s" % (m, v, v))
        exec("del(%s)" % (v))
    print(mouse, "-- complete")
print("Done")

# %%
if platform.system() == "Darwin":
    data_path = f"/Volumes/specialk_cs/2p/raw/"
else:
    data_path = f"/nadata/snlkt/specialk_cs/2p/raw/"

# create trial data
for m, mouse in enumerate(mice):

    print(mouse, "-- starting")

    # complete the path for each week in order of existence in exprs
    # (i.e. week 0 is the first week, week 1 the second)
    n_weeks = len(exprs[m]["timestamps"])

    # create variable arrays with the length of weeks
    week_filter = np.zeros(n_weeks, dtype=bool)
    data = [[] for _ in range(n_weeks)]

    for w in range(n_weeks):

        print(f"\tWeek: {w} starting...")

        led = 0
        speaker = 0
        trial_type = [
            "Airpuff",
            "Sucrose",
            "Airpuff catch",
            "Sucrose catch",
            "Airpuff with LED",
            "Sucrose with LED",
            "LED Only",
        ]

        dataframe = [pd.DataFrame() for _ in range(len(exprs[m]["trialArray"][w]))]

        led_start_array = reduce_led(exprs[m]["LED_on"][w])
        led_end_array = reduce_led(exprs[m]["LED_off"][w])

        exprs[m]["timestamps"][w] = np.array(exprs[m]["timestamps"][w])

        # if the video data has been processed
        if w < len(exprs[m]["timestamps"]):

            print(f"\t\tWeek: {w} processing...")

            for i, trial in enumerate(exprs[m]["trialArray"][w]):

                if trial in [0, 1, 2, 3, 4, 5]:

                    # element to which nearest value is to be found
                    start: float = exprs[m]["speaker_on"][w][speaker] - 10000
                    end: float = exprs[m]["speaker_on"][w][speaker] + 13000

                    # speaker index
                    speaker = speaker + 1

                    # calculate the difference array
                    start_difference_array = np.absolute(
                        exprs[m]["timestamps"][w] - start
                    )
                    end_difference_array = np.absolute(exprs[m]["timestamps"][w] - end)

                    # find the index of minimum element from the array
                    start_index = start_difference_array.argmin()
                    end_index = end_difference_array.argmin()

                    # make a new dataframe for each trial
                    dataframe[i] = pd.DataFrame(
                        {
                            "mouse_list": exprs[m]["mouse_list"][w][
                                start_index : end_index + 1
                            ],
                            "week_list": exprs[m]["week_list"][w][
                                start_index : end_index + 1
                            ],
                            "frame_list": exprs[m]["frame_list"][w][
                                start_index : end_index + 1
                            ],
                            "timestamps": exprs[m]["timestamps"][w][
                                start_index : end_index + 1
                            ],
                            "trial_num": [i for _ in range(start_index, end_index + 1)],
                            "trial_idx": [
                                trial for _ in range(start_index, end_index + 1)
                            ],
                            "trial_type": [
                                trial_type[trial]
                                for _ in range(start_index, end_index + 1)
                            ],
                            "upper_eye_x": exprs[m]["upper_eye_x"][w][
                                start_index : end_index + 1
                            ],
                            "upper_eye_y": exprs[m]["upper_eye_y"][w][
                                start_index : end_index + 1
                            ],
                            "lower_eye_x": exprs[m]["lower_eye_x"][w][
                                start_index : end_index + 1
                            ],
                            "lower_eye_y": exprs[m]["lower_eye_y"][w][
                                start_index : end_index + 1
                            ],
                            "upper_ear_x": exprs[m]["upper_ear_x"][w][
                                start_index : end_index + 1
                            ],
                            "upper_ear_y": exprs[m]["upper_ear_y"][w][
                                start_index : end_index + 1
                            ],
                            "lower_ear_x": exprs[m]["lower_ear_x"][w][
                                start_index : end_index + 1
                            ],
                            "lower_ear_y": exprs[m]["lower_ear_y"][w][
                                start_index : end_index + 1
                            ],
                            "outer_ear_x": exprs[m]["outer_ear_x"][w][
                                start_index : end_index + 1
                            ],
                            "outer_ear_y": exprs[m]["outer_ear_y"][w][
                                start_index : end_index + 1
                            ],
                            "upper_whisker_x": exprs[m]["upper_whisker_x"][w][
                                start_index : end_index + 1
                            ],
                            "upper_whisker_y": exprs[m]["upper_whisker_y"][w][
                                start_index : end_index + 1
                            ],
                            "outer_whisker_x": exprs[m]["outer_whisker_x"][w][
                                start_index : end_index + 1
                            ],
                            "outer_whisker_y": exprs[m]["outer_whisker_y"][w][
                                start_index : end_index + 1
                            ],
                            "lower_whisker_x": exprs[m]["lower_whisker_x"][w][
                                start_index : end_index + 1
                            ],
                            "lower_whisker_y": exprs[m]["lower_whisker_y"][w][
                                start_index : end_index + 1
                            ],
                            "upper_mouth_x": exprs[m]["upper_mouth_x"][w][
                                start_index : end_index + 1
                            ],
                            "upper_mouth_y": exprs[m]["upper_mouth_y"][w][
                                start_index : end_index + 1
                            ],
                            "outer_mouth_x": exprs[m]["outer_mouth_x"][w][
                                start_index : end_index + 1
                            ],
                            "outer_mouth_y": exprs[m]["outer_mouth_y"][w][
                                start_index : end_index + 1
                            ],
                            "lower_mouth_x": exprs[m]["lower_mouth_x"][w][
                                start_index : end_index + 1
                            ],
                            "lower_mouth_y": exprs[m]["lower_mouth_y"][w][
                                start_index : end_index + 1
                            ],
                            "inner_nostril_x": exprs[m]["inner_nostril_x"][w][
                                start_index : end_index + 1
                            ],
                            "inner_nostril_y": exprs[m]["inner_nostril_y"][w][
                                start_index : end_index + 1
                            ],
                            "outer_nostril_x": exprs[m]["outer_nostril_x"][w][
                                start_index : end_index + 1
                            ],
                            "outer_nostril_y": exprs[m]["outer_nostril_y"][w][
                                start_index : end_index + 1
                            ],
                        },
                        columns=[
                            "mouse_list",
                            "week_list",
                            "frame_list",
                            "timestamps",
                            "trial_num",
                            "trial_idx",
                            "trial_type",
                            "upper_eye_x",
                            "upper_eye_y",
                            "lower_eye_x",
                            "lower_eye_y",
                            "upper_ear_x",
                            "upper_ear_y",
                            "lower_ear_x",
                            "lower_ear_y",
                            "outer_ear_x",
                            "outer_ear_y",
                            "upper_whisker_x",
                            "upper_whisker_y",
                            "outer_whisker_x",
                            "outer_whisker_y",
                            "lower_whisker_x",
                            "lower_whisker_y",
                            "upper_mouth_x",
                            "upper_mouth_y",
                            "outer_mouth_x",
                            "outer_mouth_y",
                            "lower_mouth_x",
                            "lower_mouth_y",
                            "inner_nostril_x",
                            "inner_nostril_y",
                            "outer_nostril_x",
                            "outer_nostril_y",
                        ],
                        index=[j for j in range(start_index, end_index + 1)],
                    )

                    if trial in [4, 5]:
                        led = led + 1

                if trial == 6:

                    start = led_start_array[led] - 10000
                    end = led_start_array[led] + 13000

                    # calculate the difference array
                    start_difference_array = np.absolute(
                        exprs[m]["timestamps"][w] - start
                    )
                    end_difference_array = np.absolute(exprs[m]["timestamps"][w] - end)

                    led = led + 1

                    # find the index of minimum element from the array
                    start_index = start_difference_array.argmin()
                    end_index = end_difference_array.argmin()

                    # make a new dataframe for each trial
                    dataframe[i] = pd.DataFrame(
                        {
                            "mouse_list": exprs[m]["mouse_list"][w][
                                start_index : end_index + 1
                            ],
                            "week_list": exprs[m]["week_list"][w][
                                start_index : end_index + 1
                            ],
                            "frame_list": exprs[m]["frame_list"][w][
                                start_index : end_index + 1
                            ],
                            "timestamps": exprs[m]["timestamps"][w][
                                start_index : end_index + 1
                            ],
                            "trial_num": [i for _ in range(start_index, end_index + 1)],
                            "trial_idx": [
                                trial for _ in range(start_index, end_index + 1)
                            ],
                            "trial_type": [
                                trial_type[trial]
                                for _ in range(start_index, end_index + 1)
                            ],
                            "upper_eye_x": exprs[m]["upper_eye_x"][w][
                                start_index : end_index + 1
                            ],
                            "upper_eye_y": exprs[m]["upper_eye_y"][w][
                                start_index : end_index + 1
                            ],
                            "lower_eye_x": exprs[m]["lower_eye_x"][w][
                                start_index : end_index + 1
                            ],
                            "lower_eye_y": exprs[m]["lower_eye_y"][w][
                                start_index : end_index + 1
                            ],
                            "upper_ear_x": exprs[m]["upper_ear_x"][w][
                                start_index : end_index + 1
                            ],
                            "upper_ear_y": exprs[m]["upper_ear_y"][w][
                                start_index : end_index + 1
                            ],
                            "lower_ear_x": exprs[m]["lower_ear_x"][w][
                                start_index : end_index + 1
                            ],
                            "lower_ear_y": exprs[m]["lower_ear_y"][w][
                                start_index : end_index + 1
                            ],
                            "outer_ear_x": exprs[m]["outer_ear_x"][w][
                                start_index : end_index + 1
                            ],
                            "outer_ear_y": exprs[m]["outer_ear_y"][w][
                                start_index : end_index + 1
                            ],
                            "upper_whisker_x": exprs[m]["upper_whisker_x"][w][
                                start_index : end_index + 1
                            ],
                            "upper_whisker_y": exprs[m]["upper_whisker_y"][w][
                                start_index : end_index + 1
                            ],
                            "outer_whisker_x": exprs[m]["outer_whisker_x"][w][
                                start_index : end_index + 1
                            ],
                            "outer_whisker_y": exprs[m]["outer_whisker_y"][w][
                                start_index : end_index + 1
                            ],
                            "lower_whisker_x": exprs[m]["lower_whisker_x"][w][
                                start_index : end_index + 1
                            ],
                            "lower_whisker_y": exprs[m]["lower_whisker_y"][w][
                                start_index : end_index + 1
                            ],
                            "upper_mouth_x": exprs[m]["upper_mouth_x"][w][
                                start_index : end_index + 1
                            ],
                            "upper_mouth_y": exprs[m]["upper_mouth_y"][w][
                                start_index : end_index + 1
                            ],
                            "outer_mouth_x": exprs[m]["outer_mouth_x"][w][
                                start_index : end_index + 1
                            ],
                            "outer_mouth_y": exprs[m]["outer_mouth_y"][w][
                                start_index : end_index + 1
                            ],
                            "lower_mouth_x": exprs[m]["lower_mouth_x"][w][
                                start_index : end_index + 1
                            ],
                            "lower_mouth_y": exprs[m]["lower_mouth_y"][w][
                                start_index : end_index + 1
                            ],
                            "inner_nostril_x": exprs[m]["inner_nostril_x"][w][
                                start_index : end_index + 1
                            ],
                            "inner_nostril_y": exprs[m]["inner_nostril_y"][w][
                                start_index : end_index + 1
                            ],
                            "outer_nostril_x": exprs[m]["outer_nostril_x"][w][
                                start_index : end_index + 1
                            ],
                            "outer_nostril_y": exprs[m]["outer_nostril_y"][w][
                                start_index : end_index + 1
                            ],
                        },
                        columns=[
                            "mouse_list",
                            "week_list",
                            "frame_list",
                            "timestamps",
                            "trial_num",
                            "trial_idx",
                            "trial_type",
                            "upper_eye_x",
                            "upper_eye_y",
                            "lower_eye_x",
                            "lower_eye_y",
                            "upper_ear_x",
                            "upper_ear_y",
                            "lower_ear_x",
                            "lower_ear_y",
                            "outer_ear_x",
                            "outer_ear_y",
                            "upper_whisker_x",
                            "upper_whisker_y",
                            "outer_whisker_x",
                            "outer_whisker_y",
                            "lower_whisker_x",
                            "lower_whisker_y",
                            "upper_mouth_x",
                            "upper_mouth_y",
                            "outer_mouth_x",
                            "outer_mouth_y",
                            "lower_mouth_x",
                            "lower_mouth_y",
                            "inner_nostril_x",
                            "inner_nostril_y",
                            "outer_nostril_x",
                            "outer_nostril_y",
                        ],
                        index=[j for j in range(start_index, end_index + 1)],
                    )

        print(f"\tWeek: {w} ending...")

        dataframe = [
            d
            for i, d in enumerate(dataframe)
            if type(dataframe[i]) == type(pd.DataFrame())
        ]

        data[w] = pd.concat(dataframe, keys=[i for i in range(len(dataframe))])

        if week_filter[w]:
            print("\t\tWeek:", w, "Num Trials:", len(dataframe), "-- done")

        del dataframe

    if len(data) > 0:
        data = pd.concat(data, keys=[w for w in range(len(data))])
        exprs[m]["trial_data_by_mouse"] = data.dropna(
            how="any",
            subset=[
                "upper_eye_x",
                "upper_eye_y",
                "lower_eye_x",
                "lower_eye_y",
                "upper_ear_x",
                "upper_ear_y",
                "lower_ear_x",
                "lower_ear_y",
                "outer_ear_x",
                "outer_ear_y",
                "upper_whisker_x",
                "upper_whisker_y",
                "outer_whisker_x",
                "outer_whisker_y",
                "lower_whisker_x",
                "lower_whisker_y",
                "upper_mouth_x",
                "upper_mouth_y",
                "outer_mouth_x",
                "outer_mouth_y",
                "lower_mouth_x",
                "lower_mouth_y",
                "inner_nostril_x",
                "inner_nostril_y",
                "outer_nostril_x",
                "outer_nostril_y",
            ],
        )

    del data

    print(mouse, "-- complete")

print("Done")

# %%
# create data frame of weekly data
columns = [
    "mouse_list",
    "week_list",
    "frame_list",
    "upper_eye_x",
    "upper_eye_y",
    "lower_eye_x",
    "lower_eye_y",
    "upper_ear_x",
    "upper_ear_y",
    "lower_ear_x",
    "lower_ear_y",
    "outer_ear_x",
    "outer_ear_y",
    "upper_whisker_x",
    "upper_whisker_y",
    "outer_whisker_x",
    "outer_whisker_y",
    "lower_whisker_x",
    "lower_whisker_y",
    "upper_mouth_x",
    "upper_mouth_y",
    "outer_mouth_x",
    "outer_mouth_y",
    "lower_mouth_x",
    "lower_mouth_y",
    "inner_nostril_x",
    "inner_nostril_y",
    "outer_nostril_x",
    "outer_nostril_y",
]

for m, mouse in enumerate(mice):
    n_weeks = len(exprs[m]["timestamps"])

    # create variable arrays with the length of weeks
    # create a list to store data from each week
    dataframe = [{} for _ in range(n_weeks)]
    data = [_ for _ in range(n_weeks)]

    # iterate weeks
    for w in range(n_weeks):

        # for each week, convert the "mouse" variables (for each frame) into a numpy array
        # Begin the dataframe
        dataframe[w][columns[0]] = np.array(exprs[m]["mouse_list"][w], dtype=str)

        # Convert the frame # and week # into numpy arrays
        for v in columns[1:3]:
            dataframe[w][v] = np.array(exprs[m][v][w], dtype=np.int64)

        # mean center the track points
        for v in columns[3:29]:
            if type(exprs[m][v][w]) == type(np.array([0, 1, 2, 3])):

                means = np.mean(exprs[m][v][w], axis=0)

                dataframe[w][v] = exprs[m][v][w] - means

            else:
                print(f"\terror mean centering mouse: {mouse}, dataset: {v}, week: {w}")
        for v in columns[0:3]:
            dataframe[w][v] = dataframe[w][v][0 : dataframe[w][columns[6]].shape[0]]

        # Concatenate all numpy arrays into 'week' pandas dataframe
        data[w] = pd.DataFrame(dataframe[w], columns=columns)

        print(
            "\tWeek:", w, "Num columns:", len(dataframe[w].keys()), "-- done"
        )

    data = pd.concat(data, keys=[w for w in range(len(data))])
    exprs[m]["data_by_mouse"] = data.dropna(
        how="any",
        subset=[
            "upper_eye_x",
            "upper_eye_y",
            "lower_eye_x",
            "lower_eye_y",
            "upper_ear_x",
            "upper_ear_y",
            "lower_ear_x",
            "lower_ear_y",
            "outer_ear_x",
            "outer_ear_y",
            "upper_whisker_x",
            "upper_whisker_y",
            "outer_whisker_x",
            "outer_whisker_y",
            "lower_whisker_x",
            "lower_whisker_y",
            "upper_mouth_x",
            "upper_mouth_y",
            "outer_mouth_x",
            "outer_mouth_y",
            "lower_mouth_x",
            "lower_mouth_y",
            "inner_nostril_x",
            "inner_nostril_y",
            "outer_nostril_x",
            "outer_nostril_y",
        ],
    )
    del data
    del dataframe

    print(mouse, "-- complete")

print("Done")

# %%
# enumerate mice
for m, mouse in enumerate(mice):
    print(mouse, "-- starting")

    # if the video exists and is processed
    if type(exprs[m]["data_by_mouse"]) == type(pd.DataFrame()):

        raw_data = exprs[m]["data_by_mouse"].loc[:, "upper_eye_x":"outer_nostril_y"]
        targets = exprs[m]["data_by_mouse"].loc[:, "mouse_list":"frame_list"]

        centered_data = raw_data.subtract(raw_data.mean())

        centered_data = pd.concat([targets, centered_data], axis=1)

    exprs[m]["data_by_mouse_centered"] = centered_data.dropna(
        how="any",
        subset=[
            "upper_eye_x",
            "upper_eye_y",
            "lower_eye_x",
            "lower_eye_y",
            "upper_ear_x",
            "upper_ear_y",
            "lower_ear_x",
            "lower_ear_y",
            "outer_ear_x",
            "outer_ear_y",
            "upper_whisker_x",
            "upper_whisker_y",
            "outer_whisker_x",
            "outer_whisker_y",
            "lower_whisker_x",
            "lower_whisker_y",
            "upper_mouth_x",
            "upper_mouth_y",
            "outer_mouth_x",
            "outer_mouth_y",
            "lower_mouth_x",
            "lower_mouth_y",
            "inner_nostril_x",
            "inner_nostril_y",
            "outer_nostril_x",
            "outer_nostril_y",
        ],
    )

    print(mouse, "-- complete")

print("Done")



# %%
# enumerate mice
for m, mouse in enumerate(mice):
    print(mouse, "-- starting")

    # if the video exists and is processed
    if type(exprs[m]["trial_data_by_mouse"]) == type(pd.DataFrame()):

        raw_data = exprs[m]["trial_data_by_mouse"].loc[
            :, "upper_eye_x":"outer_nostril_y"
        ]
        targets = exprs[m]["trial_data_by_mouse"].loc[:, "mouse_list":"trial_type"]

        trial_data_centered = raw_data.subtract(raw_data.mean())

        trial_data_centered = pd.concat([targets, trial_data_centered], axis=1)

    exprs[m]["trial_data_by_mouse_centered"] = trial_data_centered.dropna(
        thresh=4,
        subset=[
            "upper_eye_x",
            "upper_eye_y",
            "lower_eye_x",
            "lower_eye_y",
            "upper_ear_x",
            "upper_ear_y",
            "lower_ear_x",
            "lower_ear_y",
            "outer_ear_x",
            "outer_ear_y",
            "upper_whisker_x",
            "upper_whisker_y",
            "outer_whisker_x",
            "outer_whisker_y",
            "lower_whisker_x",
            "lower_whisker_y",
            "upper_mouth_x",
            "upper_mouth_y",
            "outer_mouth_x",
            "outer_mouth_y",
            "lower_mouth_x",
            "lower_mouth_y",
            "inner_nostril_x",
            "inner_nostril_y",
            "outer_nostril_x",
            "outer_nostril_y",
        ],
    )

    del trial_data_centered
    del raw_data
    del targets

    print(mouse, "-- complete")

print("Done")

# %%
dataframes_data = [_ for _ in range(len(mice))]
centered_dataframes_data = [_ for _ in range(len(mice))]
dataframes_trials = [_ for _ in range(len(mice))]
centered_dataframes_trials = [_ for _ in range(len(mice))]
trial_type = [
    "Airpuff",
    "Sucrose",
    "Airpuff catch",
    "Sucrose catch",
    "Airpuff with LED",
    "Sucrose with LED",
    "LED only",
]
trial_type_name = [
    "Airpuff",
    "Sucrose",
    "Airpuff_catch",
    "Sucrose_catch",
    "Airpuff_with_LED",
    "Sucrose_with_LED",
    "LED_only",
]

all_data_and_pcas = {}

for m in range(len(mice)):

    dataframes_data[m] = exprs[m]["data_by_mouse"]
    centered_dataframes_data[m] = exprs[m]["data_by_mouse_centered"]

    targets_list = exprs[m]["data_by_mouse"].loc[:, "mouse_list":"frame_list"]
    centered_targets_list = exprs[m]["data_by_mouse_centered"].loc[
        :, "mouse_list":"frame_list"
    ]

    data_frame = exprs[m]["data_by_mouse"].loc[:, "upper_eye_x":"outer_nostril_y"]
    centered_data_frame = exprs[m]["data_by_mouse_centered"].loc[
        :, "upper_eye_x":"outer_nostril_y"
    ]

    pca = PCA(n_components=2)
    data_frame_pca = pca.fit_transform(data_frame)
    centered_data_frame_pca = pca.fit_transform(centered_data_frame)

    principalDf = pd.DataFrame(
        data_frame_pca, columns=["principal component 1", "principal component 2"]
    )
    centered_principalDf = pd.DataFrame(
        centered_data_frame_pca,
        columns=["principal component 1", "principal component 2"],
    )

    exprs[m]["D2_PCA_by_mouse"] = pd.concat(
        [targets_list.reset_index(), principalDf], axis=1
    )
    exprs[m]["D2_PCA_by_mouse_centered"] = pd.concat(
        [centered_targets_list.reset_index(), centered_principalDf], axis=1
    )

    pca = PCA(n_components=3)

    data_frame_pca = pca.fit_transform(data_frame)
    centered_data_frame_pca = pca.fit_transform(centered_data_frame)

    principalDf = pd.DataFrame(
        data_frame_pca,
        columns=[
            "principal component 1",
            "principal component 2",
            "principal component 3",
        ],
    )
    centered_principalDf = pd.DataFrame(
        centered_data_frame_pca,
        columns=[
            "principal component 1",
            "principal component 2",
            "principal component 3",
        ],
    )

    exprs[m]["D3_PCA_by_mouse"] = pd.concat(
        [targets_list.reset_index(), principalDf], axis=1
    )
    exprs[m]["D3_PCA_by_mouse_centered"] = pd.concat(
        [centered_targets_list.reset_index(), centered_principalDf], axis=1
    )

    dataframes_trials[m] = exprs[m]["trial_data_by_mouse"]
    centered_dataframes_trials[m] = exprs[m]["trial_data_by_mouse_centered"]

    targets_list = exprs[m]["trial_data_by_mouse"].loc[:, "mouse_list":"trial_type"]
    centered_targets_list = exprs[m]["trial_data_by_mouse_centered"].loc[
        :, "mouse_list":"trial_type"
    ]

    data_frame = exprs[m]["trial_data_by_mouse"].loc[:, "upper_eye_x":"outer_nostril_y"]
    centered_data_frame = exprs[m]["trial_data_by_mouse_centered"].loc[
        :, "upper_eye_x":"outer_nostril_y"
    ]

    pca = PCA(n_components=3)
    data_frame_pca = pca.fit_transform(data_frame)
    centered_data_frame_pca = pca.fit_transform(centered_data_frame)

    principalDf = pd.DataFrame(
        data_frame_pca,
        columns=[
            "principal component 1",
            "principal component 2",
            "principal component 3",
        ],
    )
    centered_principalDf = pd.DataFrame(
        centered_data_frame_pca,
        columns=[
            "principal component 1",
            "principal component 2",
            "principal component 3",
        ],
    )

    finalDf = pd.concat([targets_list.reset_index(), principalDf], axis=1)
    centered_finalDf = pd.concat(
        [centered_targets_list.reset_index(), centered_principalDf], axis=1
    )

    finalDf.loc[:, ["mouse_list", "week_list"]] = finalDf.astype(
        {"mouse_list": str, "week_list": str}
    )
    centered_finalDf.loc[:, ["mouse_list", "week_list"]] = centered_finalDf.astype(
        {"mouse_list": str, "week_list": str}
    )

    exprs[m]["D3_PCA_raw_trial_data_by_mouse"] = finalDf
    exprs[m]["D3_PCA_trial_data_centered_by_mouse"] = centered_finalDf

    pca = PCA(n_components=2)
    data_frame_pca = pca.fit_transform(data_frame)
    centered_data_frame_pca = pca.fit_transform(centered_data_frame)

    principalDf = pd.DataFrame(
        data_frame_pca, columns=["principal component 1", "principal component 2"]
    )
    centered_principalDf = pd.DataFrame(
        centered_data_frame_pca,
        columns=["principal component 1", "principal component 2"],
    )

    exprs[m]["D2_PCA_raw_trial_data_by_mouse"] = pd.concat(
        [targets_list.reset_index(), principalDf], axis=1
    )
    exprs[m]["D2_PCA_trial_data_centered_by_mouse"] = pd.concat(
        [centered_targets_list.reset_index(), centered_principalDf], axis=1
    )

all_data_and_pcas["all_data"] = pd.concat(dataframes_data, keys=mice)
all_data_and_pcas["all_data_centered"] = pd.concat(centered_dataframes_data, keys=mice)
all_data_and_pcas["all_trial_data"] = pd.concat(dataframes_trials, keys=mice)
all_data_and_pcas["all_trial_data_centered"] = pd.concat(
    centered_dataframes_trials, keys=mice
)

data = all_data_and_pcas["all_trial_data"]
centered_data = all_data_and_pcas["all_trial_data_centered"]

non_numeric_cols = data.loc[:, "mouse_list":"trial_type"]
non_numeric_cols_centered = centered_data.loc[:, "mouse_list":"trial_type"]

numeric_cols = data.loc[:, "upper_eye_x":"outer_nostril_y"]
centered_numeric_cols = centered_data.loc[:, "upper_eye_x":"outer_nostril_y"]

pca = PCA(n_components=2)

data_frame_pca = pca.fit_transform(numeric_cols)
data_frame_pca_centered = pca.fit_transform(centered_numeric_cols)

principalDf = pd.DataFrame(
    data_frame_pca, columns=["principal component 1", "principal component 2"]
)
principalDf_centered = pd.DataFrame(
    data_frame_pca_centered, columns=["principal component 1", "principal component 2"]
)

all_data_and_pcas["D2_PCA_trial_data"] = pd.concat(
    [non_numeric_cols.reset_index(), principalDf], axis=1
)
all_data_and_pcas["D2_PCA_trial_data_centered"] = pd.concat(
    [non_numeric_cols_centered.reset_index(), principalDf_centered], axis=1
)

std_x = statistics.stdev(principalDf.std(0).to_list())
std_y = statistics.stdev(principalDf.std(1).to_list())
cen_std_x = statistics.stdev(principalDf_centered.std(0).to_list())
cen_std_y = statistics.stdev(principalDf_centered.std(1).to_list())

principalDf_blurred = cv2.GaussianBlur(
    principalDf.to_numpy(), (3, 3), sigmaX=std_x, sigmaY=std_y
)
principalDf_blurred = pd.DataFrame(
    principalDf_blurred,
    columns=[
        "principal component 1",
        "principal component 2",
    ],
)

principalDf_centered_blurred = cv2.GaussianBlur(
    principalDf_centered.to_numpy(), (3, 3), sigmaX=cen_std_x, sigmaY=cen_std_y
)
principalDf_centered_blurred = pd.DataFrame(
    principalDf_centered_blurred,
    columns=["principal component 1", "principal component 2"],
)

all_data_and_pcas["D2_PCA_trial_data_blurred"] = pd.concat(
    [non_numeric_cols.reset_index(), principalDf_blurred], axis=1
)
all_data_and_pcas["D2_PCA_trial_data_centered_blurred"] = pd.concat(
    [non_numeric_cols_centered.reset_index(), principalDf_centered_blurred], axis=1
)

pca = PCA(n_components=3)

data_frame_pca = pca.fit_transform(numeric_cols)
data_frame_pca_centered = pca.fit_transform(centered_numeric_cols)

principalDf = pd.DataFrame(
    data_frame_pca,
    columns=["principal component 1", "principal component 2", "principal component 3"],
)
principalDf_centered = pd.DataFrame(
    data_frame_pca_centered,
    columns=["principal component 1", "principal component 2", "principal component 3"],
)

all_data_and_pcas["D3_PCA_trial_data"] = pd.concat(
    [non_numeric_cols.reset_index(), principalDf], axis=1
)
all_data_and_pcas["D3_PCA_trial_data_centered"] = pd.concat(
    [non_numeric_cols_centered.reset_index(), principalDf_centered], axis=1
)

std_x = statistics.stdev(principalDf.std(0).to_list())
std_y = statistics.stdev(principalDf.std(1).to_list())
cen_std_x = statistics.stdev(principalDf_centered.std(0).to_list())
cen_std_y = statistics.stdev(principalDf_centered.std(1).to_list())

principalDf_blurred = cv2.GaussianBlur(
    principalDf.to_numpy(), (3, 3), sigmaX=std_x, sigmaY=std_y
)
principalDf_blurred = pd.DataFrame(
    principalDf_blurred,
    columns=["principal component 1", "principal component 2", "principal component 3"],
)

principalDf_centered_blurred = cv2.GaussianBlur(
    principalDf_centered.to_numpy(), (3, 3), sigmaX=cen_std_x, sigmaY=cen_std_y
)
principalDf_centered_blurred = pd.DataFrame(
    principalDf_centered_blurred,
    columns=["principal component 1", "principal component 2", "principal component 3"],
)

all_data_and_pcas["D2_PCA_trial_data_blurred"] = pd.concat(
    [non_numeric_cols.reset_index(), principalDf_blurred], axis=1
)
all_data_and_pcas["D2_PCA_trial_data_centered_blurred"] = pd.concat(
    [non_numeric_cols_centered.reset_index(), principalDf_centered_blurred], axis=1
)

for i in range(len(trial_type)):
    if not data[data.trial_type.isin([trial_type[i]])].empty:
        data = data[data.trial_type == trial_type[i]]
        centered_data = centered_data[centered_data.trial_type == trial_type[i]]

        non_numeric_cols = data.loc[:, "mouse_list":"trial_type"]
        non_numeric_cols_centered = centered_data.loc[:, "mouse_list":"trial_type"]

        numeric_cols = data.loc[:, "upper_eye_x":"outer_nostril_y"]
        centered_numeric_cols = centered_data.loc[:, "upper_eye_x":"outer_nostril_y"]

        pca = PCA(n_components=2)

        data_frame_pca = pca.fit_transform(numeric_cols)
        data_frame_pca_centered = pca.fit_transform(centered_numeric_cols)

        principalDf = pd.DataFrame(
            data_frame_pca, columns=["principal component 1", "principal component 2"]
        )
        principalDf_centered = pd.DataFrame(
            data_frame_pca_centered,
            columns=["principal component 1", "principal component 2"],
        )

        all_data_and_pcas[f"D2_PCA_{trial_type_name[i]}"] = pd.concat(
            [non_numeric_cols.reset_index(), principalDf], axis=1
        )
        all_data_and_pcas[f"D2_PCA_{trial_type_name[i]}_centered"] = pd.concat(
            [non_numeric_cols_centered.reset_index(), principalDf_centered], axis=1
        )

        std_x = statistics.stdev(principalDf.std(0).to_list())
        std_y = statistics.stdev(principalDf.std(1).to_list())
        cen_std_x = statistics.stdev(principalDf_centered.std(0).to_list())
        cen_std_y = statistics.stdev(principalDf_centered.std(1).to_list())

        principalDf_blurred = cv2.GaussianBlur(
            principalDf.to_numpy(), (3, 3), sigmaX=std_x, sigmaY=std_y
        )
        principalDf_blurred = pd.DataFrame(
            principalDf_blurred,
            columns=[
                "principal component 1",
                "principal component 2",
            ],
        )

        principalDf_centered_blurred = cv2.GaussianBlur(
            principalDf_centered.to_numpy(), (3, 3), sigmaX=cen_std_x, sigmaY=cen_std_y
        )
        principalDf_centered_blurred = pd.DataFrame(
            principalDf_centered_blurred,
            columns=["principal component 1", "principal component 2"],
        )

        all_data_and_pcas[f"D2_PCA_{trial_type_name[i]}_blurred"] = pd.concat(
            [non_numeric_cols.reset_index(), principalDf_blurred], axis=1
        )
        all_data_and_pcas[f"D2_PCA_{trial_type_name[i]}_centered_blurred"] = pd.concat(
            [non_numeric_cols_centered.reset_index(), principalDf_centered_blurred],
            axis=1,
        )

        pca = PCA(n_components=3)

        data_frame_pca = pca.fit_transform(numeric_cols)
        data_frame_pca_centered = pca.fit_transform(centered_numeric_cols)

        principalDf = pd.DataFrame(
            data_frame_pca,
            columns=[
                "principal component 1",
                "principal component 2",
                "principal component 3",
            ],
        )
        principalDf_centered = pd.DataFrame(
            data_frame_pca_centered,
            columns=[
                "principal component 1",
                "principal component 2",
                "principal component 3",
            ],
        )

        all_data_and_pcas[f"D3_PCA_{trial_type_name[i]}"] = pd.concat(
            [non_numeric_cols.reset_index(), principalDf], axis=1
        )
        all_data_and_pcas[f"D3_PCA_{trial_type_name[i]}_centered"] = pd.concat(
            [non_numeric_cols_centered.reset_index(), principalDf_centered], axis=1
        )

        std_x = statistics.stdev(principalDf.std(0).to_list())
        std_y = statistics.stdev(principalDf.std(1).to_list())
        cen_std_x = statistics.stdev(principalDf_centered.std(0).to_list())
        cen_std_y = statistics.stdev(principalDf_centered.std(1).to_list())

        principalDf_blurred = cv2.GaussianBlur(
            principalDf.to_numpy(), (3, 3), sigmaX=std_x, sigmaY=std_y
        )
        principalDf_blurred = pd.DataFrame(
            principalDf_blurred,
            columns=[
                "principal component 1",
                "principal component 2",
                "principal component 3",
            ],
        )

        principalDf_centered_blurred = cv2.GaussianBlur(
            principalDf_centered.to_numpy(), (3, 3), sigmaX=cen_std_x, sigmaY=cen_std_y
        )
        principalDf_centered_blurred = pd.DataFrame(
            principalDf_centered_blurred,
            columns=[
                "principal component 1",
                "principal component 2",
                "principal component 3",
            ],
        )

        all_data_and_pcas[f"D3_PCA_{trial_type_name[i]}_blurred"] = pd.concat(
            [non_numeric_cols.reset_index(), principalDf_blurred], axis=1
        )
        all_data_and_pcas[f"D3_PCA_{trial_type_name[i]}_centered_blurred"] = pd.concat(
            [non_numeric_cols_centered.reset_index(), principalDf_centered_blurred],
            axis=1,
        )

# %%
# export data
with h5py.File(
    "/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/metadata.h5", "w"
) as hf:

	for m, mouse in enumerate(mice):
		print(f"{mouse} -- saving metadata")

		for key in list(exprs[m].keys()):
			if type(exprs[m][key]) is list or type(exprs[m][key]) is np.array:
				for w in range(len(exprs[m][key])):

					if type(exprs[m][key][w]) is list or type(exprs[m][key][w]) is np.array:
						hf.create_dataset(f"{mouse}/{w}/{key}", data=exprs[m][key][w])

					if type(exprs[m][key][w]) is dict:
						g = hf.create_group(f"{mouse}/{w}/{key}")
						hdfdict.dump(data=exprs[m][key][w], hdf=g)

with pd.HDFStore(
    "/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/datatables.h5", "a"
) as hf:
	for m, mouse in enumerate(mice):
		print(f"{mouse} -- saving datatables")

		for key in list(exprs[m].keys()):
			if type(exprs[m][key]) is pd.DataFrame:
				hf.put(f"{mouse}/{key}", exprs[m][key])


with pd.HDFStore(
    "/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/data.h5", "a"
) as hf:
	print(f"Saving concatenated data")
	for key in list(all_data_and_pcas.keys()):
		hf.put(f"{key}", all_data_and_pcas[key])
