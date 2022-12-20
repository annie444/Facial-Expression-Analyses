#%%
import glob
import json
import os
import platform
import statistics
import ffmpeg
import h5py
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import plotly.express as px
import plotly.graph_objects as go
from math import isnan
from scipy.stats import zscore


#%%
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
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                            y[~mask])

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
    to_ret[:window_size // 2] = 0
    return to_ret


def reduce_led(iterable):
    list = []
    j = 0
    list.append(iterable[j])
    for i in range(0, len(iterable)):
        if iterable[j] < (iterable[i] - 5000):
            j = i
            list.append(iterable[j])
    return list


#%%
# define the paths and the mice
if platform.system() == 'Darwin':
    data_path = f"/Volumes/specialk_cs/2p/raw/"
else:
    data_path = f"/nadata/snlkt/specialk_cs/2p/raw/"

mice = ["CSC009", "CSC013", "CSE008", "CSE020"]

# create a list of dictionaries of lists to sort by
# mouse (list) then variable name (dictionary) and variable values (list)
exprs = [{} for _ in range(len(mice))]

#%%
for m, mouse in enumerate(mice):
    print(mouse, "-- starting")

    # get each week folder from mouse base directory
    _, weeks, _ = next(os.walk(data_path + mouse), ([], [], []))

    # complete the path for each week in order of completion
    # (i.e. week 0 is the first week, week 1 the second)
    weeks.sort()
    n_weeks = len(weeks)
    base_path = data_path + mouse
    week_filter = np.zeros(n_weeks, dtype=bool)
    data = [[] for _ in range(n_weeks)]

    # Create behavior arrays for each varriable
    # with the length as the number of weeks
    airpuff_on = [[] for _ in range(n_weeks)]
    airpuff_off = [[] for _ in range(n_weeks)]

    licks_on = [[] for _ in range(n_weeks)]
    licks_off = [[] for _ in range(n_weeks)]

    sucrose_on = [[] for _ in range(n_weeks)]
    sucrose_off = [[] for _ in range(n_weeks)]

    LED_on = [[] for _ in range(n_weeks)]
    LED_off = [[] for _ in range(n_weeks)]

    speaker_on = [[] for _ in range(n_weeks)]
    speaker_off = [[] for _ in range(n_weeks)]

    video_metadata = [[] for _ in range(n_weeks)]
    beh_metadata = [[] for _ in range(n_weeks)]
    trialArray = [[] for _ in range(n_weeks)]
    ITIArray = [[] for _ in range(n_weeks)]

    # enumerate each week folder
    for w, week in enumerate(weeks):
        os.chdir(f"{base_path}/{weeks[w]}/")

        # get data from the csv files
        for csv in glob.glob("*.csv"):
            data[w] = pd.read_csv(f"{base_path}/{weeks[w]}/{csv}")

        # get data from the json files
        for js in glob.glob("*.json"):
            with open(f"{base_path}/{weeks[w]}/{js}", "r") as js_file:
                js_file = json.load(js_file)
                beh_metadata[w] = js_file.get("beh_metadata")
                trialArray[w] = js_file.get("beh_metadata")["trialArray"]
                ITIArray[w] = js_file.get("beh_metadata")["ITIArray"]

        # get the video metadata
        for video in glob.glob("*.mp4"):
            if len(glob.glob("*.mp4")) > 1:
                continue
            else:
                video_metadata[w] = ffmpeg.probe(
                    f"{base_path}/{weeks[w]}/{video}")["streams"][(int(
                        ffmpeg.probe(f"{base_path}/{weeks[w]}/{video}")
                        ["format"]["nb_streams"]) - 1)]

        # set the week to "True" since it exists
        # this is necessary as some weeks don't have data (yet)
        week_filter[w] = True

        # save list of values to variable arrays omitting NaN values
        airpuff_on[w] = [x for x in data[w]["Airpuff_on"] if isnan(x) == False]
        airpuff_off[w] = [
            x for x in data[w]["Airpuff_off"] if isnan(x) == False
        ]

        sucrose_on[w] = [x for x in data[w]["Sucrose_on"] if isnan(x) == False]
        sucrose_off[w] = [
            x for x in data[w]["Sucrose_off"] if isnan(x) == False
        ]

        LED_on[w] = [x for x in data[w]["LED590_on"] if isnan(x) == False]
        LED_off[w] = [x for x in data[w]["LED590_off"] if isnan(x) == False]

        speaker_on[w] = [x for x in data[w]["Speaker_on"] if isnan(x) == False]
        speaker_off[w] = [
            x for x in data[w]["Speaker_off"] if isnan(x) == False
        ]

        licks_on[w] = [x for x in data[w]["Lick_on"] if isnan(x) == False]
        licks_off[w] = [x for x in data[w]["Lick_off"] if isnan(x) == False]
        if week_filter[w]: print("\tWeek:", w, ": Date:", week, "-- done")

    # filter the variable arrays to be only the lenght of weeks with data
    # this makes sure that we don't have a week of empty values in our variable arrays
    weeks = np.array(weeks)[week_filter]
    data = [d for i, d in enumerate(data) if week_filter[i]]

    airpuff_on = [d for i, d in enumerate(airpuff_on) if week_filter[i]]
    airpuff_off = [d for i, d in enumerate(airpuff_off) if week_filter[i]]

    sucrose_on = [d for i, d in enumerate(sucrose_on) if week_filter[i]]
    sucrose_off = [d for i, d in enumerate(sucrose_off) if week_filter[i]]

    LED_on = [d for i, d in enumerate(LED_on) if week_filter[i]]
    LED_off = [d for i, d in enumerate(LED_off) if week_filter[i]]

    speaker_on = [d for i, d in enumerate(speaker_on) if week_filter[i]]
    speaker_off = [d for i, d in enumerate(speaker_off) if week_filter[i]]

    licks_on = [d for i, d in enumerate(licks_on) if week_filter[i]]
    licks_off = [d for i, d in enumerate(licks_off) if week_filter[i]]

    video_metadata = [
        d for i, d in enumerate(video_metadata) if week_filter[i]
    ]
    beh_metadata = [d for i, d in enumerate(beh_metadata) if week_filter[i]]
    trialArray = [d for i, d in enumerate(trialArray) if week_filter[i]]
    ITIArray = [d for i, d in enumerate(ITIArray) if week_filter[i]]

    # save the variable arrays to a dictionary with all the values
    for v in [
            "airpuff_on", "airpuff_off", "sucrose_on", "sucrose_off", "LED_on",
            "LED_off", "speaker_on", "speaker_off", "licks_on", "licks_off",
            "video_metadata", "beh_metadata", "trialArray", "ITIArray"
    ]:
        exec("exprs[%s]['%s'] = %s" % (m, v, v))
        exec("del(%s)" % (v))
    print(mouse, "-- complete")
print("Done")

#%%
# change path to point to the SLEAP datasets
if platform.system() == 'Darwin':
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
        if week_filter[w]: print("\tWeek:", w, ": File:", week, "-- done")

    # filter out the weeks without data
    weeks = np.array(weeks)[week_filter]
    datasets = [d for i, d in enumerate(datasets) if week_filter[i]]
    tracking_locations = [
        d for i, d in enumerate(tracking_locations) if week_filter[i]
    ]
    edge_inds = [d for i, d in enumerate(edge_inds) if week_filter[i]]
    edge_names = [d for i, d in enumerate(edge_names) if week_filter[i]]
    instance_scores = [
        d for i, d in enumerate(instance_scores) if week_filter[i]
    ]
    point_scores = [d for i, d in enumerate(point_scores) if week_filter[i]]
    track_occupancy = [
        d for i, d in enumerate(track_occupancy) if week_filter[i]
    ]
    tracking_scores = [
        d for i, d in enumerate(tracking_scores) if week_filter[i]
    ]
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

print("Cleaning...")
del (base_path)

#%%
# set original data path
if platform.system() == 'Darwin':
    data_path = f"/Volumes/specialk_cs/2p/raw/"
else:
    data_path = f"/nadata/snlkt/specialk_cs/2p/raw/"

#%%
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
    # create a list with the length # of weeks for each varaible
    mouse_list = [[] for _ in range(n_weeks)]
    week_list = [[] for _ in range(n_weeks)]
    frame_list = [[] for _ in range(n_weeks)]
    timestamps = [[] for _ in range(n_weeks)]

    for v in [
            "upper_eye_x", "upper_eye_y", "lower_eye_x", "lower_eye_y",
            "upper_ear_x", "upper_ear_y", "lower_ear_x", "lower_ear_y",
            "outer_ear_x", "outer_ear_y", "upper_whisker_x", "upper_whisker_y",
            "outer_whisker_x", "outer_whisker_y", "lower_whisker_x",
            "lower_whisker_y", "upper_mouth_x", "upper_mouth_y",
            "outer_mouth_x", "outer_mouth_y", "lower_mouth_x", "lower_mouth_y",
            "inner_nostril_x", "inner_nostril_y", "outer_nostril_x",
            "outer_nostril_y"
    ]:
        # create a list with the length # of weeks for each varaible
        code = compile(f"{v} = [ _ for _ in range({n_weeks})]", "assign",
                       "exec")
        exec(code, globals(), locals())

    # iterate each frame in the SLEAP tracks
    for w in range(0, len(exprs[m]["tracking_locations"]), 1):

        # if the video exists and is processed
        if type(exprs[m]["video_metadata"][w]) == type(dict()):

            week_filter[w] = True

            # calculate miliseconds per frame based on the video metadata
            miliseconds_per_frame = (
                eval(exprs[m]["video_metadata"][w].get("avg_frame_rate")) /
                1000)**-1

            # enumerate each node (or point) in the SLEAP tracks
            for i, name in enumerate(exprs[m]["node_names"][w]):

                # break down the complex 4D array into 1D arrays of x and y values
                exec(
                    "%s_x[%s] = np.array(%s)" %
                    (name.replace(" ", "_"), w,
                     exprs[m]['tracking_locations'][w][:, i, 0, 0].tolist()),
                    globals(), locals())
                exec(
                    "%s_y[%s] = np.array(%s)" %
                    (name.replace(" ", "_"), w,
                     exprs[m]['tracking_locations'][w][:, i, 1, 0].tolist()),
                    globals(), locals())

                # iterate each frame
                for f in range(
                        len(exprs[m]["tracking_locations"][w][:, i, 0,
                                                              0].tolist())):

                    # label frame with the specific mouse, week, frame, and timestamp
                    mouse_list[w].append(mouse)
                    week_list[w].append(w)
                    frame_list[w].append(f)
                    miliseconds = f * miliseconds_per_frame
                    timestamps[w].append(miliseconds)

        print("\tWeek:", w, ": ms/frame:", miliseconds_per_frame,
              "-- done") if week_filter[w] else print(
                  f"No data for mouse {mice[m]} on week {w}")

    # filter out the weeks without data (using exec function for redundancy)
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
        executable = compile(
            "%s = [d for i, d in enumerate(%s) if week_filter[i]]" % (v, v),
            'filter', 'exec')
        exec(executable, globals(), locals())

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

print("Cleaning...")
del (executable)
del (i)
del (m)
del (miliseconds)
del (miliseconds_per_frame)
del (mouse)
del (name)
del (v)
del (w)

#%%
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
            "Airpuff", "Sucrose", "Airpuff catch", "Sucrose catch",
            "Airpuff with LED", "Sucrose with LED", "LED Only"
        ]

        dataframe = [
            pd.DataFrame() for _ in range(len(exprs[m]['trialArray'][w]))
        ]

        # if the video data has been processed
        if w < len(exprs[m]["timestamps"]):

            print(f"\t\tWeek: {w} processing...")

            led_start_array = reduce_led(exprs[m]['LED_on'][w])
            led_end_array = reduce_led(exprs[m]['LED_off'][w])

            exprs[m]['timestamps'][w] = np.array(exprs[m]['timestamps'][w])

            for i, trial in enumerate(exprs[m]['trialArray'][w]):

                print(f"\t\tWeek: {w} iterating...")

                week_filter[w] = True

                if trial in [0, 1, 2, 3, 4, 5]:

                    # element to which nearest value is to be found
                    start: float = exprs[m]['speaker_on'][w][speaker] - 3000
                    end: float = exprs[m]['speaker_on'][w][speaker] + 6000

                    # speaker index
                    speaker = speaker + 1

                    # calculate the difference array
                    start_difference_array = np.absolute(
                        exprs[m]['timestamps'][w] - start)
                    end_difference_array = np.absolute(
                        exprs[m]['timestamps'][w] - end)

                    # find the index of minimum element from the array
                    start_index = start_difference_array.argmin()
                    end_index = end_difference_array.argmin()

                    # make a new dataframe for each trial
                    dataframe[i] = pd.DataFrame(
                        {
                            "mouse_list":
                            exprs[m]['mouse_list'][w][start_index:end_index +
                                                      1],
                            "week_list":
                            exprs[m]['week_list'][w][start_index:end_index +
                                                     1],
                            "frame_list":
                            exprs[m]['frame_list'][w][start_index:end_index +
                                                      1],
                            "timestamps":
                            exprs[m]['timestamps'][w][start_index:end_index +
                                                      1],
                            "trial_num":
                            [i for _ in range(start_index, end_index + 1)],
                            "trial_idx":
                            [trial for _ in range(start_index, end_index + 1)],
                            "trial_type": [
                                trial_type[trial]
                                for _ in range(start_index, end_index + 1)
                            ],
                            "upper_eye_x":
                            exprs[m]['upper_eye_x'][w][start_index:end_index +
                                                       1],
                            "upper_eye_y":
                            exprs[m]['upper_eye_y'][w][start_index:end_index +
                                                       1],
                            "lower_eye_x":
                            exprs[m]['lower_eye_x'][w][start_index:end_index +
                                                       1],
                            "lower_eye_y":
                            exprs[m]['lower_eye_y'][w][start_index:end_index +
                                                       1],
                            "upper_ear_x":
                            exprs[m]['upper_ear_x'][w][start_index:end_index +
                                                       1],
                            "upper_ear_y":
                            exprs[m]['upper_ear_y'][w][start_index:end_index +
                                                       1],
                            "lower_ear_x":
                            exprs[m]['lower_ear_x'][w][start_index:end_index +
                                                       1],
                            "lower_ear_y":
                            exprs[m]['lower_ear_y'][w][start_index:end_index +
                                                       1],
                            "outer_ear_x":
                            exprs[m]['outer_ear_x'][w][start_index:end_index +
                                                       1],
                            "outer_ear_y":
                            exprs[m]['outer_ear_y'][w][start_index:end_index +
                                                       1],
                            "upper_whisker_x":
                            exprs[m]['upper_whisker_x'][w]
                            [start_index:end_index + 1],
                            "upper_whisker_y":
                            exprs[m]['upper_whisker_y'][w]
                            [start_index:end_index + 1],
                            "outer_whisker_x":
                            exprs[m]['outer_whisker_x'][w]
                            [start_index:end_index + 1],
                            "outer_whisker_y":
                            exprs[m]['outer_whisker_y'][w]
                            [start_index:end_index + 1],
                            "lower_whisker_x":
                            exprs[m]['lower_whisker_x'][w]
                            [start_index:end_index + 1],
                            "lower_whisker_y":
                            exprs[m]['lower_whisker_y'][w]
                            [start_index:end_index + 1],
                            "upper_mouth_x":
                            exprs[m]['upper_mouth_x'][w]
                            [start_index:end_index + 1],
                            "upper_mouth_y":
                            exprs[m]['upper_mouth_y'][w]
                            [start_index:end_index + 1],
                            "outer_mouth_x":
                            exprs[m]['outer_mouth_x'][w]
                            [start_index:end_index + 1],
                            "outer_mouth_y":
                            exprs[m]['outer_mouth_y'][w]
                            [start_index:end_index + 1],
                            "lower_mouth_x":
                            exprs[m]['lower_mouth_x'][w]
                            [start_index:end_index + 1],
                            "lower_mouth_y":
                            exprs[m]['lower_mouth_y'][w]
                            [start_index:end_index + 1],
                            "inner_nostril_x":
                            exprs[m]['inner_nostril_x'][w]
                            [start_index:end_index + 1],
                            "inner_nostril_y":
                            exprs[m]['inner_nostril_y'][w]
                            [start_index:end_index + 1],
                            "outer_nostril_x":
                            exprs[m]['outer_nostril_x'][w]
                            [start_index:end_index + 1],
                            "outer_nostril_y":
                            exprs[m]['outer_nostril_y'][w]
                            [start_index:end_index + 1]
                        },
                        columns=[
                            "mouse_list", "week_list", "frame_list",
                            "timestamps", "trial_num", "trial_idx",
                            "trial_type", "upper_eye_x", "upper_eye_y",
                            "lower_eye_x", "lower_eye_y", "upper_ear_x",
                            "upper_ear_y", "lower_ear_x", "lower_ear_y",
                            "outer_ear_x", "outer_ear_y", "upper_whisker_x",
                            "upper_whisker_y", "outer_whisker_x",
                            "outer_whisker_y", "lower_whisker_x",
                            "lower_whisker_y", "upper_mouth_x",
                            "upper_mouth_y", "outer_mouth_x", "outer_mouth_y",
                            "lower_mouth_x", "lower_mouth_y",
                            "inner_nostril_x", "inner_nostril_y",
                            "outer_nostril_x", "outer_nostril_y"
                        ],
                        index=[j for j in range(start_index, end_index + 1)])

                    if trial in [4, 5]:
                        led = led + 1

                elif trial in [6]:
                    start = led_start_array[led] - 3000
                    end = led_start_array[led] + 6000

                    # calculate the difference array
                    start_difference_array = np.absolute(
                        exprs[m]['timestamps'][w] - start)
                    end_difference_array = np.absolute(
                        exprs[m]['timestamps'][w] - end)

                    # find the index of minimum element from the array
                    start_index = start_difference_array.argmin()
                    end_index = end_difference_array.argmin()

                    # make a new dataframe for each trial
                    dataframe[i] = pd.DataFrame(
                        {
                            "mouse_list":
                            exprs[m]['mouse_list'][w][start_index:end_index +
                                                      1],
                            "week_list":
                            exprs[m]['week_list'][w][start_index:end_index +
                                                     1],
                            "frame_list":
                            exprs[m]['frame_list'][w][start_index:end_index +
                                                      1],
                            "timestamps":
                            exprs[m]['timestamps'][w][start_index:end_index +
                                                      1],
                            "trial_num":
                            [i for _ in range(start_index, end_index + 1)],
                            "trial_idx":
                            [trial for _ in range(start_index, end_index + 1)],
                            "trial_type": [
                                trial_type[trial]
                                for _ in range(start_index, end_index + 1)
                            ],
                            "upper_eye_x":
                            exprs[m]['upper_eye_x'][w][start_index:end_index +
                                                       1],
                            "upper_eye_y":
                            exprs[m]['upper_eye_y'][w][start_index:end_index +
                                                       1],
                            "lower_eye_x":
                            exprs[m]['lower_eye_x'][w][start_index:end_index +
                                                       1],
                            "lower_eye_y":
                            exprs[m]['lower_eye_y'][w][start_index:end_index +
                                                       1],
                            "upper_ear_x":
                            exprs[m]['upper_ear_x'][w][start_index:end_index +
                                                       1],
                            "upper_ear_y":
                            exprs[m]['upper_ear_y'][w][start_index:end_index +
                                                       1],
                            "lower_ear_x":
                            exprs[m]['lower_ear_x'][w][start_index:end_index +
                                                       1],
                            "lower_ear_y":
                            exprs[m]['lower_ear_y'][w][start_index:end_index +
                                                       1],
                            "outer_ear_x":
                            exprs[m]['outer_ear_x'][w][start_index:end_index +
                                                       1],
                            "outer_ear_y":
                            exprs[m]['outer_ear_y'][w][start_index:end_index +
                                                       1],
                            "upper_whisker_x":
                            exprs[m]['upper_whisker_x'][w]
                            [start_index:end_index + 1],
                            "upper_whisker_y":
                            exprs[m]['upper_whisker_y'][w]
                            [start_index:end_index + 1],
                            "outer_whisker_x":
                            exprs[m]['outer_whisker_x'][w]
                            [start_index:end_index + 1],
                            "outer_whisker_y":
                            exprs[m]['outer_whisker_y'][w]
                            [start_index:end_index + 1],
                            "lower_whisker_x":
                            exprs[m]['lower_whisker_x'][w]
                            [start_index:end_index + 1],
                            "lower_whisker_y":
                            exprs[m]['lower_whisker_y'][w]
                            [start_index:end_index + 1],
                            "upper_mouth_x":
                            exprs[m]['upper_mouth_x'][w]
                            [start_index:end_index + 1],
                            "upper_mouth_y":
                            exprs[m]['upper_mouth_y'][w]
                            [start_index:end_index + 1],
                            "outer_mouth_x":
                            exprs[m]['outer_mouth_x'][w]
                            [start_index:end_index + 1],
                            "outer_mouth_y":
                            exprs[m]['outer_mouth_y'][w]
                            [start_index:end_index + 1],
                            "lower_mouth_x":
                            exprs[m]['lower_mouth_x'][w]
                            [start_index:end_index + 1],
                            "lower_mouth_y":
                            exprs[m]['lower_mouth_y'][w]
                            [start_index:end_index + 1],
                            "inner_nostril_x":
                            exprs[m]['inner_nostril_x'][w]
                            [start_index:end_index + 1],
                            "inner_nostril_y":
                            exprs[m]['inner_nostril_y'][w]
                            [start_index:end_index + 1],
                            "outer_nostril_x":
                            exprs[m]['outer_nostril_x'][w]
                            [start_index:end_index + 1],
                            "outer_nostril_y":
                            exprs[m]['outer_nostril_y'][w]
                            [start_index:end_index + 1]
                        },
                        columns=[
                            "mouse_list", "week_list", "frame_list",
                            "timestamps", "trial_num", "trial_idx",
                            "trial_type", "upper_eye_x", "upper_eye_y",
                            "lower_eye_x", "lower_eye_y", "upper_ear_x",
                            "upper_ear_y", "lower_ear_x", "lower_ear_y",
                            "outer_ear_x", "outer_ear_y", "upper_whisker_x",
                            "upper_whisker_y", "outer_whisker_x",
                            "outer_whisker_y", "lower_whisker_x",
                            "lower_whisker_y", "upper_mouth_x",
                            "upper_mouth_y", "outer_mouth_x", "outer_mouth_y",
                            "lower_mouth_x", "lower_mouth_y",
                            "inner_nostril_x", "inner_nostril_y",
                            "outer_nostril_x", "outer_nostril_y"
                        ],
                        index=[j for j in range(start_index, end_index + 1)])

                    led = led + 1

        print(f"\tWeek: {w} ending...")

        print((type(d) == type(pd.DataFrame())) for d in dataframe)

        dataframe = [
            d for i, d in enumerate(dataframe)
            if type(dataframe[i]) == type(pd.DataFrame())
        ]

        data[w] = pd.concat(dataframe, keys=[i for i in range(len(dataframe))])

        if week_filter[w]:
            print("\tWeek:", w, "Num Trials:", len(dataframe), "-- done")

        del (dataframe)

    data = [d for i, d in enumerate(data) if week_filter[i]]

    exprs[m]['trial_data_by_week'] = data
    del (data)

    print(mouse, "-- complete")

print("Done")

#%%
# create data frame of weekly data
columns = [
    "mouse_list", "week_list", "frame_list", "upper_eye_x", "upper_eye_y",
    "lower_eye_x", "lower_eye_y", "upper_ear_x", "upper_ear_y", "lower_ear_x",
    "lower_ear_y", "outer_ear_x", "outer_ear_y", "upper_whisker_x",
    "upper_whisker_y", "outer_whisker_x", "outer_whisker_y", "lower_whisker_x",
    "lower_whisker_y", "upper_mouth_x", "upper_mouth_y", "outer_mouth_x",
    "outer_mouth_y", "lower_mouth_x", "lower_mouth_y", "inner_nostril_x",
    "inner_nostril_y", "outer_nostril_x", "outer_nostril_y"
]

for m, mouse in enumerate(mice):
    n_weeks = len(exprs[m]["timestamps"])

    # create variable arrays with the length of weeks
    week_filter = np.zeros(n_weeks, dtype=bool)

    # create variable arrays with the length of weeks
    # create a list to store data from each week
    dataframe = [{} for _ in range(n_weeks)]
    data = [_ for _ in range(n_weeks)]

    # iterate weeks
    for w in range(n_weeks):

        # for each week, convert the "mouse" variables (for each frame) into a numpy array
        # Begin the dataframe
        dataframe[w][columns[0]] = np.array(exprs[m]["mouse_list"][w],
                                            dtype=str)

        # Convert the frame # and week # into numpy arrays
        for v in columns[1:3]:
            dataframe[w][v] = np.array(exprs[m][v][w], dtype=np.int64)

        # mean center the track points
        for v in columns[3:29]:
            if type(exprs[m][v][w]) == type(np.array([0, 1, 2, 3])):

                means = np.mean(exprs[m][v][w], axis=0)

                dataframe[w][v] = exprs[m][v][w] - means

                week_filter[w] = True

            else:
                print(
                    f"\terror mean centering mouse: {mouse}, dataset: {v}, week: {w}"
                )
        for v in columns[0:3]:
            dataframe[w][v] = dataframe[w][v][0:dataframe[w][columns[6]].
                                              shape[0]]

        # Concatenate all numpy arrays into 'week' pandas dataframe
        data[w] = pd.DataFrame(dataframe[w], columns=columns)

        print("\tWeek:", w, "Num columns:", len(dataframe[w].keys()),
              "-- done") if week_filter[w] else print(
                  f"\t\tDoes the data exist for mouse {mouse} on week {w}?")

    data = [d for i, d in enumerate(data) if week_filter[i]]

    exprs[m]['weekly_data'] = data
    del (data)
    del (dataframe)

    print(mouse, "-- complete")

print("Done")

#%%
# enumerate mice
for m, mouse in enumerate(mice):
    print(mouse, "-- starting")

    n_weeks = len(exprs[m]["weekly_data"])
    week_filter = np.zeros(n_weeks, dtype=bool)

    centered_data = [_ for _ in range(n_weeks)]

    # iterate weeks
    for w in range(n_weeks):

        # if the video exists and is processed
        if type(exprs[m]["weekly_data"][w]) == type(pd.DataFrame()):

            week_filter[w] = True

            raw_data = exprs[m]["weekly_data"][
                w].loc[:, "upper_eye_x":"outer_nostril_y"]
            targets = exprs[m]["weekly_data"][w].loc[:,
                                                     "mouse_list":"frame_list"]

            centered_data[w] = raw_data.subtract(raw_data.mean())

            centered_data[w] = pd.concat([targets, centered_data[w]], axis=1)

        print("\tWeek:", w,
              "mean centered! -- done") if week_filter[w] else print(
                  f"\t\tDoes the data exist for mouse {mouse} on week {w}?")

    centered_data = [d for i, d in enumerate(centered_data) if week_filter[i]]

    exprs[m]["weekly_data_centered"] = centered_data
    exprs[m]["data_by_mouse"] = pd.concat(
        centered_data, keys=[f"week_{w}" for w in range(len(centered_data))])

    del (centered_data)
    del (raw_data)
    del (targets)

    print(mouse, "-- complete")

print("Done")

#%%
for m, mouse in enumerate(mice):
    print(mouse, "-- starting")

    raw_data = exprs[m]["data_by_mouse"].loc[:,
                                             "upper_eye_x":"outer_nostril_y"]
    targets = exprs[m]["data_by_mouse"].loc[:, "mouse_list":"frame_list"]

    centered_data = raw_data.subtract(raw_data.mean())

    centered_data = pd.concat([targets, centered_data], axis=1)

    print("\tCentered!")

    exprs[m]["data_centered_by_mouse"] = centered_data

    del (centered_data)
    del (raw_data)
    del (targets)

    print(mouse, "-- complete")

print("Done")

print("Cleaning...")
del (end)
del (end_difference_array)
del (end_index)
del (i)
del (led)
del (led_end_array)
del (led_start_array)
del (m)
del (mouse)
del (n_weeks)
del (speaker)
del (start)
del (start_difference_array)
del (start_index)
del (trial)
del (trial_type)
del (v)
del (w)
del (week)
del (week_filter)
del (weeks)

#%%
# enumerate mice
for m, mouse in enumerate(mice):
    print(mouse, "-- starting")

    n_weeks = len(exprs[m]["trial_data_by_week"])
    week_filter = np.zeros(n_weeks, dtype=bool)

    trial_data_centered = [_ for _ in range(n_weeks)]
    trial_data = [_ for _ in range(n_weeks)]

    # iterate weeks
    for w in range(n_weeks):

        # if the video exists and is processed
        if type(exprs[m]["trial_data_by_week"][w]) == type(pd.DataFrame()):

            week_filter[w] = True

            raw_data = exprs[m]["trial_data_by_week"][
                w].loc[:, "upper_eye_x":"outer_nostril_y"]
            targets = exprs[m]["trial_data_by_week"][
                w].loc[:, "mouse_list":"trial_type"]

            trial_data_centered[w] = raw_data.subtract(raw_data.mean())

            trial_data_centered[w] = pd.concat(
                [targets, trial_data_centered[w]], axis=1)

        print("\tWeek:", w,
              "mean centered! -- done") if week_filter[w] else print(
                  f"\t\tDoes the data exist for mouse {mouse} on week {w}?")

    trial_data_centered = [
        d for i, d in enumerate(trial_data_centered) if week_filter[i]
    ]
    trial_data = [d for i, d in enumerate(trial_data) if week_filter[i]]

    exprs[m]["trial_data_centered_by_week"] = trial_data_centered
    exprs[m]["trial_data_by_mouse"] = pd.concat(
        trial_data_centered,
        keys=[f"week_{w}" for w in range(len(trial_data_centered))])
    exprs[m]["raw_trial_data_by_mouse"] = pd.concat(
        exprs[m]["trial_data_by_week"],
        keys=[f"week_{w}" for w in range(len(trial_data))])

    del (trial_data_centered)
    del (raw_data)
    del (targets)

    print(mouse, "-- complete")

print("Done")

#%%
for m, mouse in enumerate(mice):
    print(mouse, "-- starting")

    raw_data = exprs[m][
        "trial_data_by_mouse"].loc[:, "upper_eye_x":"outer_nostril_y"]
    targets = exprs[m]["trial_data_by_mouse"].loc[:, "mouse_list":"trial_type"]

    trial_data_centered = raw_data.subtract(raw_data.mean())

    trial_data_centered = pd.concat([targets, trial_data_centered], axis=1)

    print("\tCentered!")

    exprs[m]["trial_data_centered_by_mouse"] = trial_data_centered

    del (trial_data_centered)
    del (raw_data)
    del (targets)

    print(mouse, "-- complete")

print("Done")

#%%
for x in range(len(exprs)):

    targets_list = exprs[x][
        "data_centered_by_mouse"].loc[:, "mouse_list":"frame_list"]
    data_frame = exprs[x][
        "data_centered_by_mouse"].loc[:, "upper_eye_x":"outer_nostril_y"]

    pca = PCA(n_components=2)
    data_frame_pca = pca.fit_transform(data_frame)
    principalDf = pd.DataFrame(
        data_frame_pca,
        columns=["principal component 1", "principal component 2"])
    finalDf = pd.concat([principalDf, targets_list], axis=1)
    fig = px.scatter(data_frame=finalDf,
                     x="principal component 1",
                     y="principal component 2",
                     color='week_list',
                     title=f"{mice[x]}: color=week")
    fig_layout = go.Layout(height=800, width=800)
    fig_widget = go.FigureWidget(data=fig, layout=fig_layout)
    fig_widget.write_html(
        f'/Users/annieehler/Projects/Jupyter_Notebooks/trials/{mice[x]}-color-week-2d.html',
        auto_open=False)
print("Done")

#%%
pca_3d = [{} for _ in range(len(mice))]

for x in range(len(exprs)):

    targets_list = exprs[x][
        "data_centered_by_mouse"].loc[:, "mouse_list":"frame_list"]
    data_frame = exprs[x][
        "data_centered_by_mouse"].loc[:, "upper_eye_x":"outer_nostril_y"]

    pca = PCA(n_components=3)
    data_frame_pca = pca.fit_transform(data_frame)
    principalDf = pd.DataFrame(data_frame_pca,
                               columns=[
                                   "principal component 1",
                                   "principal component 2",
                                   "principal component 3"
                               ])
    finalDf = pd.concat([targets_list.reset_index(), principalDf], axis=1)
    fig = px.scatter_3d(data_frame=finalDf,
                        x="principal component 1",
                        y="principal component 2",
                        z="principal component 3",
                        color='week_list',
                        title=f"{mice[x]}: color=week")
    fig_layout = go.Layout(height=800, width=800)
    fig_widget = go.FigureWidget(data=fig, layout=fig_layout)
    fig_widget.write_html(
        f'/Users/annieehler/Projects/Jupyter_Notebooks/trials/{mice[x]}-color-week-3d.html',
        auto_open=False)

    pca_3d[x]["data_centered_by_mouse"] = finalDf

print("Done")

#%%
dataframes = [_ for _ in range(len(mice))]
centered_dataframes = [_ for _ in range(len(mice))]

for m in range(len(mice)):
    dataframes[m] = exprs[m]["raw_trial_data_by_mouse"]
    centered_dataframes[m] = exprs[m]["trial_data_centered_by_mouse"]

    targets_list = exprs[m][
        "raw_trial_data_by_mouse"].loc[:, "mouse_list":"trial_type"]
    centered_targets_list = exprs[m][
        "trial_data_centered_by_mouse"].loc[:, "mouse_list":"trial_type"]

    data_frame = exprs[x][
        "raw_trial_data_by_mouse"].loc[:, "upper_eye_x":"outer_nostril_y"]
    centered_data_frame = exprs[x][
        "trial_data_centered_by_mouse"].loc[:, "upper_eye_x":"outer_nostril_y"]

    pca = PCA(n_components=3)
    data_frame_pca = pca.fit_transform(data_frame)
    centered_data_frame_pca = pca.fit_transform(centered_data_frame)

    principalDf = pd.DataFrame(data_frame_pca,
                               columns=[
                                   "principal component 1",
                                   "principal component 2",
                                   "principal component 3"
                               ])
    centered_principalDf = pd.DataFrame(centered_data_frame_pca,
                                        columns=[
                                            "principal component 1",
                                            "principal component 2",
                                            "principal component 3"
                                        ])

    finalDf = pd.concat([targets_list.reset_index(), principalDf], axis=1)
    centered_finalDf = pd.concat(
        [centered_targets_list.reset_index(), centered_principalDf], axis=1)

    finalDf.loc[:, ["mouse_list", "week_list"]] = finalDf.astype({
        'mouse_list':
        str,
        'week_list':
        str
    })
    centered_finalDf.loc[:, ["mouse_list", "week_list"
                             ]] = centered_finalDf.astype({
                                 'mouse_list': str,
                                 'week_list': str
                             })

    fig = px.scatter_3d(data_frame=finalDf,
                        x="principal component 1",
                        y="principal component 2",
                        z="principal component 3",
                        color='trial_type',
                        title=f"{mice[x]}: color=trial_type")
    fig.update_shapes(xsizemode="pixel", ysizemode="pixel")
    fig.update_traces(marker={'size': 2})
    fig_layout = go.Layout(height=800, width=800)
    fig_widget = go.FigureWidget(data=fig, layout=fig_layout)
    fig_widget.write_html(
        f'/Users/annieehler/Projects/Jupyter_Notebooks/trials/{mice[m]}-by-trial-3d.html',
        auto_open=False)

    fig = px.scatter_3d(data_frame=centered_finalDf,
                        x="principal component 1",
                        y="principal component 2",
                        z="principal component 3",
                        color='trial_type',
                        title=f"{mice[x]}: mean centered by mouse")
    fig.update_shapes(xsizemode="pixel", ysizemode="pixel")
    fig.update_traces(marker={'size': 2})
    fig_layout = go.Layout(height=800, width=800)
    fig_widget = go.FigureWidget(data=fig, layout=fig_layout)
    fig_widget.write_html(
        f'/Users/annieehler/Projects/Jupyter_Notebooks/trials/{mice[m]}-centered-by-trial-3d.html',
        auto_open=False)

#%%
data = pd.concat(dataframes, keys=[mouse for mouse in mice])
centered_data = pd.concat(centered_dataframes, keys=[mouse for mouse in mice])

non_numeric_cols = data.loc[:, "mouse_list":"trial_type"].columns
non_numeric_cols = data[non_numeric_cols]
centered_non_numeric_cols = centered_data.loc[:, "mouse_list":
                                              "trial_type"].columns
centered_non_numeric_cols = centered_data[centered_non_numeric_cols]

numeric_cols = data.loc[:, "upper_eye_x":"outer_nostril_y"].columns
numeric_cols = data[numeric_cols].apply(zscore)

centered_numeric_cols = centered_data.loc[:, "upper_eye_x":
                                          "outer_nostril_y"].columns
centered_numeric_cols = centered_data[centered_numeric_cols].apply(zscore)

pca = PCA(n_components=3)
data_frame_pca = pca.fit_transform(numeric_cols)
centered_data_frame_pca = pca.fit_transform(centered_numeric_cols)

principalDf = pd.DataFrame(data_frame_pca,
                           columns=[
                               "principal component 1",
                               "principal component 2", "principal component 3"
                           ])
centered_principalDf = pd.DataFrame(centered_data_frame_pca,
                                    columns=[
                                        "principal component 1",
                                        "principal component 2",
                                        "principal component 3"
                                    ])

std_x = statistics.stdev(principalDf.std(0).to_list())
std_y = statistics.stdev(principalDf.std(1).to_list())
cen_std_x = statistics.stdev(centered_principalDf.std(0).to_list())
cen_std_y = statistics.stdev(centered_principalDf.std(1).to_list())

principalDf_blurred = cv2.GaussianBlur(principalDf.to_numpy(), (3, 11),
                                       sigmaX=std_x,
                                       sigmaY=std_y)
principalDf_blurred = pd.DataFrame(principalDf_blurred,
                                   columns=[
                                       "principal component 1",
                                       "principal component 2",
                                       "principal component 3"
                                   ])

centered_principalDf_blurred = cv2.GaussianBlur(
    centered_principalDf.to_numpy(), (3, 11),
    sigmaX=cen_std_x,
    sigmaY=cen_std_y)
centered_principalDf_blurred = pd.DataFrame(centered_principalDf_blurred,
                                            columns=[
                                                "principal component 1",
                                                "principal component 2",
                                                "principal component 3"
                                            ])

principalDf = pd.concat([non_numeric_cols.reset_index(), principalDf], axis=1)
principalDf_blurred = pd.concat(
    [non_numeric_cols.reset_index(), principalDf_blurred], axis=1)
principalDf_blurred.loc[:, ["mouse_list", "week_list"
                            ]] = principalDf_blurred.astype({
                                'mouse_list': str,
                                'week_list': str
                            })

centered_principalDf = pd.concat(
    [centered_non_numeric_cols.reset_index(), centered_principalDf], axis=1)
centered_principalDf_blurred = pd.concat(
    [centered_non_numeric_cols.reset_index(), centered_principalDf_blurred],
    axis=1)
centered_principalDf_blurred.loc[:, ["mouse_list", "week_list"
                                     ]] = centered_principalDf_blurred.astype({
                                         'mouse_list':
                                         str,
                                         'week_list':
                                         str
                                     })

#%%
trial_type = [
    "Airpuff", "Sucrose", "Airpuff catch", "Sucrose catch", "Airpuff with LED",
    "Sucrose with LED", "LED Only"
]

for i in range(0, 7):
    fig_data = principalDf_blurred['trial_type'].values == trial_type[i]
    fig_data = principalDf_blurred[fig_data]
    fig_data.loc[:, ["mouse_list", "week_list"]] = fig_data.astype({
        'mouse_list':
        str,
        'week_list':
        str
    })
    fig = px.line_3d(data_frame=fig_data,
                     x="principal component 1",
                     y="principal component 2",
                     z="principal component 3",
                     color='week_list',
                     symbol="mouse_list",
                     title=f'{trial_type[i]} Smoothed: color=week shape=mouse')
    fig.update_shapes(xsizemode="pixel", ysizemode="pixel")
    fig.update_traces(marker={'size': 2})
    fig_layout = go.Layout(height=800, width=800)
    fig_widget = go.FigureWidget(data=fig, layout=fig_layout)
    fig_widget.write_html(
        f'/Users/annieehler/Projects/Jupyter_Notebooks/trials/{trial_type[i].replace(" ", "_")}-smoothed-3d-line.html',
        auto_open=False)

    fig_data = centered_principalDf_blurred['trial_type'].values == trial_type[
        i]
    fig_data = centered_principalDf_blurred[fig_data]
    fig_data.loc[:, ["mouse_list", "week_list"]] = fig_data.astype({
        'mouse_list':
        str,
        'week_list':
        str
    })
    fig = px.line_3d(
        data_frame=fig_data,
        x="principal component 1",
        y="principal component 2",
        z="principal component 3",
        color='week_list',
        symbol="mouse_list",
        title=f'{trial_type[i]} Smoothed & Centered: color=week shape=mouse')
    fig.update_shapes(xsizemode="pixel", ysizemode="pixel")
    fig.update_traces(marker={'size': 2})
    fig_layout = go.Layout(height=800, width=800)
    fig_widget = go.FigureWidget(data=fig, layout=fig_layout)
    fig_widget.write_html(
        f'/Users/annieehler/Projects/Jupyter_Notebooks/trials/{trial_type[i].replace(" ", "_")}-smoothed-centered-3d-line.html',
        auto_open=False)

#%%
fig = px.scatter_3d(data_frame=principalDf_blurred,
                    x="principal component 1",
                    y="principal component 2",
                    z="principal component 3",
                    color='trial_type',
                    title="All trials projections")
fig.update_shapes(xsizemode="pixel", ysizemode="pixel")
fig.update_traces(marker={'size': 2})
fig_layout = go.Layout(height=800, width=800)
fig_widget = go.FigureWidget(data=fig, layout=fig_layout)
fig_widget.write_html(
    f'/Users/annieehler/Projects/Jupyter_Notebooks/trials/All-trials-3d.html',
    auto_open=False)

fig = px.scatter_3d(data_frame=principalDf_blurred,
                    x="principal component 1",
                    y="principal component 2",
                    z="principal component 3",
                    color='mouse_list',
                    title="All trials projections")
fig.update_shapes(xsizemode="pixel", ysizemode="pixel")
fig.update_traces(marker={'size': 2})
fig_layout = go.Layout(height=800, width=800)
fig_widget = go.FigureWidget(data=fig, layout=fig_layout)
fig_widget.write_html(
    f'/Users/annieehler/Projects/Jupyter_Notebooks/trials/All-trials-color-mouse-3d.html',
    auto_open=False)

fig = px.scatter_3d(data_frame=principalDf_blurred,
                    x="principal component 1",
                    y="principal component 2",
                    z="principal component 3",
                    color='week_list',
                    title="All trials projections")
fig.update_shapes(xsizemode="pixel", ysizemode="pixel")
fig.update_traces(marker={'size': 2})
fig_layout = go.Layout(height=800, width=800)
fig_widget = go.FigureWidget(data=fig, layout=fig_layout)
fig_widget.write_html(
    f'/Users/annieehler/Projects/Jupyter_Notebooks/trials/All-trials-color-week-3d.html',
    auto_open=False)

fig = px.scatter_3d(data_frame=centered_principalDf_blurred,
                    x="principal component 1",
                    y="principal component 2",
                    z="principal component 3",
                    color='trial_type',
                    title="All trials projections - centered")
fig.update_shapes(xsizemode="pixel", ysizemode="pixel")
fig.update_traces(marker={'size': 2})
fig_layout = go.Layout(height=800, width=800)
fig_widget = go.FigureWidget(data=fig, layout=fig_layout)
fig_widget.write_html(
    f'/Users/annieehler/Projects/Jupyter_Notebooks/trials/All-trials-centered-3d.html',
    auto_open=False)

fig = px.scatter_3d(data_frame=centered_principalDf_blurred,
                    x="principal component 1",
                    y="principal component 2",
                    z="principal component 3",
                    color='week_list',
                    title="All trials projections - centered")
fig.update_shapes(xsizemode="pixel", ysizemode="pixel")
fig.update_traces(marker={'size': 2})
fig_layout = go.Layout(height=800, width=800)
fig_widget = go.FigureWidget(data=fig, layout=fig_layout)
fig_widget.write_html(
    f'/Users/annieehler/Projects/Jupyter_Notebooks/trials/All-trials-color-week-centered-3d.html',
    auto_open=False)

fig = px.scatter_3d(data_frame=centered_principalDf_blurred,
                    x="principal component 1",
                    y="principal component 2",
                    z="principal component 3",
                    color='mouse_list',
                    title="All trials projections - centered")
fig.update_shapes(xsizemode="pixel", ysizemode="pixel")
fig.update_traces(marker={'size': 2})
fig_layout = go.Layout(height=800, width=800)
fig_widget = go.FigureWidget(data=fig, layout=fig_layout)
fig_widget.write_html(
    f'/Users/annieehler/Projects/Jupyter_Notebooks/trials/All-trials-color-mouse-centered-3d.html',
    auto_open=False)

# %%
# export data
hf = h5py.File(
    "/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/metadata.h5",
    "w")

for m, mouse in enumerate(mice):
    print(f"{mouse} -- saving metadata")

    for key in list(exprs[m].keys()):
        if type(exprs[m][key]) is list or type(exprs[m][key]) is np.array:
            for w in range(len(exprs[m][key])):

                if type(exprs[m][key][w]) is list or type(
                        exprs[m][key][w]) is np.array:
                    hf.create_dataset(f"{mouse}/{w}/{key}",
                                      data=exprs[m][key][w])

                if type(exprs[m][key][w]) is dict:
                    g = hf.create_group(f"{mouse}/{w}/{key}")
                    hdfdict.dump(data=exprs[m][key][w], hdf=g)

hf.close()

hf = pd.HDFStore(
    "/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/datatables.h5",
    "a")
for m, mouse in enumerate(mice):
    print(f"{mouse} -- saving datatables")

    for key in list(exprs[m].keys()):
        if type(exprs[m][key]) is pd.DataFrame:
            hf.put(f"{mouse}/{key}", exprs[m][key])

hf.close()
hf = pd.HDFStore(
    "/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/data.h5", "a")

print(f"Saving concatenated data")
for key in list(all_data_and_pcas.keys()):
    hf.put(f"{key}", all_data_and_pcas[key])

hf.close()
