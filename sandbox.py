import pandas as pd

mouse = ["CSE021" for _ in range(len([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]
week = [2 for _ in range(len([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]
datalen = [i for i in range(len([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]
timestamp = [(i * 33) for i in range(len([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]

meta_index = pd.MultiIndex.from_arrays(
    [mouse, week, datalen, timestamp],
    names=["mouse", "week", "index", "timepoint"])

print(meta_index)
