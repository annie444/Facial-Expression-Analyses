# %%

from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
from scipy import integrate
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# %%

with pd.HDFStore(
        "/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/data.h5"
) as data_file:
    data = data_file.get('all_trial_data')

# %%

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

# %%

cse_array = ['CSE008', 'CSE020']
csc_array = ['CSC009', 'CSC013']

cse = new_data.mouse_list.isin(cse_array)
csc = new_data.mouse_list.isin(csc_array)

data_cse = new_data[cse]
data_csc = new_data[csc]

# %%

trail_start = round((33.990482664853836**-1) * 0)

cs_start = round((33.990482664853836**-1) * 10000)

us = round((33.990482664853836**-1) * 12800)

cs_end = round((33.990482664853836**-1) * 13000)

trial_end = round((33.990482664853836**-1) * 20000)

print(trail_start, cs_start, us, cs_end, trial_end)

# %%

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

# %%

dfs = [data_csc, data_cse]

for d, df in enumerate(dfs):

    speeds = []
    accelerations = []

    diff = df[[
        "timestamps", "eye_x", "eye_y", "ear_x", "ear_y", "whisker_x",
        "whisker_y", "mouth_x", "mouth_y", "nostril_x", "nostril_y"
    ]].diff()

    for point in ["eye", "ear", "whisker", "mouth", "nostril"]:

        speed = np.sqrt((diff[f"{point}_x"]**2) + (diff[f"{point}_y"]**2))
        acceleration = speed / diff["timestamps"]

        speed.name = f"{point}_speed"
        acceleration.name = f"{point}_acceleration"

        speeds.append(speed)
        accelerations.append(acceleration)

    speeds = pd.concat(speeds, axis=1)
    accelerations = pd.concat(accelerations, axis=1)

    if d == 0:
        data_csc = data_csc.join([speeds, accelerations])
    elif d == 1:
        data_cse = data_cse.join([speeds, accelerations])

#%%


def upper_sem(x):
    return pd.Series((np.mean(x) + x.sem()), name="upper_sem")


def lower_sem(x):
    return pd.Series((np.mean(x) - x.sem()), name="lower_sem")


#%%

dfs = [data_csc, data_cse]

for d, df in enumerate(dfs):

    mean_and_sem = []

    for point in ["eye", "ear", "whisker", "mouth", "nostril"]:
        mean_and_sem.append(
            df.groupby([
                "mouse_list", "week_list", "trial_type", "frame_trial_idx"
            ]).agg({
                f"{point}_speed": ['mean', 'sem', upper_sem, lower_sem],
                f"{point}_acceleration": ['mean', 'sem', upper_sem, lower_sem]
            }))

    if d == 0:
        mean_and_sem_df_csc = pd.concat(
            mean_and_sem, axis=1).sort_index(level="frame_trial_idx")
    elif d == 1:
        mean_and_sem_df_cse = pd.concat(
            mean_and_sem, axis=1).sort_index(level="frame_trial_idx")

# %%
today = date.today().strftime("%Y%m%d")
points = ["eye", "ear", "whisker", "mouth", "nostril"]
colors = [
    'rgb(213,94,0)', 'rgb(204,121,167)', 'rgb(0,114,178)', 'rgb(240,228,66)',
    'rgb(0,158,115)', 'rgb(0,0,0)', 'rgb(220,20,60)'
]
fillcolors = [
    'rgba(213,94,0,0.2)', 'rgba(204,121,167,0.2)', 'rgba(0,114,178,0.2)',
    'rgba(240,228,66,0.2)', 'rgba(0,158,115,0.2)', 'rgba(0,0,0,0.2)',
    'rgba(220,20,60,0.2)'
]

# %%

ttests = pd.DataFrame(columns=["test statistic", "p-value"])

for week in weeks:
    for trial in trials:
        for point in points:

            fig = go.Figure()

            fig.add_shape(
                type="rect",
                x0=cs_start,
                y0=0,
                x1=cs_end,
                y1=1,
                xref='x',
                yref='paper',
                line=dict(color="rgba(0,0,0,0)"),
                fillcolor="rgba(243,219,158,0.2)",
            )

            fig.add_traces([
                go.Scatter(x=mean_and_sem_df_cse.xs(
                    (week, trial),
                    level=("week_list", "trial_type")).index.get_level_values(
                        'frame_trial_idx').to_list()[1:],
                           y=mean_and_sem_df_cse.xs(
                               (week, trial),
                               level=("week_list",
                                      "trial_type"))[(f"{point}_speed",
                                                      "mean")].iloc[1:],
                           line=dict(color=colors[6]),
                           mode='lines',
                           name=f'CSE {point}'),
                go.Scatter(x=mean_and_sem_df_csc.xs(
                    (week, trial),
                    level=("week_list", "trial_type")).index.get_level_values(
                        'frame_trial_idx').to_list()[1:],
                           y=mean_and_sem_df_csc.xs(
                               (week, trial),
                               level=("week_list",
                                      "trial_type"))[(f"{point}_speed",
                                                      "mean")].iloc[1:],
                           line=dict(color=colors[5]),
                           mode='lines',
                           name=f'Control {point}'),
            ])

            fig.add_traces([
                go.Scatter(x=mean_and_sem_df_cse.xs(
                    (week, trial),
                    level=("week_list", "trial_type")).index.get_level_values(
                        'frame_trial_idx').to_list()[1:],
                           y=mean_and_sem_df_cse.xs(
                               (week, trial),
                               level=("week_list",
                                      "trial_type"))[(f"{point}_speed",
                                                      "upper_sem")].iloc[1:],
                           mode='none',
                           hoverinfo="skip",
                           showlegend=False,
                           fill='tonexty',
                           fillcolor=fillcolors[6]),
                go.Scatter(x=mean_and_sem_df_cse.xs(
                    (week, trial),
                    level=("week_list", "trial_type")).index.get_level_values(
                        'frame_trial_idx').to_list()[1:],
                           y=mean_and_sem_df_cse.xs(
                               (week, trial),
                               level=("week_list",
                                      "trial_type"))[(f"{point}_speed",
                                                      "lower_sem")].iloc[1:],
                           mode='none',
                           hoverinfo="skip",
                           name=f'CSE {point} SEM',
                           fill='tonexty',
                           fillcolor=fillcolors[6]),
                go.Scatter(x=mean_and_sem_df_csc.xs(
                    (week, trial),
                    level=("week_list", "trial_type")).index.get_level_values(
                        'frame_trial_idx').to_list()[1:],
                           y=mean_and_sem_df_csc.xs(
                               (week, trial),
                               level=("week_list",
                                      "trial_type"))[(f"{point}_speed",
                                                      "upper_sem")].iloc[1:],
                           mode='none',
                           hoverinfo="skip",
                           showlegend=False,
                           fill='tonexty',
                           fillcolor=fillcolors[5]),
                go.Scatter(x=mean_and_sem_df_csc.xs(
                    (week, trial),
                    level=("week_list", "trial_type")).index.get_level_values(
                        'frame_trial_idx').to_list()[1:],
                           y=mean_and_sem_df_csc.xs(
                               (week, trial),
                               level=("week_list",
                                      "trial_type"))[(f"{point}_speed",
                                                      "lower_sem")].iloc[1:],
                           mode='none',
                           hoverinfo="skip",
                           name=f'Control {point} SEM',
                           fill='tonexty',
                           fillcolor=fillcolors[5])
            ])

            fig.update_layout(title=(
                f"Speed of {point} for Trial Type {trial} on Week {week}"),
                              xaxis=dict(title="sec",
                                         tickvals=[0, cs_start, cs_end, 677],
                                         ticktext=[-10, 0, 3, 13]),
                              yaxis=dict(title="Facial Movement (px/ms)",
                                         type="linear",
                                         ticks="outside",
                                         tick0=0,
                                         tickmode='array',
                                         tickvals=[0, 20, 40],
                                         ticktext=[0, 20, 40],
                                         range=[0, 45]),
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              font_family='arial',
                              height=450,
                              width=550)

            fig.add_vline(x=cs_start,
                          line_width=1,
                          line_dash="dot",
                          annotation_text="CS start",
                          annotation_textangle=90,
                          annotation_position='top right')

            fig.add_vline(x=us,
                          line_width=1,
                          line_dash="dash",
                          annotation_text="US delivery",
                          annotation_textangle=90,
                          annotation_position='top left')

            fig.add_vline(x=cs_end,
                          line_width=1,
                          line_dash="dot",
                          annotation_text="CS end",
                          annotation_textangle=90,
                          annotation_position='top right')

            pval = stats.ttest_ind(
                data_cse[(data_cse.week_list == week)
                         & (data_cse.trial_type == trial)]
                [f"{point}_speed"].values, data_csc[
                    (data_csc.week_list == week)
                    & (data_csc.trial_type == trial)][f"{point}_speed"].values)

            ttests = pd.concat([
                ttests,
                pd.DataFrame(
                    [pval.statistic, pval.pvalue],
                    columns=[
                        f"{point}_speed ~ C(mouse_type):C(week).T[{week}]:C(trial_type).T[{trial}]"
                    ],
                    index=["test statistic", "p-value"]).T
            ])

            if pval.pvalue <= 0.05:
                fig.add_shape(
                    type="line",
                    xref="paper",
                    yref="paper",
                    x0=1.02,
                    y0=0.25,
                    x1=1.02,
                    y1=0.75,
                    line=dict(
                        color="Black",
                        width=1,
                    ),
                )
                fig.add_shape(
                    type="line",
                    xref="paper",
                    yref="paper",
                    x0=1.02,
                    y0=0.25,
                    x1=1.01,
                    y1=0.25,
                    line=dict(
                        color="Black",
                        width=1,
                    ),
                )
                fig.add_shape(
                    type="line",
                    xref="paper",
                    yref="paper",
                    x0=1.02,
                    y0=0.75,
                    x1=1.01,
                    y1=0.75,
                    line=dict(
                        color="Black",
                        width=1,
                    ),
                )

                if pval.pvalue >= 0.01:
                    fig.add_annotation(text="*",
                                       xref="paper",
                                       yref="paper",
                                       align="right",
                                       x=1.1,
                                       y=0.5,
                                       showarrow=False)
                elif pval.pvalue >= 0.001:
                    fig.add_annotation(text="**",
                                       xref="paper",
                                       yref="paper",
                                       align="right",
                                       x=1.1,
                                       y=0.5,
                                       showarrow=False)
                elif pval.pvalue >= 0.001:
                    fig.add_annotation(text="***",
                                       xref="paper",
                                       yref="paper",
                                       align="right",
                                       x=1.1,
                                       y=0.5,
                                       showarrow=False)
                else:
                    fig.add_annotation(text="****",
                                       xref="paper",
                                       yref="paper",
                                       align="right",
                                       x=1.1,
                                       y=0.5,
                                       showarrow=False)

            fig.update_annotations(font_family="arial")

            fig.write_image(
                f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_mouse_by_mouse_average_of_speed_cs-{trial}_week-{week}-{point}.pdf"
            )

            fig.write_image(
                f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_mouse_by_mouse_average_of_speed_cs-{trial}_week-{week}-{point}.svg"
            )

            fig.write_html(
                f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_mouse_by_mouse_average_of_speed_cs-{trial}_week-{week}-{point}.html"
            )

fig.show()

ttests.to_csv(
    f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_cs-{trial}_week-{week}-{point}-t-tests.csv"
)

#%%

for week in weeks:
    for trial in trials:

        fig_csc = go.Figure()
        fig_cse = go.Figure()

        fig_cse.add_shape(
            type="rect",
            x0=cs_start,
            y0=0,
            x1=cs_end,
            y1=1,
            xref='x',
            yref='paper',
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(243,219,158,0.2)",
        )

        fig_csc.add_shape(
            type="rect",
            x0=cs_start,
            y0=0,
            x1=cs_end,
            y1=1,
            xref='x',
            yref='paper',
            line=dict(color="rgba(0,0,0,0)"),
            fillcolor="rgba(243,219,158,0.2)",
        )

        for i, point in enumerate(points):
            fig_csc.add_trace(
                go.Scatter(x=mean_and_sem_df_csc.xs(
                    (week, trial),
                    level=("week_list", "trial_type")).index.get_level_values(
                        'frame_trial_idx').to_list()[1:],
                           y=mean_and_sem_df_csc.xs(
                               (week, trial),
                               level=("week_list",
                                      "trial_type"))[(f"{point}_speed",
                                                      "mean")].iloc[1:],
                           line=dict(color=colors[i]),
                           mode='lines',
                           name=f'CSC {point}'))
            fig_cse.add_trace(
                go.Scatter(x=mean_and_sem_df_cse.xs(
                    (week, trial),
                    level=("week_list", "trial_type")).index.get_level_values(
                        'frame_trial_idx').to_list()[1:],
                           y=mean_and_sem_df_cse.xs(
                               (week, trial),
                               level=("week_list",
                                      "trial_type"))[(f"{point}_speed",
                                                      "mean")].iloc[1:],
                           line=dict(color=colors[i]),
                           mode='lines',
                           name=f'CSE {point}'))

            fig_csc.add_traces([
                go.Scatter(x=mean_and_sem_df_csc.xs(
                    (week, trial),
                    level=("week_list", "trial_type")).index.get_level_values(
                        'frame_trial_idx').to_list()[1:],
                           y=mean_and_sem_df_csc.xs(
                               (week, trial),
                               level=("week_list",
                                      "trial_type"))[(f"{point}_speed",
                                                      "upper_sem")].iloc[1:],
                           mode='none',
                           hoverinfo="skip",
                           showlegend=False,
                           fill='tonexty',
                           fillcolor=fillcolors[i]),
                go.Scatter(x=mean_and_sem_df_csc.xs(
                    (week, trial),
                    level=("week_list", "trial_type")).index.get_level_values(
                        'frame_trial_idx').to_list()[1:],
                           y=mean_and_sem_df_csc.xs(
                               (week, trial),
                               level=("week_list",
                                      "trial_type"))[(f"{point}_speed",
                                                      "lower_sem")].iloc[1:],
                           mode='none',
                           hoverinfo="skip",
                           name=f'CSC {point} SEM',
                           fill='tonexty',
                           fillcolor=fillcolors[i])
            ])
            fig_cse.add_traces([
                go.Scatter(x=mean_and_sem_df_cse.xs(
                    (week, trial),
                    level=("week_list", "trial_type")).index.get_level_values(
                        'frame_trial_idx').to_list()[1:],
                           y=mean_and_sem_df_cse.xs(
                               (week, trial),
                               level=("week_list",
                                      "trial_type"))[(f"{point}_speed",
                                                      "upper_sem")].iloc[1:],
                           mode='none',
                           hoverinfo="skip",
                           showlegend=False,
                           fill='tonexty',
                           fillcolor=fillcolors[i]),
                go.Scatter(x=mean_and_sem_df_cse.xs(
                    (week, trial),
                    level=("week_list", "trial_type")).index.get_level_values(
                        'frame_trial_idx').to_list()[1:],
                           y=mean_and_sem_df_cse.xs(
                               (week, trial),
                               level=("week_list",
                                      "trial_type"))[(f"{point}_speed",
                                                      "lower_sem")].iloc[1:],
                           mode='none',
                           hoverinfo="skip",
                           name=f'CSE {point} SEM',
                           fill='tonexty',
                           fillcolor=fillcolors[i])
            ])

        for fig in [fig_cse, fig_csc]:

            fig.update_layout(title=(
                f"Average facial speed for Trial Type {trial} on Week {week}"),
                              xaxis=dict(title="sec",
                                         tickvals=[0, cs_start, cs_end, 677],
                                         ticktext=[-10, 0, 3, 13]),
                              yaxis=dict(title="Facial Movement (px/ms)",
                                         type="linear",
                                         ticks="outside",
                                         tick0=0,
                                         tickmode='array',
                                         tickvals=[0, 20, 40],
                                         ticktext=[0, 20, 40],
                                         range=[0, 45]),
                              paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              font_family='arial',
                              height=450,
                              width=550)

            fig.add_vline(x=cs_start,
                          line_width=1,
                          line_dash="dot",
                          annotation_text="CS start",
                          annotation_textangle=90,
                          annotation_position='top right')

            fig.add_vline(x=us,
                          line_width=1,
                          line_dash="dash",
                          annotation_text="US delivery",
                          annotation_textangle=90,
                          annotation_position='top left')

            fig.add_vline(x=cs_end,
                          line_width=1,
                          line_dash="dot",
                          annotation_text="CS end",
                          annotation_textangle=90,
                          annotation_position='top right')

            fig.update_annotations(font_family="arial")

        fig_csc.write_image(
            f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_CSC_mouse_by_mouse_average_of_speed_cs-{trial}_week-{week}.pdf"
        )

        fig_csc.write_image(
            f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_CSC_mouse_by_mouse_average_of_speed_cs-{trial}_week-{week}.svg"
        )

        fig_csc.write_html(
            f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_CSC_mouse_by_mouse_average_of_speed_cs-{trial}_week-{week}.html"
        )

        fig_cse.write_image(
            f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_CSE_mouse_by_mouse_average_of_speed_cs-{trial}_week-{week}.pdf"
        )

        fig_cse.write_image(
            f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_CSE_mouse_by_mouse_average_of_speed_cs-{trial}_week-{week}.svg"
        )

        fig_cse.write_html(
            f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_CSE_mouse_by_mouse_average_of_speed_cs-{trial}_week-{week}.html"
        )

fig_csc.show()
fig_cse.show()

# %%

integrals_csc = pd.DataFrame(index=pd.MultiIndex.from_product(
    [['CSC009', 'CSC013'], ['Sucrose', 'Airpuff'],
     ['eye', 'ear', 'whisker', 'mouth', 'nostril']],
    names=["mouse", "trial_type", "point"]),
                             columns=['0', '1', '5', '6'])
integrals_cse = pd.DataFrame(index=pd.MultiIndex.from_product(
    [['CSE008', 'CSE020'], ['Sucrose', 'Airpuff'],
     ['eye', 'ear', 'whisker', 'mouth', 'nostril']],
    names=["mouse", "trial_type", "point"]),
                             columns=['0', '1', '5', '6'])
integrals = [integrals_csc, integrals_cse]

for i, cs in enumerate([data_csc, data_cse]):
    for mouse in cs.mouse_list.unique():
        for week in cs.week_list.unique():
            for trial in cs.trial_type.unique():
                for point in points:
                    integral = integrate.trapz(
                        cs.loc[(cs_start < cs["frame_trial_idx"]) &
                               (cs["frame_trial_idx"] <
                                (cs_end + 90)) & (cs["mouse_list"] == mouse) &
                               (cs["week_list"] == week) &
                               (cs["trial_type"] == trial),
                               [f"{point}_speed"]].values.flatten())
                    integrals[i].loc[(f"{mouse}", f'{trial}', f'{point}'),
                                     f'{week}'] = integral

integrals = pd.concat(integrals, keys=["CSC", "CSE"], names=["mouse_type"])

integrals.columns = ["baseline", "early stress", "late stress", "ketamine"]

# %%

df_melt = pd.melt(
    integrals.reset_index(),
    id_vars=["mouse_type", "mouse", "trial_type", "point"],
    value_vars=["baseline", "early stress", "late stress", "ketamine"])

df_melt.columns = [
    "mouse_type", "mouse", "trial_type", "point", 'week', 'distance'
]
df_melt = df_melt.astype({
    "mouse_type": str,
    "mouse": str,
    "trial_type": str,
    "point": str,
    'week': str,
    'distance': 'float64'
})

table8 = ols('distance ~ ' + 'C(mouse_type)+' + 'C(trial_type)+' +
             'C(point)+' + 'C(week)+' + 'C(mouse_type):C(week)+' +
             'C(mouse_type):C(trial_type)+' +
             'C(mouse_type):C(trial_type):C(week)',
             data=df_melt).fit()
table9 = anova_lm(table8)

table9

# %%

sig_comps = table9.loc[(table9["PR(>F)"] <= 0.05)].index.tolist()
sig_comps = [
    index.replace('C', '').replace('(', '').replace(')',
                                                    '').replace(':', ', ')
    for index in sig_comps
]

#%%

emendation = ["", "-Aripuff_only", "-Sucrose_only"]

fig0 = go.Figure()
fig1 = go.Figure()
fig2 = go.Figure()
for f, fig in enumerate([fig0, fig1, fig2]):
    for i in range(len(integrals)):
        if integrals.loc[:, 'baseline':'ketamine'].iloc[
                i, :].name[2] == "Airpuff" and f in [0, 1]:
            fig.add_trace(
                go.Scatter(
                    x=integrals.columns.tolist(),
                    y=integrals.loc[:, 'baseline':'ketamine'].iloc[
                        i, :].values.flatten(),
                    meta=[
                        integrals.loc[:, 'baseline':'ketamine'].iloc[
                            i, :].name[1],
                        integrals.loc[:, 'baseline':'ketamine'].iloc[
                            i, :].name[2],
                        integrals.loc[:,
                                      'baseline':'ketamine'].iloc[i, :].name[3]
                    ],
                    mode="lines+markers",
                    marker=dict(
                        color=colors[6] if
                        integrals.loc[:,
                                      'baseline':'ketamine'].iloc[i, :].name[0]
                        == 'CSE' else colors[5],
                        symbol='circle',
                        size=10),
                    line=dict(
                        color=fillcolors[6] if
                        integrals.loc[:,
                                      'baseline':'ketamine'].iloc[i, :].name[0]
                        == 'CSE' else fillcolors[5],
                        width=3),
                    hovertemplate='Distance: %{y} px/ms<br>' +
                    'Week: week %{x}<br>' + "Mouse: %{meta[0]}<br>" +
                    "Trial: %{meta[1]}<br>" + "Facial feature: %{meta[2]}<br>",
                    ids=['week 0', 'week 1', 'week 5', 'week 6'],
                    legendgroup="Airpuff",
                    name="Airpuff CMS"
                    if integrals.loc[:,
                                     'baseline':'ketamine'].iloc[i, :].name[0]
                    == 'CSE' else "Airpuff Control",
                    showlegend=False if i not in [5, 39] else True))

        elif integrals.loc[:, 'baseline':'ketamine'].iloc[
                i, :].name[2] == "Sucrose" and f in [0, 2]:
            fig.add_trace(
                go.Scatter(
                    x=integrals.columns.tolist(),
                    y=integrals.loc[:, 'baseline':'ketamine'].iloc[
                        i, :].values.flatten(),
                    meta=[
                        integrals.loc[:, 'baseline':'ketamine'].iloc[
                            i, :].name[1],
                        integrals.loc[:, 'baseline':'ketamine'].iloc[
                            i, :].name[2],
                        integrals.loc[:,
                                      'baseline':'ketamine'].iloc[i, :].name[3]
                    ],
                    mode="lines+markers",
                    marker=dict(
                        color=colors[6] if
                        integrals.loc[:,
                                      'baseline':'ketamine'].iloc[i, :].name[0]
                        == 'CSE' else colors[5],
                        symbol='circle',
                        size=10),
                    line=dict(
                        color=fillcolors[6] if
                        integrals.loc[:,
                                      'baseline':'ketamine'].iloc[i, :].name[0]
                        == 'CSE' else fillcolors[5],
                        width=3),
                    hovertemplate='Distance: %{y} px/ms<br>' +
                    'Week: week %{x}<br>' + "Mouse: %{meta[0]}<br>" +
                    "Trial: %{meta[1]}<br>" + "Facial feature: %{meta[2]}<br>",
                    ids=['week 0', 'week 1', 'week 5', 'week 6'],
                    legendgroup="Sucrose",
                    name="Sucrose CMS"
                    if integrals.loc[:,
                                     'baseline':'ketamine'].iloc[i, :].name[0]
                    == 'CSE' else "Sucrose Control",
                    showlegend=False if i not in [0, 34] else True))

    if f == 0:
        pval = table9.loc["C(mouse_type):C(trial_type)", "PR(>F)"]
    else:
        pval = table9.loc["C(mouse_type)", "PR(>F)"]

    if pval <= 0.05:
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=1.02,
            y0=0.02,
            x1=1.02,
            y1=0.98,
            line=dict(
                color="Black",
                width=2,
            ),
        )
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=1.02,
            y0=0.02,
            x1=1.01,
            y1=0.02,
            line=dict(
                color="Black",
                width=2,
            ),
        )
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=1.02,
            y0=0.98,
            x1=1.01,
            y1=0.98,
            line=dict(
                color="Black",
                width=2,
            ),
        )

        if pval >= 0.01:
            fig.add_annotation(text="*",
                               xref="paper",
                               yref="paper",
                               align="right",
                               x=1.07,
                               y=0.5,
                               showarrow=False)
        elif pval >= 0.001:
            fig.add_annotation(text="**",
                               xref="paper",
                               yref="paper",
                               align="right",
                               x=1.07,
                               y=0.5,
                               showarrow=False)
        elif pval >= 0.001:
            fig.add_annotation(text="***",
                               xref="paper",
                               yref="paper",
                               align="right",
                               x=1.07,
                               y=0.5,
                               showarrow=False)
        else:
            fig.add_annotation(text="****",
                               xref="paper",
                               yref="paper",
                               align="right",
                               x=1.07,
                               y=0.5,
                               showarrow=False)

    pval = table9.loc["C(week)", "PR(>F)"]

    if pval <= 0.05:
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=0.02,
            y0=1.02,
            x1=0.98,
            y1=1.02,
            line=dict(
                color="Black",
                width=2,
            ),
        )
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=0.02,
            y0=1.02,
            x1=0.02,
            y1=1.01,
            line=dict(
                color="Black",
                width=2,
            ),
        )
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=0.98,
            y0=1.02,
            x1=0.98,
            y1=1.01,
            line=dict(
                color="Black",
                width=2,
            ),
        )

        if pval >= 0.01:
            fig.add_annotation(text="*",
                               xref="paper",
                               yref="paper",
                               align="center",
                               x=0.5,
                               y=1.07,
                               showarrow=False)
        elif pval >= 0.001:
            fig.add_annotation(text="**",
                               xref="paper",
                               yref="paper",
                               align="center",
                               x=0.5,
                               y=1.07,
                               showarrow=False)
        elif pval >= 0.001:
            fig.add_annotation(text="***",
                               xref="paper",
                               yref="paper",
                               aalign="center",
                               x=0.5,
                               y=1.07,
                               showarrow=False)
        else:
            fig.add_annotation(text="****",
                               xref="paper",
                               yref="paper",
                               align="center",
                               x=0.5,
                               y=1.07,
                               showarrow=False)

    fig.update_annotations(font_family="arial")

    fig.update_layout(
        title=(
            "Distance traveled from CS onset to three seconds after US end"),
        xaxis=dict(title="Week"),
        yaxis=dict(
            title="Facial Movement (px)",
            type="linear",
            ticks="outside",
            tick0=0,
            tickmode='array',
            tickvals=[0, 10000, 20000] if f in [0, 2] else [0, 5000, 10000],
            ticktext=[0, "10k", "20k"] if f in [0, 2] else [0, "5k", "10k"],
            range=[0, 25000] if f in [0, 2] else [0, 10000]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_family='arial',
        height=450,
        width=550)

    fig.write_image(
        f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_distance-traveled{emendation[f]}.pdf"
    )

    fig.write_image(
        f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_distance-traveled{emendation[f]}.svg"
    )

    fig.write_html(
        f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_distance-traveled{emendation[f]}.html"
    )

    fig.show()

# %%

table9.loc[[
    "C(week)", "C(mouse_type)", "C(mouse_type):C(trial_type)"
], :].to_csv(
    f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_distance-traveled-ANOVA.csv"
)

# %%
emendation = ["", "-Aripuff_only", "-Sucrose_only"]

mean_and_sem_dfs = pd.concat([mean_and_sem_df_csc, mean_and_sem_df_cse],
                             axis=0,
                             keys=["CSC", "CSE"],
                             names=["mouse_type"])
data_csc.index.names = ["mouse", "week", "trial", "frame"]
data_cse.index.names = ["mouse", "week", "trial", "frame"]
dat = pd.concat([data_csc, data_cse],
                axis=0,
                keys=["CSC", "CSE"],
                names=["mouse_type"])

integrals = integrals.astype({
    "baseline": np.float64,
    "early stress": np.float64,
    "late stress": np.float64,
    'ketamine': np.float64,
})

mean_and_sem_dfs.to_csv("/Volumes/specialk_cs/Mean_and_SEM_dataframes.csv")
dat.to_csv("/Volumes/specialk_cs/data.csv")
integrals.to_csv("/Volumes/specialk_cs/Integrated_points.csv")

mean_and_sem_dfs.to_excel("/Volumes/specialk_cs/Mean_and_SEM_dataframes.xls")
dat.to_excel("/Volumes/specialk_cs/data.xls")
integrals.to_excel("/Volumes/specialk_cs/Integrated_points.xls")

output = "/Volumes/specialk_cs/individual-traces.h5"
with pd.HDFStore(output) as file:
    file.put("Mean_and_SEM_dataframes", mean_and_sem_dfs)
    file.put("Normalized_indecies", dat)
    file.put("Integrated_points", integrals)

# %%
fig0 = go.Figure()
fig1 = go.Figure()
fig2 = go.Figure()
for f, fig in enumerate([fig0, fig1, fig2]):
    for i in range(len(integrals)):
        if integrals.loc[:, 'baseline':'ketamine'].iloc[
                i, :].name[2] == "Airpuff" and f in [0, 1]:
            fig.add_trace(
                go.Scatter(
                    x=integrals.columns.tolist(),
                    y=integrals.loc[:, 'baseline':'ketamine'].iloc[
                        i, :].values.flatten(),
                    meta=[
                        integrals.loc[:, 'baseline':'ketamine'].iloc[
                            i, :].name[1],
                        integrals.loc[:, 'baseline':'ketamine'].iloc[
                            i, :].name[2],
                        integrals.loc[:,
                                      'baseline':'ketamine'].iloc[i, :].name[3]
                    ],
                    mode="lines",
                    line=dict(
                        color=fillcolors[6] if
                        integrals.loc[:,
                                      'baseline':'ketamine'].iloc[i, :].name[0]
                        == 'CSE' else fillcolors[5],
                        width=3),
                    hovertemplate='Distance: %{y} px/ms<br>' +
                    'Week: week %{x}<br>' + "Mouse: %{meta[0]}<br>" +
                    "Trial: %{meta[1]}<br>" + "Facial feature: %{meta[2]}<br>",
                    ids=['week 0', 'week 1', 'week 5', 'week 6'],
                    legendgroup="Airpuff",
                    name="Airpuff CMS"
                    if integrals.loc[:,
                                     'baseline':'ketamine'].iloc[i, :].name[0]
                    == 'CSE' else "Airpuff Control",
                    showlegend=False if i not in [5, 39] else True))

        elif integrals.loc[:, 'baseline':'ketamine'].iloc[
                i, :].name[2] == "Sucrose" and f in [0, 2]:
            fig.add_trace(
                go.Scatter(
                    x=integrals.columns.tolist(),
                    y=integrals.loc[:, 'baseline':'ketamine'].iloc[
                        i, :].values.flatten(),
                    meta=[
                        integrals.loc[:, 'baseline':'ketamine'].iloc[
                            i, :].name[1],
                        integrals.loc[:, 'baseline':'ketamine'].iloc[
                            i, :].name[2],
                        integrals.loc[:,
                                      'baseline':'ketamine'].iloc[i, :].name[3]
                    ],
                    mode="lines",
                    line=dict(
                        color=fillcolors[6] if
                        integrals.loc[:,
                                      'baseline':'ketamine'].iloc[i, :].name[0]
                        == 'CSE' else fillcolors[5],
                        width=3),
                    hovertemplate='Distance: %{y} px/ms<br>' +
                    'Week: week %{x}<br>' + "Mouse: %{meta[0]}<br>" +
                    "Trial: %{meta[1]}<br>" + "Facial feature: %{meta[2]}<br>",
                    ids=['week 0', 'week 1', 'week 5', 'week 6'],
                    legendgroup="Sucrose",
                    name="Sucrose CMS"
                    if integrals.loc[:,
                                     'baseline':'ketamine'].iloc[i, :].name[0]
                    == 'CSE' else "Sucrose Control",
                    showlegend=False if i not in [0, 34] else True))

        if f == 0 and i < 4:

            n = 'CMS' if integrals.loc[:, 'baseline':'ketamine'].groupby(
                level=["mouse_type", "trial_type"]).apply('mean').iloc[
                    i, :].name[0] == 'CSE' else 'Control'
            t = integrals.loc[:, 'baseline':'ketamine'].groupby(
                level=["mouse_type", "trial_type"]).apply('mean').iloc[
                    i, :].name[1] == 'Airpuff'
            c = 0.4 if t else 0.6

            fig.add_trace(
                go.Scatter(
                    x=integrals.loc[:, 'baseline':'ketamine'].groupby(
                        level=["mouse_type", "trial_type"]).apply(
                            'mean').columns.tolist(),
                    y=integrals.loc[:, 'baseline':'ketamine'].groupby(
                        level=["mouse_type", "trial_type"]).apply('mean').iloc[
                            i, :],
                    meta=[
                        integrals.loc[:, 'baseline':'ketamine'].groupby(
                            level=["mouse_type", "trial_type"]).apply(
                                'mean').iloc[i, :].name[0],
                        integrals.loc[:, 'baseline':'ketamine'].groupby(
                            level=["mouse_type", "trial_type"]).apply(
                                'mean').iloc[i, :].name[1],
                    ],
                    mode="lines",
                    line=dict(color=f'rgba(220,20,60,{c})' if
                              integrals.loc[:, 'baseline':'ketamine'].groupby(
                                  level=["mouse_type", "trial_type"]).apply(
                                      'mean').iloc[i, :].name[0] == 'CSE' else
                              f'rgba(0,0,0,{c})',
                              width=15),
                    hovertemplate='Mean Distance: %{y} px/ms<br>' +
                    "Mouse type: %{meta[0]}<br>" + "Trial: %{meta[1]}<br>",
                    ids=['week 0', 'week 1', 'week 5', 'week 6'],
                    name=
                    f"{n} {integrals.loc[:, 'baseline':'ketamine'].groupby(level=['mouse_type', 'trial_type']).apply('mean').iloc[i, :].name[1]}",
                ))

        if f in [1, 2] and i < 2:

            name = integrals.xs(
                "Sucrose",
                level="trial_type").loc[:, 'baseline':'ketamine'].groupby(
                    level=["mouse_type"]
                ).apply('mean').iloc[i, :].name if f == 2 else integrals.xs(
                    "Airpuff",
                    level="trial_type").loc[:, 'baseline':'ketamine'].groupby(
                        level=["mouse_type"]).apply('mean').iloc[i, :].name

            fig.add_trace(
                go.Scatter(
                    x=integrals.xs("Sucrose", level="trial_type").
                    loc[:, 'baseline':'ketamine'].groupby(
                        level=["mouse_type"]).apply('mean').columns.tolist()
                    if f == 2 else integrals.xs("Airpuff", level="trial_type").
                    loc[:, 'baseline':'ketamine'].groupby(
                        level=["mouse_type"]).apply('mean').columns.tolist(),
                    y=integrals.xs("Sucrose", level="trial_type").
                    loc[:, 'baseline':'ketamine'].groupby(
                        level=["mouse_type"]).apply('mean').iloc[i, :]
                    if f == 2 else integrals.xs("Airpuff", level="trial_type").
                    loc[:, 'baseline':'ketamine'].groupby(
                        level=["mouse_type"]).apply('mean').iloc[i, :],
                    meta=[
                        name,
                    ],
                    mode="lines",
                    line=dict(color='rgba(220,20,60,0.5)'
                              if name == 'CSE' else 'rgba(0,0,0,0.5)',
                              width=15),
                    hovertemplate='Mean Distance: %{y} px/ms<br>' +
                    "Mouse type: %{meta[0]}<br>",
                    ids=['week 0', 'week 1', 'week 5', 'week 6'],
                    name='CMS' if name == 'CSE' else 'Control',
                ))

    if f == 0:
        pval = table9.loc["C(mouse_type):C(trial_type)", "PR(>F)"]
        # fig.add_trace(go.Sankey(
        #     node = dict(
        #     pad = 15,
        #     thickness = 20,
        #     line = dict(color = "black", width = 0.5),
        #     label = ["A1", "A2", "B1", "B2", "C1", "C2"],
        #     color = "blue"
        #     ),
        #     link = dict(
        #     source = [0, 1, 0, 2, 3, 3], # indices correspond to labels, eg A1, A2, A1, B1, ...
        #     target = [2, 3, 3, 4, 4, 5],
        #     value = [8, 4, 2, 8, 4, 2]
        # )))

    else:
        pval = table9.loc["C(mouse_type)", "PR(>F)"]

    if pval <= 0.05:
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=1.02,
            y0=0.02,
            x1=1.02,
            y1=0.98,
            line=dict(
                color="Black",
                width=2,
            ),
        )
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=1.02,
            y0=0.02,
            x1=1.01,
            y1=0.02,
            line=dict(
                color="Black",
                width=2,
            ),
        )
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=1.02,
            y0=0.98,
            x1=1.01,
            y1=0.98,
            line=dict(
                color="Black",
                width=2,
            ),
        )

        if pval >= 0.01:
            fig.add_annotation(text="*",
                               xref="paper",
                               yref="paper",
                               align="right",
                               x=1.07,
                               y=0.5,
                               showarrow=False)
        elif pval >= 0.001:
            fig.add_annotation(text="**",
                               xref="paper",
                               yref="paper",
                               align="right",
                               x=1.07,
                               y=0.5,
                               showarrow=False)
        elif pval >= 0.001:
            fig.add_annotation(text="***",
                               xref="paper",
                               yref="paper",
                               align="right",
                               x=1.07,
                               y=0.5,
                               showarrow=False)
        else:
            fig.add_annotation(text="****",
                               xref="paper",
                               yref="paper",
                               align="right",
                               x=1.07,
                               y=0.5,
                               showarrow=False)

    pval = table9.loc["C(week)", "PR(>F)"]

    if pval <= 0.05:
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=0.02,
            y0=1.02,
            x1=0.98,
            y1=1.02,
            line=dict(
                color="Black",
                width=2,
            ),
        )
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=0.02,
            y0=1.02,
            x1=0.02,
            y1=1.01,
            line=dict(
                color="Black",
                width=2,
            ),
        )
        fig.add_shape(
            type="line",
            xref="paper",
            yref="paper",
            x0=0.98,
            y0=1.02,
            x1=0.98,
            y1=1.01,
            line=dict(
                color="Black",
                width=2,
            ),
        )

        if pval >= 0.01:
            fig.add_annotation(text="*",
                               xref="paper",
                               yref="paper",
                               align="center",
                               x=0.5,
                               y=1.07,
                               showarrow=False)
        elif pval >= 0.001:
            fig.add_annotation(text="**",
                               xref="paper",
                               yref="paper",
                               align="center",
                               x=0.5,
                               y=1.07,
                               showarrow=False)
        elif pval >= 0.001:
            fig.add_annotation(text="***",
                               xref="paper",
                               yref="paper",
                               aalign="center",
                               x=0.5,
                               y=1.07,
                               showarrow=False)
        else:
            fig.add_annotation(text="****",
                               xref="paper",
                               yref="paper",
                               align="center",
                               x=0.5,
                               y=1.07,
                               showarrow=False)

    fig.update_annotations(font_family="arial")

    fig.update_layout(title=(
        f"Distance traveled from CS onset to three seconds after US end\n{emendation[f].replace('-', '').replace('_', ' ')}"
    ),
                      xaxis=dict(title="Week"),
                      yaxis=dict(
                          title="Facial Movement (px)",
                          type="linear",
                          ticks="outside",
                          tick0=0,
                          tickmode='array',
                          tickvals=[0, 10000, 20000]
                          if f in [0, 2] else [0, 5000, 10000],
                          ticktext=[0, "10k", "20k"]
                          if f in [0, 2] else [0, "5k", "10k"],
                          range=[0, 25000] if f in [0, 2] else [0, 10000]),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      font_family='arial',
                      height=450,
                      width=550)

    fig.write_image(
        f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_distance-traveled{emendation[f]}.pdf"
    )

    fig.write_image(
        f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_distance-traveled{emendation[f]}.svg"
    )

    fig.write_html(
        f"/Users/annieehler/Projects/Jupyter_Notebooks/python_outputs/graphs/{today}_distance-traveled{emendation[f]}.html"
    )

    fig.show()

# %%
# %%

I