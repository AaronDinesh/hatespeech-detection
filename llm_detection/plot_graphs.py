"""
This file is mainly to help plot the graphs. It is not intended to be reused as-is. Just a hacky script to get something
plotted for the poster + report
"""

import plotly.express as px
import plotly.graph_objects as go
import numpy as np


mat = np.array([[0.12544234, 0.04886848, 0.29476446, 0.53092472],
          [0.09618921, 0.04677458, 0.27525809, 0.58177813],
          [0.0673358 , 0.0375066 , 0.23014615, 0.66501145],
          [0.04408677, 0.02682529, 0.17133193, 0.75775601]])

fig = px.imshow(
    mat,
    x=["0", "1", "2", "3"], y=["0", "1", "2", "3"],
    labels=dict(x="LLM", y="Human", color="P(Human|LLM)"),
    color_continuous_scale="viridis",  # ← swap here
    text_auto=".2f",
    zmin=0, zmax=1,
)

fig.show()

