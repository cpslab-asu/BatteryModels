import logging
import math
from math import pi
import numpy as np
from collections import OrderedDict
import plotly.graph_objects as go

from model_file import BatteryModel

from staliro.core import worst_eval, worst_run
from staliro.options import Options
from staliro.specifications import RTAMTDense
from staliro.staliro import simulate_model, staliro
from staliro.options import Options, SignalOptions
from staliro.signals import piecewise_constant



from staliroBoInterface import Behavior, BO
from bo.gprInterface import InternalGPR
from bo.bayesianOptimization import InternalBO

import pickle

# condition_string = " and ".join([f"v{i} > 2" for i in range(1, 9)])
# phi = f"G({condition_string})"
phi = "G(T<333)"
spec = RTAMTDense(phi, {
    "T":0,
        "v1": 1,
        "v2": 2,
        "v3": 3,
        "v4": 4,
        "v5": 5,
        "v6": 6,
        "v7": 7,
        "v8": 8,
    })

signals = [
    SignalOptions(control_points = [(-20,-0.65)]*7, signal_times=np.linspace(0.,4000.,7, endpoint=False), factory=piecewise_constant),
]

options = Options(runs=1, iterations=300, interval=(0, 36), signals=signals, seed = 1234)

# Define the optimizer
gpr_model = InternalGPR()
bo_model = InternalBO()
optimizer = BO(20, gpr_model, bo_model, "lhs_sampling", Behavior.FALSIFICATION)

f16_model = BatteryModel()


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    result = staliro(f16_model, spec, optimizer, options)
    best_sample = worst_eval(worst_run(result)).sample
    # best_result = simulate_model(f16_model, options, best_sample)
    # print(best_result.trace.states)
    # print(np.array(best_result.trace.states).shape)
    # figure = go.Figure()
    # figure.add_trace(
    #     go.Scatter(
    #         name="Altitude",
    #         x=np.array(best_result.trace.times),
    #         y=np.array(best_result.trace.states)[:,2],
    #         mode="lines",
    #         line_color = "blue",
    #     )
    # )
    # figure.add_trace(
    #     go.Scatter(
    #         name="Speed",
    #         x=np.array(best_result.trace.times),
    #         y=np.array(best_result.trace.states)[:,5],
    #         mode="lines",
    #         line_color = "green",
    #     )
    # )
    # figure.update_layout(title=f"Example 1: {[round(a,5) for a in best_sample.values]}", xaxis_title="time (s)")
    # figure.write_image("fig1.pdf")
    with open("MODE_1.pkl", "wb") as f:
        pickle.dump([result, best_sample], f)