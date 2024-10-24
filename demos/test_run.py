from model_file import BatteryModel
from pytonBatteryModel import PyBammBatteryModel_onlyCurrent

import pathlib
import numpy as np
from matplotlib import pyplot as plt

from staliro.core.sample import Sample
from staliro.options import Options, SignalOptions
from staliro.specifications import RTAMTDense
from staliro.staliro import simulate_model
from staliro.signals import piecewise_constant



#####################################################################################################################
# Define Specifications

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

#####################################################################################################
# Define Signals
signals = [
    SignalOptions(control_points = [(-20,-0.65)]*7, signal_times=np.linspace(0.,4000.,7, endpoint=False), factory=piecewise_constant),
]

#####################################################################################################
# Define Options
options = Options(runs=1, iterations=1, interval=(0, 4000), signals=signals, seed = 1234)

#####################################################################################################

#####################################################################################################
# Define Run

signal_bounds = sum((signal.control_points for signal in signals), ())
bounds = options.static_parameters + signal_bounds
rng = np.random.default_rng(12345)
sample = np.array(Sample([rng.uniform(bound.lower, bound.upper) for bound in bounds]).values)

# sample = np.array([-1.6054437561308035,
#                     -19.822721644349404,
#                       -19.699069528847534,
#                         -12.510631207910507,
#                           -12.316772777540336,
#                             -8.875077713084128,
#                               -13.671999524755861])

sample = np.array([-7,-7,-7,-7,-7,-7,-7])*-1
simulinkBattery = BatteryModel(initialSoc=0.2)
battery_model = PyBammBatteryModel_onlyCurrent(samplingStep=0.1, initialSoc = 0.2)

result1 = simulate_model(simulinkBattery, options, sample)
result2 = simulate_model(battery_model, options, sample)


fig, ax = plt.subplots()    
ax.plot(result1.extra.times , result1.extra.states, label = "Current Drawn")
ax.legend()
plt.savefig("1_in_CurrentDrawn.pdf")

    
out_trace1 = np.array(result1.trace.states)
out_trace2 = np.array(result2.trace.states)
my_data = np.genfromtxt('battery_discharge_1c_7000mah.csv', delimiter=',')
new_data = my_data[1:, :]

t_vals = new_data[:,0]*60
temp_vals = (new_data[:,1] - 32) * (5 / 9) + 273.15
v = new_data[:,2]

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))    
ax1.plot(result1.trace.times, out_trace1[:,-8:].mean(1), label = "Temperature (K) Simulink")
ax1.plot(result2.trace.times, out_trace2[:,9], label = "Temperature (K) PyBaMM")
# ax1.plot(t_vals, temp_vals, label = "Temperature (K) from Measurement")
# ax1.plot(range(4000), [273.15+60]*4000, label = "NCR18650GA Threshold")
ax1.legend()
ax1.set_xlabel("Time (s)")
# plt.savefig("3_out_Temperature.pdf")

# fig, ax = plt.subplots()    
ax2.plot(result1.trace.times, out_trace1[:,17], label = "Voltage (V) Simulink")
ax2.plot(result2.trace.times, out_trace2[:,8], label = "Voltage (V) PyBaMM")
# ax2.plot(t_vals, v, label = "Voltage (V) from Measurement")
ax2.set_xlabel("Time (s)")
ax2.legend()

# plt.show()
plt.savefig("Plots_OverVoltage.pdf")
