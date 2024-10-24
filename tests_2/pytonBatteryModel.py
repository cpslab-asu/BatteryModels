import logging
import pathlib
import enum
import liionpack as lp

import numpy as np
from numpy.random import default_rng

from staliro.core.interval import Interval
from staliro.core.model import Model, ModelInputs, Trace, ExtraResult
import numpy as np
from numpy.typing import NDArray

from pybamm.parameters.parameter_values import ParameterValues

import pybamm

BatteryDataT = NDArray[np.float_]
BatteryResultT = ExtraResult[BatteryDataT, BatteryDataT]
 
class LiIonBatteryModel(enum.IntEnum):
    DFN = enum.auto()
    SPM = enum.auto()

class PyBammBatteryModel_onlyCurrent(Model[BatteryDataT, None]):
    MODEL_NAME = "PyBaMM Battery Model with only Current as Input"

    def __init__(self, samplingStep:float, initialSoc:float) -> None:

        battery_options = {"thermal": "x-full"} 
        # self.model = pybamm.lithium_ion.DFN(options=battery_options)
        self.initialSoc = initialSoc
        parameter_values = pybamm.ParameterValues("Chen2020")

        # parameter_values['Lower voltage cut-off [V]'] = 0.3
        # parameter_values['Current function [A]'] = 7
        parameter_values['Nominal cell capacity [A.h]'] = 3.35
        parameter_values['Electrode width [m]'] = 1.1025
        parameter_values['Positive electrode thickness [m]'] = 7.2e-5
        parameter_values['Positive electrode porosity'] = 0.203
        parameter_values['Positive electrode conductivity [S.m-1]'] = 0.03
        parameter_values['Positive particle radius [m]'] = 5.45e-6
        parameter_values['Positive electrode diffusivity [m2.s-1]'] = 2.5e-14
        parameter_values['Separator thickness [m]'] = 2.5e-5
        parameter_values['EC initial concentration in electrolyte [mol.m-3]'] = 1e3
        parameter_values['Negative electrode thickness [m]'] = 9.3e-5
        parameter_values['Negative electrode porosity'] = 0.231
        parameter_values['Negative electrode conductivity [S.m-1]'] = 4.7e-1
        parameter_values['Negative electrode active material volume fraction'] = 0.6932
        parameter_values['Negative electrode diffusivity [m2.s-1]'] = 8e-15
        # parameter_values['Upper voltage cut-off [V]'] = 10


        # battery_options = {"thermal": "x-full"}

        self.parameter_values = parameter_values

        self.sampling_step = samplingStep


    def simulate(self, signals: ModelInputs, intrvl: Interval) -> BatteryResultT:
        
        n_times = (intrvl.length // self.sampling_step)+2
        signal_times = np.linspace(intrvl.lower, intrvl.upper, num=int(n_times))
        signal_values = np.array(
            [[signal.at_time(t) for t in signal_times] for signal in signals.signals]
        )

        netlist = lp.setup_circuit(Np=2, Ns=4, Rb=1e-3, Rc=1e-2)
        current_drawn_sinal= np.column_stack([signal_times, -1*signal_values.T])

        # experiment = pybamm.Experiment(["Charge at 7A for 120 minutes"], period="10 seconds")

        experiment = pybamm.Experiment([pybamm.step.current(current_drawn_sinal)], period=f"{self.sampling_step} second")
        # experiment = pybamm.Experiment(["Discharge at 7A for 120 minutes"], period="1 second")
        


        output_variables = [
            "Volume-averaged cell temperature [K]",    
        ]

        solution = lp.solve(netlist=netlist,sim_func=lp.thermal_simulation,
                            parameter_values=self.parameter_values,
                            output_variables=output_variables,
                            experiment=experiment,
                            initial_soc=self.initialSoc,
                            inputs={"Total heat transfer coefficient [W.m-2.K-1]": np.ones(8) * 10})  
        

        times: list[float] = solution['Time [s]']
        
        # states: list[list[float]] = [solution["Pack terminal voltage [V]"].tolist(), solution["Volume-averaged cell temperature [K]"].mean(1).tolist()]
        times: list[float] = solution['Time [s]']
        data = [solution["Terminal voltage [V]"][:,col].tolist() for col in range(solution["Terminal voltage [V]"].shape[1])]
        data.append(solution["Pack terminal voltage [V]"].tolist())
        data.append(solution["Volume-averaged cell temperature [K]"].mean(1).tolist())
        # states: list[list[float]] = [solution["Pack terminal voltage [V]"].tolist(), solution["Volume-averaged cell temperature [K]"].mean(1).tolist()]
        states: list[list[float]] = np.array(data).T.tolist()

        trace = Trace(times, states)
        inTrace = Trace(signal_times, np.array(signal_values))
        return BatteryResultT(trace, inTrace)






