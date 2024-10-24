from staliro.core.interval import Interval
from staliro.core.model import Model, ModelInputs, Trace, ExtraResult
import numpy as np
from numpy.typing import NDArray
try:
    import matlab
    import matlab.engine
except ImportError:
    _has_matlab = False
else:
    _has_matlab = True
 
BatteryDataT = NDArray[np.float_]
BatteryResultT = ExtraResult[BatteryDataT, BatteryDataT]
 
 
class BatteryModel(Model[BatteryDataT, None]):
    MODEL_NAME = "test_fan_aircooling_3"
 
    def __init__(self, initialSoc) -> None:
        if not _has_matlab:
            raise RuntimeError(
                "Simulink support requires the MATLAB Engine for Python to be installed"
            )
 
        engine = matlab.engine.start_matlab()
        model_opts = engine.simget(self.MODEL_NAME)
        self.model_opts = engine.simset(model_opts, "SaveFormat", "Array")
        engine.load("Batteries_v2.mat")
        # engine.addpath("examples")
        
        self.sampling_step = 0.1
        self.engine = engine
        self.engine.eval(f"cellInitialSoc = repmat({initialSoc},8,1);", nargout=0)
        print("Model Initialized")

    def simulate(self, signals: ModelInputs, intrvl: Interval) -> BatteryResultT:
        sim_t = matlab.double([0, intrvl.upper])
        n_times = (intrvl.length // self.sampling_step) + 2
        signal_times = np.linspace(intrvl.lower, intrvl.upper, int(n_times))
        signal_values = np.array([[signal.at_time(t) for t in signal_times] for signal in signals.signals])

        model_input = matlab.double(np.row_stack((signal_times, signal_values)).T.tolist())

        self.engine.workspace['u'] = model_input
        self.engine.workspace['T'] = intrvl.upper

        
        results = self.engine.sim(\
                self.MODEL_NAME, 'StopTime', 'T', \
                'LoadExternalInput', 'on', 'ExternalInput', 'u', \
                'SaveTime', 'on', 'TimeSaveName', 'tout', \
                'SaveOutput', 'on', 'OutputSaveName', 'yout', \
                'SaveFormat', 'array', nargout=1)
        # self.engine.workspace["data"] = self.engine.sim(
        #     self.MODEL_NAME, sim_t, self.model_opts, model_input, nargout=1
        # )
        self.engine.workspace["results"] = results
        timestamps_list = np.array(self.engine.eval("results.tout", nargout = 1)).flatten()
        data_list = np.array(self.engine.eval("results.yout", nargout = 1))

        new_t, new_y, index = self.chopTrace(timestamps_list, data_list)

        trace = Trace(new_t, new_y)

        inTrace = Trace(signal_times[:index], np.array(signal_values)[0, :index].T)
        return BatteryResultT(trace, inTrace)
    
    def chopTrace(self, t, y):
        index = np.where((y[:,9] < 0) | (y[:,9] > 1))[0][0]
        
        y = y[:index, :]
        t = t[:index]
        return t,y, len(t)