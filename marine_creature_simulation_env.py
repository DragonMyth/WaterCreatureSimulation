import numpy as np
import pydart2 as pydart

import creature_simulators


class MarineCreatureSimEnv(object):
    def __init__(self, simulator):
        self.simulator = simulator

    def debug_on(self):
        self.simulator.debug = True

    def debug_off(self):
        self.simulator.debug = False

    def render_result(self):
        pydart.gui.viewer.launch_pyqt5(self.simulator)

    def simulate(self, horizon):
        terminate_flag = False
        while self.simulator.t < horizon and not terminate_flag:
            self.simulator.step()
            curr_q = self.simulator.skeletons[0].q
            if np.isnan(curr_q).any():
                terminate_flag = True
                print("NAN")
        res = self.simulator.skeletons[0].q
        if terminate_flag:
            for i in range(len(res)):
                res[i] = 0
        return res