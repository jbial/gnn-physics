"""Base object for the physical systems
Credit to https://github.com/YunzhuLi/CompositionalKoopmanOperators
"""
import numpy as np


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def rand_int(lo, hi):
    return np.random.randint(lo, hi)


def calc_dis(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def norm(x, p=2):
    return np.power(np.sum(x ** p), 1. / p)


class Engine(object):
    def __init__(self, dt, state_dim, param_dim):
        self.dt = dt
        self.state_dim = state_dim
        self.param_dim = param_dim

        self.state = None
        self.action = None
        self.param = None

        self.init()

    def init(self):
        pass

    def get_param(self):
        return self.param.copy()

    def set_param(self, param):
        self.param = param.copy()

    def get_state(self):
        return self.state.copy()

    def set_state(self, state):
        self.state = state.copy()

    def get_scene(self):
        return self.state.copy(), self.param.copy()

    def set_scene(self, state, param):
        self.state = state.copy()
        self.param = param.copy()

    def d(self, state, t, param):
        # time derivative
        pass

    def step(self):
        pass

    def render(self, state, param):
        pass

    def clean(self):
        pass


