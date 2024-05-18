import random
from collections import OrderedDict
from collections import defaultdict
import warnings
import copy
import numpy as np
import math
import math
import operator
import ast
import logging
logger = logging.getLogger("logger")

class SearchVariable:
    def __init__(self, name, _type, domain):
        self.name = name
        self._type = _type
        self.domain = domain

class Action:
    def __init__(self, coordinates, search_variable_names):
        self.coordinates = coordinates
        self.search_variable_names = search_variable_names
        self.details = dict(zip(search_variable_names, coordinates))

class NelderMead:
    def __init__(self, name, malicious_mode, num_classes, malicious_group_list, main_goal,
                 step=1, alpha=1., gamma=2., rho=-0.5, sigma=0.5):
        self.initial_action_space = []
        self.name = name
        self.malicious_mode = malicious_mode
        self.num_classes = num_classes
        self.malicious_group_list = malicious_group_list
        self.g_main = main_goal
        self.current_action = None
        self.mode = 'initial'
        self.submode = 'front'
        self.search_space = None
        self.best = None
        self.iters = 0
        self.init_curr_dim=0
        self.step = step
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.xr = None
        self.x0 = None
        self.xe = None
        self.xc = None
        self.nres = None
        self.x1 = None
        self.reduction_loop_i = None
        self.redx = None

        # 2D plane search space and the 2 key hyperparameters
        # goal_concentration: how many of the goals has to be the same from the g_main
        # goal_conce == class num, untarget_1;
        # goal_conce == 0, untarget_2
        self.goal_conce = SearchVariable(name="goal_concentration", _type='discrete_continuous', domain=list(range(self.num_classes)))
        # zoom_factor, the factor times the malicious train's gradient, the factor is the index of 10
        self.zoom_factor = SearchVariable(name="zoom_factor", _type='continuous', domain=[1, 0, -1, -2, -3])
        shape = (len(self.goal_conce.domain), len(self.zoom_factor.domain))
        self.search_space = np.empty(shape, dtype=object)

        for i in range(len(self.goal_conce.domain)):
            for j in range(len(self.zoom_factor.domain)):
                self.search_space[i][j] = [self.goal_conce.domain[i], self.zoom_factor.domain[j]]

        # four_corners = self.search_space[[0, 0, -1, -1], [0, -1, 0, -1]]
        # random.shuffle(four_corners)
        # for i in four_corners:
        #     parameter_values = {goal_conce.name: i[0], zoom_factor.name: i[1]}
        #     self.initial_action_space.append(Action(i, parameter_values))

        # init
        self.x_start = [find_middle(self.goal_conce.domain), find_middle(self.zoom_factor.domain)]
        self.dim = len(self.x_start)
        self.prev_best = None
        self.res = []

    def nelder_mead(self):
        if self.mode == 'initial':
            if self.iters == 0:
                self.iters += 1
                return self.coordinates_to_action(self.x_start)
            elif 1 <= self.iters <= len(self.x_start) + 1:
                x = copy.copy(self.x_start)
                x[self.init_curr_dim] = random.choice(
                    [x[self.init_curr_dim] + self.step, x[self.init_curr_dim] - self.step])
                self.iters += 1
                self.init_curr_dim += 1
                if self.iters == len(self.x_start) + 1:
                    self.mode = 'centroid'
                return self.coordinates_to_action(x)
        elif self.mode == 'centroid':
            self.x0 = [0.] * self.dim
            for tup in self.res[:-1]:
                for i, c in enumerate(tup[0]):
                    self.x0[i] += c / (len(self.res) - 1)
            self.mode = 'reflection'
            return self.nelder_mead
        elif self.mode == 'reflection':
            if self.submode == 'before':
                self.xr = self.x0 + self.alpha * (self.x0 - self.res[-1][0])
                return self.coordinates_to_action(self.xr)
            elif self.submode == 'after':
                if self.res[0][1] <= self.rscore < self.res[-2][1]:
                    del self.res[-1]
                    self.res.append([self.xr, self.rscore])
                self.mode = 'expansion'
                self.submode = 'before'
        elif self.mode == 'expansion':
            if self.submode == 'before':
                if self.rscore < self.res[0][1]:
                    self.xe = self.x0 + self.gamma * (self.x0 - self.res[-1][0])
                    return self.coordinates_to_action(self.xe)
            elif self.submode == 'after':
                if self.escore < self.rscore:
                    del self.res[-1]
                    self.res.append([self.xe, self.escore])
                else:
                    del self.res[-1]
                    self.res.append([self.xr, self.rscore])
                self.mode = 'contraction'
                self.submode = 'before'
        elif self.mode == 'contraction':
            if self.submode == 'before':
                self.xc = self.x0 + self.rho * (self.x0 - self.res[-1][0])
                return self.coordinates_to_action(self.xc)
            elif self.submode == 'after':
                if self.cscore < self.res[-1][1]:
                    del self.res[-1]
                    self.res.append([self.xc, self.cscore])
            self.mode = 'reduction'
            self.submode = 'before'
        elif self.mode == 'reduction':
            if self.submode == 'before':
                self.reduction_loop_i = 0
                self.nres = []
                self.x1 = self.res[0][0]

            if 0 <= self.reduction_loop_i <= len(self.res):
                self.redx = self.x1 + self.sigma * (self.res[self.reduction_loop_i][0] - self.x1)
                return self.coordinates_to_action(self.redx)

    def coordinates_to_action(self, coordinates):
        search_variable_names = [self.goal_conce.name, self.zoom_factor.name]
        return Action(coordinates, search_variable_names)

    def record(self, rewards):
        self.res.append([self.current_action.coordinates, rewards])

        if self.mode == 'reflection':
            self.rscore = rewards
            self.submode = 'after'
        elif self.mode == 'expansion':
            self.escore = rewards
            self.submode = 'after'
        elif self.mode == 'contraction':
            self.cscore = rewards
            self.submode = 'after'
        elif self.mode == 'reduction':
            self.dscore = rewards
            self.nres.append([self.redx, self.dscore])
            if len(self.nres) == len(self.res):
                self.res = self.nres
                self.submode = 'before'
                self.mode = 'centroid'


def find_middle(input_list):
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return input_list[int(middle - .5)]
    else:
        return random.choice([input_list[int(middle)], input_list[int(middle-1)]])