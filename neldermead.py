import copy
import random
import numpy as np
import math
import warnings
import logging

logger = logging.getLogger("logger")


class NelderMead:
    def __init__(self, name, x_start, progress, rl_manual, manual_coordinates, empirical_points,
                 step=0.1, alpha=1., gamma=2., rho=-0.5, sigma=0.5):
        self.name = name
        self.rl_manual = rl_manual
        self.manual_coordinates = np.asarray([float(i) for i in manual_coordinates])
        self.mode = 'initial'
        self.submode = 'front'
        self.search_space = None
        self.progress = progress
        self.best = None
        if self.progress:
            self.iters = 0
        else:
            self.iters = 0
        self.init_curr_dim = 0
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
        self.dscore = None
        self.cscore = None
        self.escore = None
        self.rscore = None
        self.if_restart = False
        self.init_spot_search = True
        self.empirical_points = empirical_points

        # init
        if not self.rl_manual:
            self.x_start = x_start
        else:
            self.x_start = self.manual_coordinates
        self.dim = len(self.x_start)
        self.prev_best = None
        self.res = []
        self.reduction_loop_i = 0
        self.init_loop_i = 0
        self.nres = []
        self.history = []
        self.triangle = generate_init_triangle(self.x_start, self.x_start, self.step)
        self.rounds_no_improv = 0

    def nelder_mead(self):
        # print("self.iters", self.iters)
        if self.mode == 'initial':
            if self.iters == 0 or self.if_restart:
                self.reduction_loop_i = 0
                self.iters += 1
                self.if_restart = False
                # return self.x_start
                return self.triangle[0]
            else:
                if self.reduction_loop_i <= self.dim - 1:
                    # x = copy.copy(self.x_start)

                    # diff ways to start the triangle
                    # 1. add 1 unit of step
                    # x[self.init_curr_dim] = x[self.init_curr_dim] + self.step
                    # 2. randomly select add or reduce a step
                    # x[self.init_curr_dim] = random.choice(
                    #     [x[self.init_curr_dim] + self.step, x[self.init_curr_dim] - self.step])
                    # 3. select from randomly generated triangle on circle
                    print("self.reduction_loop_i", self.reduction_loop_i)
                    print("self.triangle", self.triangle)
                    print("self.init_curr_dim", self.init_curr_dim)
                    x = self.triangle[self.init_curr_dim + 1]

                    self.init_curr_dim += 1
                    self.reduction_loop_i += 1
                    self.iters += 1
                    return x
                else:
                    self.mode = 'centroid'
                    # print("self.res", self.res)
                    return False
        elif self.mode == 'centroid':
            # order
            # logger.info(f"self.res: {self.res}")
            self.res = sorted(self.res, key=lambda x: x[1])

            # print("centroid len self.res:", len(self.res))
            self.best = self.res[0][1]
            self.reduction_loop_i = 0

            logger.info(f"best so far: {self.res[0]}")

            # centroid
            self.x0 = [0.] * self.dim
            for tup in self.res[:-1]:
                for i, c in enumerate(tup[0]):
                    self.x0[i] += c / (len(self.res) - 1)
            self.mode = 'reflection'
            self.submode = 'before'
            return False
        elif self.mode == 'reflection':
            if self.submode == 'before':
                self.xr = self.x0 + self.alpha * (self.x0 - self.res[-1][0])
                self.iters += 1
                return self.xr
            if self.submode == 'after':
                if self.res[0][1] <= self.rscore < self.res[-2][1]:
                    # print("ref self.res 1", len(self.res))
                    del self.res[-1]
                    # print("ref self.res 2", len(self.res))
                    self.res.append([self.xr, self.rscore])
                    # print("self.res.append", len(self.res))
                    self.mode = 'centroid'
                    self.submode = 'before'
                    return False
                else:
                    # print('ref-after 2')
                    self.mode = 'expansion'
                    self.submode = 'before'
                    return False
        elif self.mode == 'expansion':
            if self.submode == 'before':
                # print('expansion - top')
                if self.rscore < self.res[0][1]:
                    # print('expansion - rscore')
                    self.xe = self.x0 + self.gamma * (self.x0 - self.res[-1][0])
                    self.iters += 1
                    return self.xe
                else:
                    # print('expansion - passing')
                    self.mode = 'contraction'
                    self.submode = 'before'
                    return False
            if self.submode == 'after':
                if self.escore < self.rscore:
                    # print("exp self.res 1", len(self.res))
                    del self.res[-1]
                    # print("exp self.res 2", len(self.res))
                    self.res.append([self.xe, self.escore])
                    # print("self.res.append", len(self.res))
                    self.mode = 'centroid'
                    self.submode = 'before'
                    return False
                else:
                    # print("exp self.res 3", len(self.res))
                    del self.res[-1]
                    # print("exp self.res 4", len(self.res))
                    self.res.append([self.xr, self.rscore])
                    # print("self.res.append", len(self.res))
                    self.mode = 'centroid'
                    self.submode = 'before'
                    return False

        elif self.mode == 'contraction':
            if self.submode == 'before':
                self.xc = self.x0 + self.rho * (self.x0 - self.res[-1][0])
                return self.xc
            if self.submode == 'after':
                if self.cscore < self.res[-1][1]:
                    # print("contr 1", len(self.res))
                    del self.res[-1]
                    # print("contr 2", len(self.res))
                    self.res.append([self.xc, self.cscore])

                    self.mode = 'centroid'
                    self.submode = 'before'
                    return False
                else:
                    self.mode = 'reduction'
                    self.submode = 'before'
                    return False
        elif self.mode == 'reduction':
            # if self.submode == 'before':
            #     self.x1 = self.res[0][0]
            # print("self.reduction_loop_i", self.reduction_loop_i)
            # print("len(self.res)", len(self.res))
            # print("neldermead - in reduction loop!")
            if 0 <= self.reduction_loop_i <= len(self.res) - 1:
                if self.reduction_loop_i == 0:
                    self.x1 = self.res[0][0]
                elif self.reduction_loop_i == len(self.res) - 1:
                    self.reduction_loop_i = 0
                self.redx = self.x1 + self.sigma * (self.res[self.reduction_loop_i][0] - self.x1)
                self.reduction_loop_i += 1
                self.submode = 'after'
                self.iters += 1
                return self.redx
            else:
                warnings.warn('Warning Message: reduction self.reduction_loop_i not within range')
                return False

    def record(self, kpi, action, progress):
        action = np.array(action)

        if progress is True:
            if self.iters == 0:
                self.history.append([action, kpi, None])
                if self.rl_manual or self.init_spot_search:
                    self.iters += 1
                return
            else:
                # with progress flag, rewards is kpi - kpi(last round)
                rewards = (kpi - self.history[-1][1])

                if rewards == 0.0:
                    logger.info("Debugging when Reward = 0")
                    logger.info(f"self.mode: {self.mode}")
                    logger.info(f"self.submode: {self.submode}")
                elif rewards < 0:
                    logger.info(f"Found negative reward in iter {self.iters}")
                self.history.append([action, kpi, rewards])
        else:
            rewards = kpi
            self.history.append([action, kpi, rewards])

        if self.rl_manual or self.init_spot_search:
            self.iters += 1

        logger.info("rl history")
        for i in range(len(self.history)):
            logger.info(f"{i + 1}, {self.history[i]}")

        if self.mode == 'initial':
            self.res.append([action, rewards])
            # print("self.initial.append", len(self.res))
        if self.mode == 'centroid':
            pass
        elif self.mode == 'reflection':
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

    def get_dominate_strategy(self):
        if len(self.history) == 0:
            return None
        return str(list(self.history[-1][0]))

    def reach_restart_threshold(self, threshold):
        if len(self.history) <= 2:
            return False
        elif len(self.history) >= 3:
            last_reward = self.history[-1][2]
            second_last_reward = self.history[-2][2]
            # if last_reward > 0 and last_reward > second_last_reward:
            if last_reward > 0:
                self.rounds_no_improv += 1
            else:
                self.rounds_no_improv = 0
            # when reach threshold
            if self.rounds_no_improv == threshold:
                self.rounds_no_improv = 0
                logger.info(f"RL search restart at {len(self.history)} rounds")
                return True
            else:
                return False
        """
        last_reward = self.history[-1][2]
        # second_last_reward = self.history[-2][2]
        if last_reward is not None:
            if last_reward <= 0:
                self.rounds_no_improv = 0
                return False
            else:
                if second_last_reward <= last_reward:
                    self.rounds_no_improv = 0
                else:
                    self.rounds_no_improv += 1
                    if self.rounds_no_improv == threshold+1:
                        self.rounds_no_improv = 0
                        logger.info(f"RL search restart at {len(self.history)} rounds")
                        return True
                    else:
                        return False
        else:
            return False
        """

    def restart(self, new_start):
        self.triangle = generate_init_triangle(self.x_start, new_start, self.step)
        self.mode = 'initial'
        self.reduction_loop_i = 0
        self.init_loop_i = 0
        self.init_curr_dim = 0
        self.if_restart = True
        self.res = []


def f(x, domain):
    for i in range(len(x)):
        if domain[i][0] <= x[i] <= domain[i][1]:
            continue
        else:
            return float('inf')
    # Himmelblau function
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

def f1d(x, domain):
    # Heuristic function
    return (x[0]%271)*np.sin(x[0]/37)*np.exp(-0.001*x[0])

def generate_init_triangle(x_start, center, r):
    if len(x_start) == 2:
        circle_x = center[0]
        circle_y = center[1]
        triangle = []
        # random angle
        alpha = 2 * math.pi * random.random()
        for i in range(len(center) + 1):
            x = r * math.cos(alpha) + circle_x
            y = r * math.sin(alpha) + circle_y
            alpha += math.pi * 2 / 3
            triangle.append([x, y])
    elif len(x_start) == 1:
        center = center[0]
        triangle = [[center - r], [center + r]]
    else:
        warnings.warn("x_start need to be in 1, or 2 dimensions")
        return False
    return triangle


def find_middle(input_list):
    middle = float(len(input_list)) / 2
    if middle % 2 != 0:
        return input_list[int(middle - .5)]
    else:
        return random.choice([input_list[int(middle)], input_list[int(middle - 1)]])


def main():
    """
    #2D
    x_domain = np.array([[-6, 6], [-6, 6]])

    x_start = []
    for i in x_domain:
        x_start.append(np.average(i))

    nm_rl = NelderMead(name='nm', x_start=np.array([2.2, 3.0]), step=1, progress=False,
                       rl_manual=False, manual_coordinates=[0,0])

    eff_action = 0
    for i in range(180):
        action = nm_rl.nelder_mead()
        # print("action", action)

        if action is not False:
            rewards = f(action, x_domain)
            eff_action += 1
        else:
            continue

        nm_rl.record(rewards, action, progress=False)
    print("History")
    print(*nm_rl.history, sep='\n')
    """
    # 1D
    empirical_points = [[-300], [-10], [400], [100]]
    x_domain = np.array([[-6, 6], [-6, 6]])

    nm_rl = NelderMead(name='nm', x_start=np.array([-400]), step=1.0, progress=False,
                       rl_manual=False, manual_coordinates=[0,0])
    spot_search = True

    for i in range(100):
        # print("nm_rl.res", nm_rl.res)
        if len(empirical_points) != 0:
            action = random.choice(empirical_points)
            empirical_points.remove(action)
            rewards = f1d(action, x_domain)
            nm_rl.record(rewards, action, progress=False)
        else:
            if spot_search:
                rewards_empirical = [row[2] for row in nm_rl.history]
                best_empirical = nm_rl.history[rewards_empirical.index(min(rewards_empirical))][0]
                nm_rl.restart(new_start=best_empirical)
                spot_search = False
            action = nm_rl.nelder_mead()
            # print("action", action)

            if action is not False:
                rewards = f1d(action, x_domain)
            else:
                continue

            nm_rl.record(rewards, action, progress=False)
    print("History")
    print(*nm_rl.history, sep='\n')

if __name__ == '__main__':
    main()
