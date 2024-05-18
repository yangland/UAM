# Naive MCTS with arm expansion
# Naive Sampling Monte Carlo Tree Search
# Arm expansion for discrete continuous variable
# To solve the Combinatorial Multi-Armed Bandit problem
import random
from collections import OrderedDict
from collections import defaultdict
import warnings
import numpy as np
import math
import operator
import ast
import logging
logger = logging.getLogger("logger")

class Search_Variable:
    def __init__(self, name, _type, domain):
        self.name = name
        self._type = _type
        self.domain = domain

class Action:
    def __init__(self, name, search_variables, values):
        self.name = name
        self.search_variables = search_variables
        self.values = values
        self.details = {}
        for i in range(len(self.search_variables)):
            self.details[self.search_variables[i].name] = values[i]
class Naive_Mcts_Simp:
    def __init__(self, name, malicious_mode, num_classes, malicious_group_list, goal, pi_0_eps, init_search_len):
        self.init_search = None
        self.name = name
        self.malicious_mode = malicious_mode
        self.num_classes = num_classes
        self.malicious_group_list = malicious_group_list
        self.g_main = goal
        self.pi_0_eps = pi_0_eps
        self.mode = 'initial_explore'
        self.search_space = None

        # searching variables, domains and type
        # Old action arm based parameters and search space
        alpha1 = Search_Variable(name="alpha1", _type='discrete_continuous', domain=[len(self.g_main), 1])
        beta1 = Search_Variable(name="beta1", _type='discrete_continuous', domain=[0, self.num_classes])
        beta2 = Search_Variable(name="beta2", _type='discrete_continuous', domain=[0, self.num_classes])
        respect_server = Search_Variable(name="respect_server", _type='discrete', domain=[True, False])
        mali_group_agg = Search_Variable(name="mali_group_agg", _type='discrete', domain=[True, False])
        reverse_grad = Search_Variable(name="reverse_grad", _type='discrete', domain=[True, False])

        self.search_space = [alpha1, beta1, beta2, respect_server, mali_group_agg, reverse_grad]

        self.init_search_len = init_search_len
        self.global_orders = defaultdict(lambda: dict(avg_reward=0, count=0, action_obj=None))

        # create a multi-bandit action space
        action_values_1 = [len(self.g_main), 0, 0, False, True, False]
        action_values_2 = [0, self.num_classes, self.num_classes, True, False, False]
        action_values_3 = [len(self.g_main), 0, 0, True, True, True]
        action_values_4 = [len(self.g_main), 0, 0, True, True, False]
        action_values_5 = [0, self.num_classes, self.num_classes, True, False, False]
        action_1 = Action(name="untarget_1", search_variables=self.search_space, values=action_values_1)
        action_2 = Action(name="untarget_2", search_variables=self.search_space, values=action_values_2)
        action_3 = Action(name="reverse_grad", search_variables=self.search_space, values=action_values_3)
        action_4 = Action(name="closer_untarget_1", search_variables=self.search_space, values=action_values_4)
        action_5 = Action(name="closer_untarget_2", search_variables=self.search_space, values=action_values_5)
        self.action_space = [action_1, action_2, action_3]

        # manual set the self.init_search
        # two_orders = [[action_1, action_2, action_1, action_2, action_1, action_2, action_1, action_2],
        #               [action_2, action_1, action_2, action_1, action_2, action_1, action_2, action_1]]
        #
        # self.init_search = random.choice(two_orders)
        # self.init_search = [action_2, action_2, action_2, action_2, action_2, action_2, action_2, action_2]
        # self.init_search = [action_1, action_1, action_1, action_1, action_1, action_1, action_1, action_1]

        # auto set the self.inti_search
        action_space_shuffle = list(self.action_space)
        random.shuffle(action_space_shuffle)
        self.init_search = (action_space_shuffle * math.ceil(self.init_search_len/len(self.action_space)))[:self.init_search_len]
        # self.init_search = [action_1, action_1, action_1, action_1, action_1, action_1, action_1, action_1]
        # self.init_search = [action_3, action_3, action_3, action_3, action_3, action_3, action_3, action_3]
        # self.init_search = [action_4, action_4, action_4, action_4, action_4, action_4, action_4, action_4, action_4, action_4]
        # self.init_search = [action_5, action_5, action_5, action_5, action_5, action_5, action_5, action_5, action_5, action_5]
        self.init_search = [action_2, action_2, action_2, action_2, action_2, action_2, action_2, action_2]

        # memory
        # global history #tuple of vi, reward [(), *]
        # only global history
        self.global_history = []

        self.current_action = None
        # [action, reward]
        self.current_optimum = []
        self.dominate_strategy = None

    def record(self, mali_accuracy_record, mali_asr_record):
        # reward need to be the diff between the current KPI and last round's KPI (To-Do)
        kpi = None
        # reward object based on malicious_mode
        if self.malicious_mode == 'label_flipping':
            logger.info("mali_asr_record", mali_asr_record)
            last_items = [item[-1] for item in list(mali_asr_record.values())]
            kpi = sum(last_items) / len(last_items)
        elif self.malicious_mode == 'untargeted':
            kpi = 1 - mali_accuracy_record
        else:
            logger.info("Unsupported self.malicious_mode")

        # Global
        # record for the global history
        # the first a few iterations the accuracy improves too fast
        if len(self.global_history) in [0]:
            reward = 0
        else:
            # use the moving speed of kpi as reward
            reward = kpi - self.global_history[-1][1]
        if self.current_action is None:
            self.global_history.append([None, kpi, reward])
        else:
            self.global_history.append([self.current_action.name, kpi, reward])

        logger.info("History for all FL rounds (self.global history)")
        for i in range(len(self.global_history)):
            logger.info(f"{i}, {self.global_history[i]}")
        # save combinations been played, times and the avg reward



        if self.current_action is not None:
            super_arm_name = self.current_action.name

            prev_avg = self.global_orders[super_arm_name]['avg_reward']
            prev_count = self.global_orders[super_arm_name]['count']

            curr_avg = (prev_avg * prev_count + reward) / (prev_count + 1)
            curr_count = prev_count + 1

            self.global_orders[super_arm_name]['avg_reward'] = curr_avg
            self.global_orders[super_arm_name]['count'] = curr_count
            self.global_orders[super_arm_name]['action_obj'] = self.current_action

            # get the current optimum
            # if self.current_optimum is None or curr_avg > self.current_optimum[1]:
            #     self.current_optimum = [self.current_action, curr_avg]
            #
            # print("self.current_optimum", [self.current_optimum[0].name, self.current_optimum[1]])
            # for key, value in self.global_orders.items():
            #     if len(self.current_optimum)==0 or value['avg_reward'] > self.current_optimum[1]:
            #         self.current_optimum = [key, value['avg_reward'], self.current_action]

            if len(self.current_optimum)==0:
                self.current_optimum = [self.current_action.name, curr_avg, self.current_action]
            else:
                # for key, value in self.global_orders.items():
                #     if value['avg_reward'] > self.current_optimum[1]:
                #         self.current_optimum = [key, value['avg_reward'], self.current_action]
                all_reward_vals = dict()
                for key, value in self.global_orders.items():
                    all_reward_vals[key] = value['avg_reward']
                optimum_name = max(all_reward_vals, key=all_reward_vals.get)
                self.current_optimum = [optimum_name, self.global_orders[optimum_name]['avg_reward'],
                                        self.global_orders[optimum_name]['action_obj']]
                # logger.info(f"record self.current_optimum, {self.current_optimum}")


        # print("self.global_orders", dict(self.global_orders))
        logger.info("Action, its saved average reward, count (self.global_orders)")
        for key, value in self.global_orders.items():
            logger.info(f"{key}, {value['avg_reward']}, {value['count']}")
        return None

    def mode_update(self):
        if len(self.global_history) > self.init_search_len:
            self.mode = 'exploit_explore'


    def mcts(self):
        # check the current mode
        self.mode_update()

        # do initial search
        if self.mode == 'initial_explore':
            logger.info("RL mode: initial_explore")
            action = dict()
            # take the action in the self.init_search for one searching variable
            est_action = self.init_search[len(self.global_history)-1]
            logger.info(f"attempt action:{est_action.name}")

            # current_action = OrderedDict(action.details)
            self.current_action = est_action
            return est_action
        elif self.mode == 'exploit_explore':
            logger.info("RL mode: exploit_explore")
            # pi0 epsilon greedy
            p_0 = np.random.random()
            if p_0 < self.pi_0_eps:
                logger.info(f"p_0 < pi_0_eps, explore {p_0}, {self.pi_0_eps}")
                explore_action = random.choice(self.action_space)
                # current_action = OrderedDict(explore_action.details)
                self.current_action = explore_action
                return explore_action
            else:
                # exploit using the global pure greedy policy pi_g
                logger.info(f"p_0 >= pi_0_eps, exploit, {p_0}, {self.pi_0_eps}")
                logger.info(f"current_optimum: {self.current_optimum[:-1]}")
                # print("current optimum name", self.current_optimum[-1].name)
                self.current_action = self.current_optimum[-1]
                return self.current_optimum[-1]
                # return dict(ast.literal_eval(self.current_optimum[0]))

        else:
            warnings.warn('Unknown mode name for Naive_Mcts Class!')

    def get_dominate_strategy(self):
        for key, value in self.global_orders.items():
            if self.dominate_strategy is None or value['count'] > self.global_orders[self.dominate_strategy]['count']:
                self.dominate_strategy = key
        return self.dominate_strategy