# Naive MCTS with arm expansion
# Naive Sampling Monte Carlo Tree Search
# Arm expansion for discrete continuous variable
# To solve the Combinatorial Multi-Armed Bandit problem
import random
from collections import OrderedDict
from collections import defaultdict
import warnings
import numpy as np
import operator
import ast

class Search_Variable:
    def __init__(self, name, _type, domain):
        self.name = name
        self._type = _type
        self.domain = domain

class Naive_Mcts:
    def __init__(self, name, malicious_mode, num_classes, malicious_group_list, goal, pi_0_eps, pi_l_eps,
                 pi_0_eps_decay, pi_0_eps_floor, pi_l_eps_decay, pi_l_eps_floor):
        self.init_search = None
        self.name = name
        self.malicious_mode = malicious_mode
        self.num_classes = num_classes
        self.malicious_group_list = malicious_group_list
        self.g_main = goal
        self.pi_0_eps = pi_0_eps
        self.pi_l_eps = pi_l_eps
        self.pi_0_eps_decay = pi_0_eps_decay
        self.pi_0_eps_floor = pi_0_eps_floor
        self.pi_l_eps_decay = pi_l_eps_decay
        self.pi_l_eps_floor = pi_l_eps_floor
        self.mode = 'initial_explore'

        # searching variables, domains and type
        alpha1 = Search_Variable(name="alpha1", _type='discrete_continuous', domain=[1, len(self.g_main)])
        beta1 = Search_Variable(name="beta1", _type='discrete_continuous', domain=[0, self.num_classes])
        beta2 = Search_Variable(name="beta2", _type='discrete_continuous', domain=[0, self.num_classes])
        respect_server = Search_Variable(name="respect_server", _type='discrete', domain=[False, True])
        mali_group_agg = Search_Variable(name="mali_group_agg", _type='discrete', domain=[False, True])
        self.search_space = [alpha1, beta1, beta2, respect_server, mali_group_agg]
        self.init_search = []
        self.disc_cont_vars = []

        # random.shuffle(self.search_space)

        for i in self.search_space:
            for j in i.domain:
                # in initial search the different value of one arm search next to each other
                self.init_search.append((i.name, j))
            if i._type == 'discrete_continuous':
                self.disc_cont_vars.append(i)
        # random.shuffle(self.init_search)

        # memory
        # global history #tuple of vi, reward [(), *]
        self.global_history = []
        self.global_orders = {}
        # create a nested dict
        self.local_history = {}
        for i in self.search_space:
            self.local_history[i.name] = defaultdict(lambda: dict(marg_avg_reward=0, count=0))

        self.current_action = None
        # [action, reward]
        self.current_optimum = None

    def record(self, mali_accuracy_record, mali_asr_record):
        # reward need to be the diff between the current KPI and last round's KPI (To-Do)
        kpi = None
        # reward object based on malicious_mode
        if self.malicious_mode == 'label_flipping':
            print("mali_asr_record", mali_asr_record)
            last_items = [item[-1] for item in list(mali_asr_record.values())]
            kpi = sum(last_items) / len(last_items)
        elif self.malicious_mode == 'untargeted':
            kpi = 1 - mali_accuracy_record

        # Global
        # record for the global history
        # the first a few iterations the accuracy improves too fast
        if len(self.global_history) in [0, 1, 2]:
            reward = 0
        else:
            reward = kpi - self.global_history[-1][1]
        self.global_history.append([self.current_action, kpi, reward])

        print("self.global history")
        for item in self.global_history:
            print(item)
        # save combinations been played, times and the avg reward
        if self.current_action is not None:
            super_arm_name = str(list(self.current_action.items()))
            if super_arm_name not in self.global_orders:
                self.global_orders[super_arm_name] = defaultdict(lambda: dict(avg_reward=0, count=0))
                self.global_orders[super_arm_name]['avg_reward'] = reward
                self.global_orders[super_arm_name]['count'] = 1
            else:
                prev_avg = self.global_orders[super_arm_name]['avg_reward']
                prev_count = self.global_orders[super_arm_name]['count']

                curr_avg = (prev_avg * prev_count + reward) / (prev_count + 1)
                curr_count = prev_count + 1
                self.global_orders[super_arm_name]['marg_avg_reward'] = curr_avg
                self.global_orders[super_arm_name]['count'] = curr_count

            if self.current_optimum is None or reward > self.current_optimum[1]:
                self.current_optimum = [super_arm_name, reward]

            # Local
            # record for the value options in each arm
            for arm, value in self.current_action.items():
                # print("self.local_history[arm][value]['marg_avg_reward']", self.local_history[arm][value]['marg_avg_reward'])
                # print("self.local_history[arm][value]['count']", self.local_history[arm][value]['count'])
                self.local_history[arm][value]['marg_avg_reward'] = (self.local_history[arm][value]['marg_avg_reward'] *
                    self.local_history[arm][value]['count'] + reward) / (self.local_history[arm][value]['count'] + 1)
                self.local_history[arm][value]['count'] += 1
            print("self.local_history")
            for key, value in self.local_history.items():
                print(key, ' : ', dict(value))
        return None

    def mode_update(self):
        if len(self.global_history) >= len(self.init_search):
            self.mode = 'exploit_explore'

        # if len(self.global_history) >= 2: # for debugging
        #     self.mode = 'exploit_explore'

    def mcts(self):
        # check the current mode
        self.mode_update()

        # do initial search with extreme values of the discrete_continuous value and the discrete values
        if self.mode == 'initial_explore':
            print("RL mode: initial_explore")
            action = dict()
            # take the action in the self.init_search for one searching variable
            attempt = self.init_search[len(self.global_history)]
            print("attempt:", attempt)
            action[attempt[0]] = attempt[1]

            # other variables take the random actions
            for i in self.search_space:
                if i.name != attempt[0]:
                    action[i.name] = random.choice(i.domain)
            current_action = OrderedDict(sorted(action.items()))
            self.current_action = current_action
            return current_action
        elif self.mode == 'exploit_explore':
            print("RL mode: exploit_explore")
            # pi0 epsilon greedy
            p_0 = np.random.random()
            if p_0 < self.pi_0_eps:
                print("p_0 < pi_0_eps, explore locally", p_0, self.pi_0_eps)
                action = {}
                # explore using the local policy pi_l
                p_l = np.random.random()

                if p_l < self.pi_l_eps:
                    print("p_l < pl_0_eps, explore wth arm expansion", p_l, self.pi_l_eps)
                    # expansion of a random discrete continuous arm
                    arm_exp = self.disc_cont_vars[random.randint(0, len(self.disc_cont_vars)-1)]
                    # check the current arm's values - reward, expand based on 1. choices 2. the best location
                    print("arm to expand:", arm_exp.name)
                    current_arm_vars = list(self.local_history[arm_exp.name].keys())

                    # check if there is still room for expansion (To-Do)
                    # 1. find the best var
                    # 2. find the neighbors of the best var, (a, b) or only (a), or (b)
                    # 3. expand towards the larger one from (a, b) if there is still space to expand

                    # if len(current_arm_vars) >= 3 and check_arithmetic_progression(current_arm_vars) is False:
                    if len(current_arm_vars) >= 3:
                        # To-Do where to expand beyond 3
                        # find the max value, expand
                        print("len(current arm var) >= 3, no expanding")
                        pass
                    else:
                        # add new var using the avg from the current two extreme values
                        middle_var = sum(list(self.local_history[arm_exp.name].keys()))/2
                        available_vars = list(range(arm_exp.domain[0], arm_exp.domain[1]+1))
                        new_var = min(available_vars, key=lambda x:abs(x-middle_var))
                        self.local_history[arm_exp.name][new_var]['marg_avg_reward'] = 0
                        self.local_history[arm_exp.name][new_var]['count'] = 0
                        # print(arm_exp.name + "arm value expanded to " + str(new_var))
                # self.local_history did not record new arm (To-Do)
                # apply UCB for each arm
                for arm, arm_dict in self.local_history.items():
                    print("arm:", arm)
                    print("arm_dict:", dict(arm_dict))
                    ucb = {}
                    # calculate UCB index
                    for var in arm_dict.keys():
                        sample_mean = arm_dict[var]['marg_avg_reward']
                        N = len(self.global_history)
                        n = arm_dict[var]['count']
                        if n != 0:
                            c = np.sqrt(2 * N / n)
                            ucb[var] = sample_mean + c

                            # value that max in the UCB
                            action[arm] = max(ucb.items(), key=operator.itemgetter(1))[0]
                        else:
                            print("n is 0")
                            action[arm] = var
                    print("ucb for arm " + arm + str(ucb))
                self.current_action = action
                return action
            else:
                # exploit using the global pure greedy policy pi_g
                print("p_0 >= pi_0_eps, exploit", p_0, self.pi_0_eps)
                print("current_optimum", self.current_optimum)
                return dict(ast.literal_eval(self.current_optimum[0]))

        # adjust the eps for pi_0 and pi_l
        decay_iter = len(self.global_history) - len(self.init_search)
        self.pi_0_eps *= (1. / (1. + self.pi_0_eps_decay * decay_iter)) # if iteration is in decay schedule
        if self.pi_0_eps < self.pi_0_eps_floor:
            self.pi_0_eps = self.pi_0_eps_floor

        self.pi_l_eps *= (1. / (1. + self.pi_l_eps_decay * decay_iter))  # if iteration is in decay schedule
        if self.pi_l_eps < self.pi_l_eps_floor:
            self.pi_l_eps = self.pi_l_eps_floor

        else:
            warnings.warn('Unknown mode name for Naive_Mcts Class!')

def check_arithmetic_progression(arr):
    diff = arr[1] - arr[0]
    n = len(arr)

    for i in range(2, n):
        if arr[i] - arr[i - 1] != diff:
            return False
    return True