"""
Todo:
    Complete three algorithms. Please follow the instructions for each algorithm. Good Luck :)
"""
import numpy as np

class EpislonGreedy(object):
    """
    Implementation of epislon-greedy algorithm.
    """
    def __init__(self, NumofBandits=10, epislon=0):
        """
        Initialize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        assert (0. <= epislon <= 1.0), "[ERROR] Epislon should be in range [0,1]"
        self._epislon = epislon
        self._nb = NumofBandits
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table. No need to return any result.
        """
        ################### Your code here #######################
        # raise NotImplementedError('[EpislonGreedy] update function NOT IMPLEMENTED')
        self._Q[action] = self._Q[action] + (immi_reward - self._Q[action]) / self._action_N[action]
        ##########################################################

    def act(self, t):
        """
        Step 3: Choose the action via greedy or explore. 
        Return: action selection
        """
        ################### Your code here #######################
        # raise NotImplementedError('[EpislonGreedy] act function NOT IMPLEMENTED')
        eps = np.random.uniform(0, 1)
        ret = 0
        if eps < self._epislon:
            ret = np.random.choice(range(self._nb))
        else:
            ret = np.argmax(self._Q)
        self._action_N[ret] += 1
        return ret
        ##########################################################

class UCB(object):
    """
    Implementation of upper confidence bound.
    """
    def __init__(self, NumofBandits=10, c=2):
        """
        Initailize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        self._nb = NumofBandits
        self._c = c
        self._Q = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table
        """
        ################### Your code here #######################
        # raise NotImplementedError('[UCB] update function NOT IMPLEMENTED')
        self._Q[action] = self._Q[action] + (immi_reward - self._Q[action]) / self._action_N[action]
        ##########################################################

    def act(self, t):
        """
        Step 3: use UCB action selection. We'll pull all arms once first!
        HINT: Check out p.27, equation 2.8
        """
        ################### Your code here #######################
        # raise NotImplementedError('[UCB] act function NOT IMPLEMENTED')
        ucb = np.zeros(self._nb, dtype=float)
        for i in range(self._nb):
            if self._action_N[i] == 0:
                self._action_N[i] += 1
                return i
            else:
                ucb[i] = self._Q[i] + self._c * np.sqrt(np.log(t) / self._action_N[i])
        ret = np.argmax(ucb)
        self._action_N[ret] += 1
        return ret
        ##########################################################

class Gradient(object):
    """
    Implementation of your gradient-based method
    """
    def __init__(self, NumofBandits=10, epislon=0.1):
        """
        Initailize the class.
        Step 1: Initialize your Q-table and counter for each action
        """
        self._nb = NumofBandits
        self._H = np.zeros(self._nb, dtype=float)
        self._action_N = np.zeros(self._nb, dtype=int)
        self._action_prob = np.zeros(self._nb, dtype=float)
        self._alpha = 0.4
        self._avg_reward = 0
        self._counter = 0

    def update(self, action, immi_reward):
        """
        Step 2: update your Q-table
        """
        ################### Your code here #######################
        # raise NotImplementedError('[gradient] update function NOT IMPLEMENTED')
        self._counter += 1
        self._avg_reward = self._avg_reward + (immi_reward - self._avg_reward) / self._counter
        for i in range(self._nb):
            if i == action:
                self._H[i] = self._H[i] + self._alpha * (immi_reward - self._avg_reward) * (1 - self._action_prob[i])
            else:
                self._H[i] = self._H[i] - self._alpha * (immi_reward - self._avg_reward) * self._action_prob[i]
        ##########################################################

    def act(self, t):
        """
        Step 3: select action with gradient-based method
        HINT: Check out p.28, eq 2.9 in your textbook
        """
        ################### Your code here #######################
        # raise NotImplementedError('[gradient] act function NOT IMPLEMENTED')
        softmax = [np.e ** self._H[i] for i in range(self._nb)]
        den = sum(softmax)
        for i in range(self._nb):
            self._action_prob[i] = softmax[i] / den
        ran = np.random.uniform(0, 1)
        tmp = 0
        for i in range(self._nb):
            tmp += self._action_prob[i]
            if ran < tmp:
                return i
        ##########################################################
