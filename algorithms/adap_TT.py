#Author: Aymen Al Marjani

import numpy as np

from algorithms.toptwo import TopTwo, Kappa

class AdaPTopTwo(TopTwo):
    def __init__(self, config):
        """
        Initializes the adaptive private TopTwo algorithm with K arms.

        Args:
            eps (float): privacy parameter.
            
        """
        config["name"] = "AdaPTT"
        super().__init__(config)
        
        self.eps = config["eps"]
        
        #phase statistics (specific to this private variant of top two)
        self.phase = np.zeros(self.K)
        self.ph_counts = np.zeros(self.K)
        self.ph_rewards = np.zeros(self.K)
        self.ph_best = None
        self.last_ph_counts = np.zeros(self.K)
        
           
    def run(self, bandit):
        """
        Runs adaptive private TopTwo algorithm on  bandit.

        Args:
            bandit (Bandit): instance of the Bandit class.
        """ 
        # Play each arm once to initialize
        for arm in range(self.K):
            reward = bandit.pull(arm)
            self.ph_counts[arm] += 1
            self.ph_rewards[arm] += reward
            self.update(arm)
        
        #main loop       
        while True : 
            arm = self.select_arm()
            reward = bandit.pull(arm)
            self.ph_counts[arm] += 1
            self.ph_rewards[arm] += reward
            if self.doubled_counts(arm):
                self.update(arm)
                #check if stopping rule is triggered
                if self.check_stopping():
                    print(f"name : {self.name}, \n \
                        stopping time is {self.counts.sum()},\
                        best arm guess: {self.ph_best}")
                    self.save_logs()
                    return self.counts.sum(), np.argmax(self.values)
                


    def compute_ucb_leader(self):
        """ 
        Computes the leader based on a privatized UCB index.
        """
        n = self.counts.sum()
        s = self.last_ph_counts
        ucb_index = self.values  +  np.log(n) / (self.eps*s)\
            + np.sqrt( np.log(n) / (2*s) )
        return np.argmax(ucb_index)
    
    def compute_challenger(self, leader):
        """ 
        Computes the challenger arm based on estimates from previous phase.
        
        Args:
            leader (int): index of the leader arm
        """
        t = self.counts.sum()
        #stats of best arm
        n_leader = self.counts[leader] + self.ph_counts[leader]
        v_leader = self.values[leader]

        challenger = None 
        minCost = np.inf
        for j in range(self.K):
            n_j = self.counts[j] + self.ph_counts[j]
            #transportation cost
            cost = (v_leader - self.values[j]+ Kappa(t))\
                /np.sqrt(1/n_leader +  1/n_j)
            if j != leader and cost < minCost:
                challenger = j
                minCost = cost
        return challenger
    
    def check_stopping(self):
        """
        Checks whether the stopping condition is met.

        Args: 
            ph_counts (numpy.array): counts of the current phase 
    
        """
        #stats of best arm
        best = self.ph_best
        n_best = self.last_ph_counts[best]
        v_best = self.values[best]
        
        #check stopping rule
        for j in range(self.K):
            #transportation cost
            n_j = self.last_ph_counts[j]
            cost_j = 0.5*(v_best - self.values[j])**2 / (1/n_best +  1/n_j)
            threshold_j = emp_c(n_best, n_j, self.delta, self.eps)
            if j != best and  cost_j < threshold_j:
                return False
        return True
    
    def update(self, arm):
        """
        Updates phase number and total counts then constructs an eps-DP estimator of the mean rewards of every arm. 

        Args:
            arm (int): The index of the arm to update.
            reward (float): The reward received for playing the arm.
        """
        self.phase[arm] += 1
        self.counts[arm] += self.ph_counts[arm]
        self.last_ph_counts[arm] = self.ph_counts[arm]
        
        # forget previous rewards and construct new eps-DP estimate
        noise = np.random.laplace(scale = 1 / (self.eps* self.ph_counts[arm]))
        self.values[arm] = self.ph_rewards[arm]/self.ph_counts[arm] + noise
        
        #compute new best for next phase
        if (self.last_ph_counts == 0).any():
            self.ph_best = np.argmax(self.values)
        else:
            self.ph_best = self.compute_ucb_leader()
 
        #reset phase stats
        self.ph_counts[arm] = 0
        self.ph_rewards[arm] = 0
        
    def doubled_counts(self, arm):
        """
        Checks whether the counts of arm have doubled.

        Args: 
            arm(int) : index of the arm
    
        """
        return (self.ph_counts[arm] == self.counts[arm]) 
        
        

def emp_c(n, m, delta, eps):
    """
    Computes an empirical variant of the private threshold involved in the stopping condition.

    Args: 
        n (int): counts of the current best arm
        m (int): counts of the challenger
        K (int): number of arms
        delta (float): risk parameter
        ph (int): index of the phase
        eps (float): privacy parameter

    """
    term_0 = np.log(1/delta) + np.log(1 + np.log(n)) + np.log(1 + np.log(m))
    dp_term = (1/n + 1/m) * (1/eps**2) * np.log(1/delta)**2
    return term_0 + dp_term


if __name__=="__main__":
    import os
    import sys
    sys.path.append(os.getcwd())
    
    from bandit import Bandit 
    
    K = 5
    mu = np.linspace(0, 1, 5)
    config = {"K": K, "beta": 0.5, "eps": 1.0, "delta": 0.1}
    
    my_bandit = Bandit(mu)
    adap_top_two = AdaPTopTwo(config)
    adap_top_two.run(my_bandit)
