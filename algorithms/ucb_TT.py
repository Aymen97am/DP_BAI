#Author: Aymen Al Marjani

import numpy as np

from algorithms.toptwo import TopTwo, Kappa


class TTUCB(TopTwo):
    def __init__(self, config):
        """
        Initializes the vanilla TopTwo algorithm with K arms.
        No privacy. Leader arm is based on UCB index.
        """
        config["name"] = "UCBTT"
        super().__init__(config)
        
        self.eps = config["eps"]  #ignored in computations, only for comparison proper plots 
        # cumulated rewards
        self.rewards = np.zeros(self.K)
        
    
    def run(self, bandit):
        """
        Runs vanilla TopTwo algorithm on  bandit.

        Args:
            bandit (Bandit): instance of the Bandit class.
        """
        #intialization phase
        for arm in range(self.K):
            reward = bandit.pull(arm)
            self.update(arm, [reward])
        
        #main loop       
        while True : 
            arm = self.select_arm()
            reward = bandit.pull(arm)
            self.update(arm, [reward]) #update counts and values  
            #check if stopping rule is triggered
            if self.check_stopping():
                print(f"name : {self.name}, \n \
                    stopping time is {self.counts.sum()},\
                        best arm guess: {self.best}")
                self.save_logs()
                return self.counts.sum(), self.best
                
    
    def compute_ucb_leader(self):
        """ 
        Computes the leader based on a standard UCB index.
        """
        n = self.counts.sum()
        s = self.counts
        ucb_index = self.values + np.sqrt( np.log(n) / (2*s) )
        return np.argmax(ucb_index)

    def compute_challenger(self, leader):
        """ 
        Computes the challenger arm based on estimates from previous phase.
        
        Args:
            leader (int): index of the leader arm
        """
        # stats of leader arm
        n_leader = self.counts[leader] 
        v_leader = self.values[leader]
        
        # computing the exploration parameter  
        t = self.counts.sum() #number of samples by the algo
        kappa = Kappa(t)
        
        #compute challenger
        challenger = None 
        minCost = np.inf
        for j in range(self.K):
            n_j = self.counts[j]
            #transportation cost
            cost = (v_leader - self.values[j]+ kappa)\
                /np.sqrt(1/n_leader +  1/n_j)
            if j != leader and cost < minCost:
                challenger = j
                minCost = cost
        return challenger
    
    def check_stopping(self):
        """
        Checks whether the stopping condition is verified.
    
        """
        t = self.counts.sum()
        
        #stats of empirical best arm
        best = self.best
        n_best = self.counts[best] 
        v_best = self.values[best]
        
        threshold = emp_c(t, self.delta, self.K)
        
        for j in range(self.K):
            #transportation cost
            n_j = self.counts[j] #counts of the previous phase of j
            cost_j = 0.5*(v_best - self.values[j])**2 / (1/n_best +  1/n_j)
            if j != best and cost_j < threshold:
                return False  
        return True
    
    def update(self, arm, rewards):
        """
        Updates count and mean-reward of pulled arm, computes new best arm  

        Args:
            arm (int): The index of the arm to update.
            rewards (numpy.array): array of rewards received when playing the arm.
        """
        m = len(rewards)
        n = self.counts[arm]
        self.values[arm] = (n*self.values[arm] + m*np.sum(rewards))/ (n+m)
        self.counts[arm] += m
        
        #compute new best arm
        self.best = np.argmax(self.values)
    
        
            

def emp_c(t, delta, K):
    """
    Computes an empirical variant of the threshold involved in the stopping condition.

    Args: 
        n (int): counts of the current best arm
        m (int): counts of the challenger
        K (int): number of arms
        delta (float): risk parameter
        ph (int): index of the phase
        eps (float): privacy parameter

    """
    return np.log( (1+np.log(t)) / delta) + 0.5*np.log(K)


if __name__=="__main__":
    
    import os
    import sys
    sys.path.append(os.getcwd())
    
    from bandit import Bandit 
    
    K = 5
    mu = np.linspace(0, 1, 5)
    my_bandit = Bandit(mu)
    config = {"K": K, "beta": 0.5, "eps": 1.0, "delta": 0.1}
    
    vanilla_TT = TTUCB(config)
    vanilla_TT.run(my_bandit)
