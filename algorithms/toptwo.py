#Author: Aymen Al Marjani

import numpy as np

import os
import uuid


class TopTwo:
    def __init__(self, config):
        """
        Parent class for TopTwo algorithms with K arms.

        Args:
            conf (dict): dictionary with the following keys
            K (int): The number of arms.
            beta (float): probability of playing the empirical best arm.
            eps (float) (optional): privacy parameter.
            delta (float): risk parameter.
        """
        #hyperparameters
        self.K = config["K"]
        self.delta = config["delta"]
        self.beta = config["beta"]
        
        #name of the toptwo variant
        self.algo_name = "TopTwo_unspecified"
        if "name" in config.keys():
            self.algo_name = config["name"]   
        self.id = uuid.uuid4().hex[:8] # random id of the instance
        self.name =  self.algo_name+"_"+self.id  
        
        #name of the experiment
        self.exp_name = "FromTerminal"
        if "exp_name"in config.keys():
            self.exp_name = config["exp_name"]
                
        #total counts and mean-reward estimates
        self.counts = np.zeros(self.K)
        self.values = np.zeros(self.K)
        
        #useful to log info
        self.directory = "./experiments/" + self.exp_name +"/"+ self.algo_name +"/"
     
        
    def save_logs(self):
        #save logs of experiments in dedicated csv file
        tau  = self.counts.sum()
        a_star = np.argmax(self.values)

        if not os.path.isdir(self.directory):
            try:
                os.makedirs(self.directory)
            except FileExistsError:
                print(f"{self.name} file exists")
        
        filename = self.directory+self.name+".csv"
        data = np.array([self.K, self.delta, self.eps, tau, a_star])
        np.savetxt(filename, data, delimiter=",")
    
    
    def run(self, bandit):
        """
        Runs the TopTwo algorithm on  bandit.

        Args:
            bandit (Bandit): instance of the Bandit class.
        """ 
        pass
                
    
    def select_arm(self):
        """
        Selects which arm to play next using the sampling rule of TopTwo.

        Returns:
            int: The index of the arm to play.
        """
            
        # compute leader arm & challenger from previous phase
        leader = self.compute_ucb_leader()
        challenger  = self.compute_challenger(leader)
        
        if np.random.uniform() <= self.beta:
            return leader
        else:
            return challenger

    def compute_ucb_leader(self):
        """ 
        Computes the leader arm.
        """
        pass
    
    def compute_challenger(self, leader):
        """ 
        Computes the challenger arm .
        
        Args:
            leader (int): index of the leader arm
        """
        pass 
    
    def check_stopping(self):
        """
        Checks whether the stopping condition is met. 
        """
        pass
        
        
def Kappa(t):
    """
    Computes the exploration bonus used in the challenger formula.
    
    Args: 
        t (int): total number of samples used so far.
    """
    alpha = 0.5
    return np.log(1+t)** (-alpha/2)