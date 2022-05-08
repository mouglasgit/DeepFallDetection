# -*- coding: utf-8 -*-

import numpy as np
from numpy import dot
from scipy.linalg import inv, block_diag


class KalmanFilterTrack(): 

    def __init__(self):
        
        self.start_variables()
        self.displacement = np.array([[1, self.time, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, self.time, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, self.time, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, self.time],
                           [0, 0, 0, 0, 0, 0, 0, 1]])
        
        self.measu_mtx = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0]])
        
        
        self.V = 100.0
        self.W = np.diag(self.V * np.ones(8))
        
        self.init_cov = np.array([[self.time ** 4 / 2., self.time ** 3 / 2.], [self.time ** 3 / 2., self.time ** 2]])
        self.covariance_matrix = block_diag(self.init_cov, self.init_cov, self.init_cov, self.init_cov)
        
        self.cov_rate = 1.0 / 16.0
        self.diag = np.diag(self.cov_rate * np.array([self.V, self.V, self.V, self.V]))
    
    def start_variables(self):
        self.id = 0   
        self.time = 1. 
        self.coordinates = []
        self.current_state = []  
        self.pattern_counter = 0
        self.no_pattern_counter = 0 
         
    def inference(self): 
        state = dot(self.displacement, self.current_state)
        self.W = dot(self.displacement, self.W).dot(self.displacement.T) + self.covariance_matrix
        self.current_state = state.astype(int)
    
    def update_state(self): 
        self.diag = np.diag(self.cov_rate * np.array([self.V, self.V, self.V, self.V]))
        
    def gain_update_measurement(self, z): 
        state = self.current_state
        state = dot(self.displacement, state)
        self.W = dot(self.displacement, self.W).dot(self.displacement.T) + self.covariance_matrix
        self.update_measurement(state, z)
    
    def update_measurement(self, state, z):
        S = dot(self.measu_mtx, self.W).dot(self.measu_mtx.T) + self.diag
        kalman_gain = dot(self.W, self.measu_mtx.T).dot(inv(S))  
        residual = z - dot(self.measu_mtx, state)  
        state += dot(kalman_gain, residual)
        self.W = self.W - dot(kalman_gain, self.measu_mtx).dot(self.W)
        self.current_state = state.astype(int)
                                   
