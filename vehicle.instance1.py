
from VehicleModel4Wh import Vehicle
import numpy as np

class FS4Vehicle:
    #values are not compelely accurate to FS4 just yet
    def __init__(self):
        self.slr = 300
        self.dlr = 310
        self.initial_camber = 2
        self.toe_in = 0.4
        self.tw = 1500
        self.wb = 2500
        self.wr_front = 30.647
        self.wr_rear = 35.025
        self.tire_stiffness_front = 220
        self.tire_stiffness_rear = 230
        self.pinion = 5
        self.tirep = 36
        self.dila = -38
        self.linkage_effort = 1.7
        self.linkage_kpm = 9.0



     

