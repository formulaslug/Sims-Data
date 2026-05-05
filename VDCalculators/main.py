#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:21:41 2022

@author: simon
"""
from kinsolve import *

#COORDINATES FOR REAR RIGHT QUARTER OF CAR SUSPENSION
def main():

    """ Suspension Points """
    # In form of Point([x,y,z])
    # Wheel_Center
    # Y point of wc should be track width / 2
    # Wheel_Center
    wc = Point([973.138, 593, 147.8])
    
    # Lower Wishbone
    lfi = Point([813.09, 240.6, 97.53]) # Lower_Fore_Inner
    lai = Point([908.93, 224.22, 101.62]) # Lower_Aft_Inner
    lo  = Point([973.119, 617.362, 70.652]) # Lower_Upright_Point
    # Upper Wishbone
    ufi = Point([792.28, 264.39, 220.42]) # Upper_Fore_Inner
    uai = Point([909.24, 247.31, 238.90]) # Upper_Aft_Inner
    uo  = Point([973.032, 612.902, 239.431]) # Upper_Upright_Point
    # Tie Rod or Steering Rod
    tri = Point([798.115, 236.174, 140.185]) # Tie_Rod_Inner
    tro = Point([913.402, 626.763, 121.687]) # Tie_Rod_Outer
    
    unit = "mm"  # used in graph axis labels, not used in code (yet...)

    # Pushrod or Pullrod Points
    # The P-rod inner point is the outboard (usually) point of the rocker/bellcrank
    pro = Point([893.70, 303.83, 66.97]) #we might need to swrap pri/pro (outer and inner confusion)
    pri = Point([945.282, 528.916, 221.115])
    
    # Rocker Center of Rotation
    rkr = Point([869.388, 218.649, 74.741])
    
    # Shock Pickup Points (upper, lower)
    # The shock upper point is the inner (usually) point of the rocker/bellcrank
    sku = Point([890.73, 312.99, 149.24])
    skl = Point([873.3, 291.69, 308.65])

    """ Suspension Setup """
    # Full jounce and rebound mark the bounds for the solver
    # if they are too large, and cannot be achieved with your linkage system
    # the code will not throw an error but will either not finish solving or give erroneous results
    full_jounce = 40
    full_rebound = -17
    
    # toe, camber and caster are used for static offsets on the graphs
    # these will not affect the solver
    toe = 0
    camber = 0
    caster = 0
    
    kin = KinSolve(
        wheel_center=wc,
        lower_wishbone=(lfi, lai, lo),
        upper_wishbone=(ufi, uai, uo),
        tie_rod=(tri, tro),
        
        p_rod=(pri, pro),
        rocker=rkr,
        shock=(skl, sku),

        full_jounce=full_jounce,
        full_rebound=full_rebound,

        unit=unit,
    )
    
    """ Solver Parameters """
    # number of steps in each direction, so a value of 10 will yield 20 datapoints
    # algorithm runs fast enough that its fine to use 1000+, but 100 is just as accurate
    # and it will result in a comprehensible amount of data
    num_steps = 10
    
    kin.solve(
        steps=num_steps,
        offset_toe=toe,
        offset_camber=camber,
        offset_caster=caster
    )
    
    """Link Force Solver Inputs"""
    # NOT VALIDATED YET
    Fx = 0 # N
    Fy = 0 # N
    Fz = 100 # N
    kin.linkforce(Fx, Fy, Fz)
    
    
    """ Plot """
    kin.plot(
        suspension=True,  # Visualize the corner
        
        bump_steer=False,  # Bump Steer vs vertical travel
        bump_steer_in_deg=False,  # Sets y-axis of bump steer plot to roll angle in deg

        camber_gain=False,  # Camber Gain vs vertical travel
        camber_gain_in_deg=False,  # Sets y-axis of camber gain plot to roll angle in deg

        caster_gain=False,  # Caster gain plot
        caster_gain_in_deg=False,  # Sets y-axis of caster gain plot to roll angle in deg

        scrub_gain = False, # Scrub change plot
        scrub_gain_in_deg = False,    # Sets y-axis of scrub gain plot to roll angle in deg

        roll_center_in_roll=True,  # Path of roll center as the car rolls
        
        motion_ratio=False, # Motion Ratio vs vertical travel
        motion_ratio_in_deg=False # Sets y-axis of motion ratio plot to roll angle in deg
    )


if __name__ == "__main__":
    main()
