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
    wc = Point([0,	564.422, 153.009])
    # Lower Wishbone
    lfi = Point([159.437,	    238.840,	97]) # Lower_Fore_Inner
    lai = Point([63.89,	    222.56,	101.38]) # Lower_Aft_Inner
    lo  = Point([0,	    564.422,	68.732]) # Lower_Upright_Point
    # Upper Wishbone
    ufi = Point([180.268,	    262.589,	220.38]) # Upper_Fore_Inner
    uai = Point([63.565,	246.330,	242.604]) # Upper_Aft_Inner
    uo  = Point([0,	    564.422,	237.287]) # Upper_Upright_Point
    # Tie Rod or Steering Rod
    tri = Point([181.925, 228.453, 159.254]) # Tie_Rod_Inner
    tro = Point([61.604, 564.422, 187.198]) # Tie_Rod_Outer
    
    unit = "mm"  # used in graph axis labels, not used in code (yet...)

    # Pushrod or Pullrod Points
    # The P-rod inner point is the outboard (usually) point of the rocker/bellcrank
    pri = Point([ 62.554, 252.117  ,  49.548 ])
    pro = Point([ 24.771, 500.482,  236.16])
    
    # Rocker Center of Rotation
    rkr = Point([ 94.167  , 194.523     ,  57.086])
    
    # Shock Pickup Points (upper, lower)
    # The shock upper point is the inner (usually) point of the rocker/bellcrank
    sku = Point([ 84.83     , 283.605     ,  133.389])
    skl = Point([ 157.257     , 278.920     ,  289.326])

    """ Suspension Setup """
    # Full jounce and rebound mark the bounds for the solver
    # if they are too large, and cannot be achieved with your linkage system
    # the code will not throw an error but will either not finish solving or give erroneous results
    full_jounce = 25
    full_rebound = -25
    
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
    num_steps = 100
    
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
        
        bump_steer=True,  # Bump Steer vs vertical travel
        bump_steer_in_deg=True,  # Sets y-axis of bump steer plot to roll angle in deg

        camber_gain=False,  # Camber Gain vs vertical travel
        camber_gain_in_deg=False,  # Sets y-axis of camber gain plot to roll angle in deg

        caster_gain=False,  # Caster gain plot
        caster_gain_in_deg=False,  # Sets y-axis of caster gain plot to roll angle in deg

        scrub_gain = False, # Scrub change plot
        scrub_gain_in_deg = False,    # Sets y-axis of scrub gain plot to roll angle in deg

        roll_center_in_roll=False,  # Path of roll center as the car rolls
        
        motion_ratio=False, # Motion Ratio vs vertical travel
        motion_ratio_in_deg=False # Sets y-axis of motion ratio plot to roll angle in deg
    )


if __name__ == "__main__":
    main()
