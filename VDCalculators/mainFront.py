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
    wc = Point([959.168,	607.085, 154.086])
    # Lower Wishbone
    lfi = Point([813.904,	    241.972,	97.37]) # Lower_Fore_Inner
    lai = Point([909.274,	    225.761,	101.434]) # Lower_Aft_Inner
    lo  = Point([973.138,	    609.294,	69.693]) # Lower_Upright_Point
    # Upper Wishbone
    ufi = Point([793.236,	    265.679,	220.497]) # Upper_Fore_Inner
    uai = Point([909.615,	248.855,	238.879]) # Upper_Aft_Inner
    uo  = Point([973.138,	    604.875,	238.479]) # Upper_Upright_Point
    # Tie Rod or Steering Rod
    tri = Point([795.613, 248.764, 106.0]) # Tie_Rod_Inner
    tro = Point([909.638, 608.962, 82.388]) # Tie_Rod_Outer
    
    unit = "mm"  # used in graph axis labels, not used in code (yet...)

    # Pushrod or Pullrod Points
    # The P-rod inner point is the outboard (usually) point of the rocker/bellcrank
    pri = Point([ 949.673, 540.450  ,  237.037 ]) #we might need to swrap pri/pro (outer and inner confusion)
    pro = Point([ 860.930, 224.522,  60.001])
    
    # Rocker Center of Rotation
    rkr = Point([ 857.363  , 211.029     ,  70.890])
    
    # Shock Pickup Points (upper, lower)
    # The shock upper point is the inner (usually) point of the rocker/bellcrank
    sku = Point([ 869.757     , 255.461     ,  88.567])
    skl = Point([ 868.629     , 243.776     ,  260.166])

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
        bump_steer_in_deg=False,  # Sets y-axis of bump steer plot to roll angle in deg

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
