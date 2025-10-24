# Wheel Speed Calibration

## Active Members

1. Noah Santos
1. Linse Bobadilla

## Design Review Dates/Timeline

Due in ~1 week (Oct 28/29)

## Introduction

We capture wheel speed by attaching toothed gears to our hubs and then counting every tooth that goes by. The sensor outputs a pulse every time a tooth goes past so we convert the frequency into an analog voltage read by a microcontroller (0 -> 3.3V). This is then interpreted as a value from 0 to 32767 as it is a signed 16 bit integer. We need a factor by which to scale this value (0-32767) down to rpm at the wheel. It would also be useful to have an accurate conversion from wheel RPM to MPH of the vehicle by determining the effective wheel radius (slightly different from just the tire radius because when the car compresses the wheel with its weight it gets a bit smaller.)

## Overview of the Problem

This is largely a dimensional analysis problem involving some communication with the electrical team to determine the exact calibration and circuitry of the peripheral boards (on which the frequency to voltage conversion happens.) 

## Steps in the project

1. Speak with the electrical team (probably via Wesley/Jack for LV and firmware) to determine the scaling of frequency to voltage and ADC conversion
1. Use this factor along with wheel-speed-disk tooth count (60)
1. Ensure the value is reasonable and compare to rpm data from the motor + gear ratio to see if the values seem correct
1. Determine effective wheel radius in conversation with mechanical team + maybe just a ruler
1. Calculate speed based on wheels and ensure it matches GPS speed

## Suggested direction

1. Start by outlining a plan for what you'll do. Take some notes. You can use this md file for notes. Ask questions about what you don't understand or what doesn't make sense.
1. Reach out to relevant members about needed information.
1. Take a look at data that exists to see what wheel speed currently looks like
1. Perform calculation
1. Compare results and write up some results documentation.
1. Prepare to present/discuss and let us know when you're done!