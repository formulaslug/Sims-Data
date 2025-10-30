# Tractive Battery Thermal Modeling + Fan

## Active Members

1. Dhruv Kolte
1. Praneeth Mandalapu

## Design Review Dates/Timeline

- Can talk about this at data meeting

## Introduction

Batteries generate a predictable amount of heat during operation. It is proportional to the interal impdence/resistance ($Z$), and current($I$) $P = I^2R$. Combine with with the equation $mc∆T=Q$ where $m$ is the mass of the material $c$ is the specific heat capacity, $∆T$ is the change in temperature, and $Q$ which is the heat (energy), we can predict the change in temperature due to this energy output. With a temperature change, we can use yet another physics equation to predict the amount of energy leaving the material via convection. I leave these up to you to read [here](https://fsae.slack.com/files/U05V4TEQL67/F0909L6EPNW/heattransferbooklet.pdf). The first page will be the most important/relevant.

## Overview of the Problem

We want to create an accurate thermal model of our TB (Tractive Battery) using these equations. To do this, we can roughly calculate the specific heat capacity the cell/plastic interior and look up the aluminum specific heat capacity. Your goal is to determine how much energy is leaving the accumulator as a function of the difference in temperature between the TB and Ambient, and of the fan % (like 0% fans vs 100% fans.) To do this, you have data from testing of the accumulator heating up in different ambient temperatures (which you can find by looking at the acc temp at the beginning of a day while not driving and occasionally in logs).

This project also ties closely into FS-3 thermal modeling (which no one chose on sims) so you may have a bit of work to do there (aligning your values with our actual battery).

## Steps in the project

1. Start by outlining a plan for what you'll do. Take some notes. You can use this md file for notes. Ask questions about what you don't understand or what doesn't make sense.
1. Determine a thermal model using the thermal equations linked in the introduction. It should relate energy removed to fan% and difference between tractive battery temp and ambient temp.
    - Will require that you look up some thermal constants of materials and ask around about materials used in the tractive battery air path. I (Nathaniel) have some notes too that may be of use.
1. Prepare your data to fit to that equation.
1. Use a curve fitting method of some kind to determine your constants.
1. Validate your model against another run that wasn't part of your calibration to see how well it matches.

## Suggested direction

Use polars for data preparation and scipy curve_fit to fit the data to your equation. Examples exist in [imuCalibration.py](../Data/imuCalibration.py)
