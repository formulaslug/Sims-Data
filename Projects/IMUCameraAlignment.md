# Attitude Estimation for an IMU

## Active Members

1. Edward Mendoza Morales

## Design Review Dates/Timeline

PDR in ~1 week from start

## Introduction

We would like to know our camera angle to better estimate the location of the ground when doing depth estimation and localization for autonomous.

## Overview of the Problem

It can be difficult to estimate the position given just a single camera so we will be calibrating based on an IMU. An IMU contains a 3-axis magnetometer, a 3-axis gyroscope, and a 3-axis accelerometer. The magnetometer measures the vector of the magnetic field in 3d space (roughly north due to the earth's magnetic field). The gyroscope measures angular velocity (Rate of rotation) in 3d space, and the accelerometer measures acceleration (change in velocity) in 3d space. Each of these individual components has their own issues that we should discuss at a meeting.

## Steps in the project

1. Take some time to discuss and research IMUs and read the articles [Here](https://fsae.slack.com/archives/C07U62H6R46/p1739385450464149)
1. Develop a plan that you present at a PDR for how you will tackle this problem
1. Calibrate based on the methods detailed in the articles
1. Validate with a live IMU

## Suggested direction

There is some IMU data I (Nathaniel) have created in fs-data. It should be decently well labeled to be found. Once you have a calibration plan and know how to calibrate, that data should prove useful for some calibration methods. If there are other bits of data you need, I can help you generate them and I also have an IMU you can do your live tests on.