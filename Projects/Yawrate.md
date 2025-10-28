# Yaw Rate Modeling

## Active Members
1. Anthony Padilla
1. Brian Lee

## Design Review Dates/Timeline
- Literature review within 2 weeks of first sims meeting, November 11
- Implementation by end of first quarter
- Data based validation by end of first quarter

## Introduction
Yaw rate modeling is fundamental to all vehicle dynamics work, because
with a standard 3DoF vd model (Steering, Brake, Throttle), steering
represents one of the only degrees of control. The steering model must
model tranisent behavior of the vehicle given step steer inputs
allowing for effective data driven decisions.

## Overview of the Problem
A good steering model accounts for varying slip angle, cornering
stiffness, ackerman steering geometry, velocity, and steering angle
input. Additionally, the model is expected to be transient and
effectively deal with updates in step steer prior to the model
reaching steady state behavior. It is also important to understand the
force going through the steering column for driverless.

## Steps in the project
1. Literature review on models
1. Understanding of what is available and what numbers are given by
   other subteams
1. Implement a chosen model
1. Validate the model against data

## Suggested direction
Talk to Daniel Rhee for first steps and resources
