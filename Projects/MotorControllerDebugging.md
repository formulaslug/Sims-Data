# FS-3 Motor Controller Debugging Fall 2025

## Active Members

## Design Review Dates/Timeline

N/A -- ASAP (by next testing day)

## Introduction

Our tractive system (high voltage path which delivers energy to the motor) is designed for about 80kW peak power for short bursts. This is a part of the powertrain design and high voltage system design (mechanical and electrical integration). In practice, we must limit our power to about 30-40kW because of issues with our motor controller that occasionally prevent the car from moving in semi-rare and unpredictable ways.

We set our max current in firmware on the ETC board (Electronic Throttle Control) and:
1. At 300A there are next to no issues.
1. At 400A we have occasional issues.
1. At 600A we have semi regular issues.

## Overview of the Problem

## Steps in the project

## Suggested direction