# Blue Max Lap Analysis

## Active Members
1. Girish Prabu
1. Sal Chavez Palacios
1. Haaris Wasti

## Design Review Dates/Timeline

- Data meeting soon for discussion about data projects
- PDR (Preliminary Design Review) ~ Oct 30 - Nov 2 to establish that you have a clear plan for how to proceed
- Design Review ~1 week later

## Introduction

Blue max kart club is where we've tested for about the past year. You can find it on google maps if you want to look at the track. We want to know what kind of driving makes for a fast lap but this is a very complicated problem. It depends a bit on track conditions (including the occasional karter on the track) and a lot on the driver. Your goal is to figure out how drivers were able to go faster or slower based on their different times spent in each turn, potentially stuff like racing lines, amount of energy usage, and the like.

## Overview of the Problem

Using our track data from blue max endurance runs, determine what makes for a fast lap. The first thing you'll likely notice is more power = fater lap but that isn't always the case. Break down the laps into parts, determine where time is lost, and come up with ways the driver can go faster or drive more optimally.

## Steps in the project

1. Determine files that contain endurance data to compare. May also be worth including some data from comp but we only have power usage from comp and it's a completely different track.
1. Read some literature about how to optmize lap performance from data. Some information/articles are available in pinned messages in #software.
1. Analyze lap components based on methods outlined in research.
1. Deliverable is a report of how laps differ and what impacts lap performance including things like:
    1. Suspension
    1. Tractive Battery (Also known as Accumulator in old data)
    1. Steering data (if available)
    1. GPS data
    1. (etc.)

## Suggested direction

Use Polars and Matplotlib to visualize some lap data using the functions available in [AnalysisFunctions.py](Data\AnalysisFunctions.py). lapSegementation and timeCol will be useful for doing per lap analysis. 