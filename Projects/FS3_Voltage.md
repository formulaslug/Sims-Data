# FS-3 Voltage Model

## Active Members

1. Caydence Tatlow

## Design Review Dates/Timeline

PDR in ~1 week

## Introduction

In the past, a few people have made voltage models for FS-3 but they fail in their ability to account for histeresis. Read the link in the next section for more info on overview and introduction. The cell we used are the Murata VTC5A cells. Their discharge curve and other information can be found pretty readily online.

## Overview of the Problem

[More info](../Docs/Battery%20Modeling.md)

## Steps in the project

1. First of all, work with the person doing FS-4 voltage modeling to determine a good model.
1. Come up with a plan together of what equations to use and be ready for a PDR.
1. Fit your model to existing FS-3 Data
1. Validate it against other FS-3 Data

## Suggested direction

I messed around with using convolution for histeresis (for a kind of rolling average of the previous x data points) which may work if you get your convolution matrix right but a more sophisticated function (like something actually logarithmic) may be in order. Another kind of model may also be useful based on current draw or charge usage. Worth reading more about.

The SOC charge model exists in various forms. [This one](Data/CellModel.py) may be a good place to start and is relatively simple.

Temperature hasn't been done yet but could use something very similar to the ```CellModel.py``` version for just a small increase in voltage.

Then scale these two values based on fit coefficients and use scipy's ```curve_fit``` to fit values to these coefficients.