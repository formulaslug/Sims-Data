# FS-4 Voltage Model

## Active Members

1. Eva Jain

## Design Review Dates/Timeline

PDR in ~1 week

## Introduction

In the past, a few people have made voltage models for FS-3 but they fail in their ability to account for histeresis. No one has made one for FS-4 yet. Read the link in the next section for more info on overview and introduction. The cell for FS-4 will be a PB-30

## Overview of the Problem

[More info](../Docs/Battery%20Modeling.md)

## Steps in the project

1. First of all, work with the person doing FS-3 voltage modeling to determine a good model.
1. Come up with a plan together of what equations to use and be ready for a PDR.
1. We will hopefully get Batemo data soon that you can fit your model to. We may also be able to test cells that we get sponsored from another company + the testing equipment we get from Hioki
1. Validate whatever model is made against cell testing data and then eventually FS-4 data.

## Suggested direction

I messed around with using convolution for histeresis (for a kind of rolling average of the previous x data points) which may work if you get your convolution matrix right but a more sophisticated function (like something actually logarithmic) may be in order. Another kind of model may also be useful based on current draw or charge usage. Worth reading more about.

The SOC charge model exists in various forms for FS-3. [This one](Data/CellModel.py) may be a good place to start and is relatively simple (Again, FS-3 so will need to be adjusted to discharge data for the PB-30s).

Temperature hasn't been done yet but could use something very similar to the ```CellModel.py``` version for just a small increase in voltage.

Then scale these two values based on fit coefficients and use scipy's ```curve_fit``` to fit values to these coefficients.