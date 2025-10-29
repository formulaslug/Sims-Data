# Battery Thermal Modeling General Docs

## What is a battery model?

For the sake of our car, a battery model is a set of equations (or neural network or other computational data structure) that produces the next time step given the previous data. We want to do this for Voltage and Temperature/stored energy based on SOC, current draw, and some histeresis related to current draw and voltage.

Some battery modeling exists without accounting for histeresis in the MBS folder.

## Voltage Modeling

Voltage is based on the SOC and current draw from a previous time step. The SOC part can be estimated based on a discharge curve like [these](https://www.murata.com/-/media/webrenewal/products/batteries/cylindrical/datasheet/us18650vtc5a-product-datasheet.ashx?la=en-us&cvid=20250324010000000000). 

If current draw isn't taken into account, voltage follows a fairly predictable line. With current draw as part of the mix, the voltage tends to sag when high current draw occurs and then recovers slowly. The histeresis based current bit must be trained based on our existing data to fit some kind of curve (to be determined by you.)

Temperature also plays a role. Increases in temperature can increase the voltage and the amount of energy we can extract from the tractive battery per unit charge.

## Thermal Modeling

Power output due to batteries is roughly $I^2*R$ where $I$ is the current draw and $R$ is the impedence of the cells listed on the data sheet. Combine this with $mc∆T=Q$ where $m$ is the mass, $c$ is the specific heat capacity of the cells, $∆T$ is the change in temperature, and $Q$ is the heat energy. With these two equations, you can estimate the change in temperature of the battery cells, and then use convection and conduction to look at heat transfer and loss to the surrounding environment. You can reference [this doc](https://fsae.slack.com/archives/C07U62H6R46/p1749169487545169?thread_ts=1749169442.918919&cid=C07U62H6R46) for the thermal equations related to convection and conduction (radiation is probably negligable but worth checking honestly with a brief calculation). 