# General Modeling Notes

## Table of Contents

1. [Brakes](#brakes)
<!-- 1. [Model Training](#mt)
    1. [Tire Modeling](#tm)
    1. [Voltage Modeling](#vm) -->


<h2 id="brakes"> Brakes </h2>

Willwood gp200 calipers. Purple compound on FS-3. Switching to Red compound on FS-4 which has higher coeff of friction and less temperature dependence (purple tends to get worse at higher temperature). 1.23 in^2 piston area. $Area * Pressure = Brake Force$. And with $PSI$ and $in^2$ you get $lb force$ out which must be converted to $N$ at $1 lb = 4.448222 N$. 