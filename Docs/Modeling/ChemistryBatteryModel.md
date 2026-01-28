# Deriving a Battery model from first principles/chemistry
by Nathaniel Platt

In gen chem I learned about galvanic cells AKA your typical battery. They are made up of 2 half reactions: one for the anode and one for the cathode. You take the sum of their half reaction voltages to get $V_0$ or the voltage under equilibrium (same concentration of charge in the anode and cathode.) My guess is this is the same as the nominal voltage but I'm not sure. The whole equation is written out as $$V = V_0 + \frac{RT}{nF}*ln(\frac{[anode]}{[cathode]})$$ where $[anode]$ and $[cathode]$ are the concentration of ions in the anode/cathode respectively. 
1. $R$ is the gas constant equal to ```8.31```. 
1. $T$ is temperature in Kelvin ```293``` for 20 deg C.
1. $n$ is the number of electrons processed in each reaction. I don't remember what this is at the time of creating this model so I assumed ```1``` which is a fine guess and will be fixed either way by an ML constant later. It will not show up in any more equations because it is assumed to be 1.
1. $F$ is the faraday constant or ```96,485``` 

If you make some basic assumptions you can expand this model further and allow a bit of machine learning.

### Assumption 1: The volumes of each electrolyte are equivalent

The concentration of ions in the anode and cathode are each just $\frac{Charge}{Volume}$ and if the volumes are the same then they cancel out:
$$\frac{\frac{Q\substack{anode}}{\cancel{V\substack{anode}}}}{\frac{Q\substack{cathode}}{\cancel{V\substack{cathode}}}}$$
- $Q\substack{anode}$ is the charge in the anode
- $V\substack{anode}$ is the volume of the anode

so we just get $\frac{Q_a}{Q_c}$ and since $Q_a + Q_c = Q\substack{total}$ or $6Ah$ we can effectively say it is $\frac{SOC}{1-SOC}$ so with that we have 
$$V = V_0 + \frac{RT}{F}*ln(\frac{SOC}{1-SOC})$$

### Assumption 2: The cell never goes below about 0.001M in either direction

This assumption is based on not going below ```2.5V``` or above ```4.2V``` which correspond to SOC close to 0 or 1. When SOC approaches 0 or 1, the value out of the natural log skyrockets in the positive or negative direction (as seen in discharge curves when they go outside of those voltage bounds). In practice batteries tend to start growing dendrites in their anode or cathode and eventally short circuit and catch on fire. To accomplish this I essentially scaled down SOC a little bit by subtracting $0.1^a$ where an $a$ of $3$ worked well. This then looks like 

$$V = V_0 + \frac{RT}{F}*ln(\frac{SOC-0.1^3}{1-(SOC-0.1^3)})$$

### Assumption 3: Some of the values I came up with are wrong

This is where the machine learning comes in. I threw in a few constants in areas where I knew I would be wrong that the model could correct for. It comes out to 

$$V = V_0*C_4 + C_2\frac{RT}{F}*ln(\frac{C_1(SOC-0.1^3) + C_3}{1-(SOC-0.1^3)})$$

1. $C_1$ accounts for my first assumption. It is likely that the anode and cathode volume are not the same and this allows that to be a parameter.
1. $C_2$ accounts for any error in temperature dependence, to best fit the discharge curve slope, and for any errors in by assumption abotu $n$.
1. $C_3$ is part of the $0.001M$ correction and helps shift it in the correct direction.
1. $C_4$ accounts for errors in my assumption that the nominal voltage is equal to $V_0$ and generally just allows the model to fit that value rather than take it as an input.

### Fix 1:

So turns out I forgot ln rules and the $C_4$ and $C_1$ are redundant so removing $C_4$...

### Fix 2:

Adding $C_4$ to the bottom to do something similar to $C_3$. Also adjusting it so $C_1$ multiplies $C_3$ which makes it less separable but less confusing.

$$V = V_0 + C_2\frac{RT}{F}*ln(\frac{C_1(SOC-0.1^3 + C_3)}{1-(SOC-0.1^3) + C_4})$$