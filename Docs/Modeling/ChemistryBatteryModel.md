# Deriving a Battery model from first principles/chemistry
by Nathaniel Platt

In gen chem I learned about galvanic cells AKA your typical battery. They are made up of 2 half reactions: one for the anode and one for the cathode. You take the sum of their half reaction voltages to get $V_0$ or the voltage under equilibrium (same concentration of charge in the anode and cathode.) My guess is this is the same as the nominal voltage but I'm not sure. The whole equation is written out as $$V = V_0 + \frac{RT}{nF}*ln(\frac{[anode]}{[cathode]})$$ where $[anode]$ and $[cathode]$ are the concentration of ions in the anode/cathode respectively. 
1. $R$ is the gas constant equal to ```8.31```. 
1. $T$ is temperature in Kelvin ```293``` for 20 deg C.
1. $n$ is the number of electrons processed in each reaction. I don't remember what this is at the time of creating this model so I assumed ```1``` which is a fine guess and will be fixed either way by an ML constant later.
1. $F$ is the faraday constant or ```96,485``` 

If you make some basic assumptions you can expand this model further and allow a bit of machine learning.

### Assumption 1: The volumes of each electrolyte are equivalent

The concentration of ions in the anode and cathode are each just $\frac{Charge}{Volume}$ and if the volumes are the same then they cancel out:
$$\frac{\frac{Q\substack{anode}}{\cancel{V\substack{anode}}}}{\frac{Q\substack{cathode}}{\cancel{V\substack{cathode}}}}$$
- $Q\substack{anode}$ is the charge in the anode
- $V\substack{anode}$ is the volume of the anode

so we just get $\frac{Q_a}{Q_c}$ and since $Q_a + Q_c = Q\substack{total}$ or $6Ah$ we can effectively say it is $\frac{SOC}{1-SOC}$ so with that we have 
$$V = V_0 + \frac{RT}{nF}*ln(\frac{SOC}{1-SOC})$$