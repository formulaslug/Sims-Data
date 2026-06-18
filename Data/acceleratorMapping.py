import numpy as np
import matplotlib.pyplot as plt

def accelerator_mapping (pedal_travel):
    region1 = 0.3
    region1_scale = 3
    region2 = 0.4
    region3_scale = 3

    region3 = 1 - region1 - region2
    scaled_region1 = region1/region1_scale
    scaled_region3 = region3/region3_scale
    scaled_region2 = 1 - scaled_region1 - scaled_region3
    region2_scale = region2/scaled_region2

    if pedal_travel < scaled_region1:
        return pedal_travel*region1_scale
    if pedal_travel < scaled_region2 + scaled_region1:
        p1 = scaled_region1
        p1_power = p1*region1_scale
        p2 = pedal_travel - scaled_region1
        p2_power = p2*region2_scale
        return p1_power + p2_power
    p1 = scaled_region1
    p1_power = p1*region1_scale
    p2 = scaled_region2
    p2_power = p2*region2_scale
    p3 = pedal_travel - scaled_region1 - scaled_region2
    p3_power = p3*region3_scale
    return p1_power + p2_power + p3_power
    
    

arr = np.arange(0, 1, 0.01)

plt.plot(arr, [accelerator_mapping(x) for x in arr])
plt.plot(arr, 0.5*np.arctanh(2*arr-1))
plt.show()