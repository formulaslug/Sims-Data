mass = 278.92
weight = 2736  # N
frontWD = 0.4632
rearWD = 1 - frontWD
leftWD = 0.492
CGHeight = 0.234  # meters
trackWidth = 1.234
wheelBase = 1.59
RCFront = 0.0203
RCRear = 0.0493
motionRatioF = 1.006
motionRatioR = 1.004
TRG = 0.0314   # target roll gradient in rad/g

###     Axle Weights    ###

frontAW = frontWD * weight
rearAW = rearWD * weight

###     Roll Moments    ###

frontRM = frontAW * (CGHeight - RCFront)
rearRM = rearAW * (CGHeight - RCRear)

###     Finding Required Roll stiffness     ###

frontRS = frontRM / TRG
rearRS = rearRM / TRG

###     Moment from one side about CG       ###
# roll = Φ
# track width = t
# Δz = Φ * t/2
# wheel rate per side = kw
# Force at each wheel:
#     F = kw * Φ * t/2
# Moment from one side:
#     Mside = F * (t/2) = kw * Φ * (t/2)^2

# (t/2)^2 term
halfTrackSq = (trackWidth / 2)**2

###     Wheel Rates    ###
# Kphi = 2 * kw * (t/2)^2
# kw = Kphi / (2 * (t/2)^2)

frontKW = frontRS / (2 * halfTrackSq)
rearKW = rearRS / (2 * halfTrackSq)

###     Spring Rates (using motion ratios)     ###
# kw = ks * MR^2
# ks = kw / MR^2

frontKS = frontKW / (motionRatioF**2)
rearKS = rearKW / (motionRatioR**2)

###     Print results     ###

print("Front Wheel Rate (kN/m):", frontKW/1000)
print("Rear Wheel Rate (kN/m):", rearKW/1000)
print("Front Spring Rate (N/mm):", frontKS / 1000)
print("Rear Spring Rate (N/mm):", rearKS / 1000)
