"""
Breakdown of spring-rate difference between no-aero and 40 m/s aero cases.
Shows load, moment, and stiffness calculations step-by-step.
"""
import numpy as np

# ============ Constants ============
mass = 293.97
weight = 2883.8457  # N
frontWD = 0.4632
rearWD = 1 - frontWD
CGHeight = 0.234  # meters
trackWidth = 1.325
wheelBase = 1.59
RCFront = 0.0203
RCRear = 0.0493
motionRatioF = 1.006
motionRatioR = 1.004
TRG = 0.01524409115  # target roll gradient, rad/g
masterAy = 1.7  # lateral g for spring sizing

# ============ No-Aero Case ============
print("=" * 70)
print("NO-AERO CASE")
print("=" * 70)

# Aero loads (zero)
frontAero_no = 0.0
rearAero_no = 0.0
totalAero_no = frontAero_no + rearAero_no

# Axle loads (static only)
frontAW_no = frontWD * weight + frontAero_no
rearAW_no = rearWD * weight + rearAero_no
totalAW_no = frontAW_no + rearAW_no

print(f"\nStatic weight distribution:")
print(f"  Front: {frontWD * weight:.2f} N")
print(f"  Rear:  {rearWD * weight:.2f} N")
print(f"  Aero contribution: {totalAero_no:.2f} N (front={frontAero_no:.2f} N, rear={rearAero_no:.2f} N)")
print(f"\nTotal axle loads at v=40 m/s, ay=0:")
print(f"  Front: {frontAW_no:.2f} N")
print(f"  Rear:  {rearAW_no:.2f} N")
print(f"  Total: {totalAW_no:.2f} N")

# Roll moments
frontRM_no = frontAW_no * (CGHeight - RCFront) * masterAy
rearRM_no = rearAW_no * (CGHeight - RCRear) * masterAy
M_roll_no = frontRM_no + rearRM_no

print(f"\nRoll moment calculation at ay={masterAy}g:")
print(f"  Front arm (CG - RC_F): {CGHeight - RCFront:.5f} m")
print(f"  Rear arm (CG - RC_R):  {CGHeight - RCRear:.5f} m")
print(f"  M_roll_front = {frontAW_no:.2f} N × {CGHeight - RCFront:.5f} m × {masterAy} = {frontRM_no:.2f} Nm")
print(f"  M_roll_rear  = {rearAW_no:.2f} N × {CGHeight - RCRear:.5f} m × {masterAy} = {rearRM_no:.2f} Nm")
print(f"  Total M_roll: {M_roll_no:.2f} Nm")

# Roll stiffness
frontRS_no = frontRM_no / TRG
rearRS_no = rearRM_no / TRG
Kphi_no = frontRS_no + rearRS_no

print(f"\nRoll stiffness (from target roll gradient TRG={TRG:.8f} rad/g):")
print(f"  K_phi_front = M_roll_front / TRG = {frontRM_no:.2f} / {TRG:.8f} = {frontRS_no:.2f} N⋅m/rad")
print(f"  K_phi_rear  = M_roll_rear / TRG  = {rearRM_no:.2f} / {TRG:.8f} = {rearRS_no:.2f} N⋅m/rad")
print(f"  Total K_phi: {Kphi_no:.2f} N⋅m/rad")

# Wheel rates
halfTrackSq = (trackWidth / 2)**2
frontKW_no = frontRS_no / (2 * halfTrackSq)
rearKW_no = rearRS_no / (2 * halfTrackSq)

print(f"\nWheel rates (from roll stiffness; track={trackWidth} m):")
print(f"  (track/2)² = {halfTrackSq:.5f} m²")
print(f"  K_W_front = K_phi_front / (2 × {halfTrackSq:.5f}) = {frontKW_no:.2f} N/m = {frontKW_no/1000:.3f} kN/m")
print(f"  K_W_rear  = K_phi_rear / (2 × {halfTrackSq:.5f})  = {rearKW_no:.2f} N/m = {rearKW_no/1000:.3f} kN/m")

# Spring rates
frontKS_no = frontKW_no / (motionRatioF**2)
rearKS_no = rearKW_no / (motionRatioR**2)

print(f"\nSpring rates (from wheel rates; motion ratios F={motionRatioF}, R={motionRatioR}):")
print(f"  K_S_front = K_W_front / {motionRatioF}² = {frontKS_no:.2f} N/m = {frontKS_no/1000:.3f} kN/m")
print(f"  K_S_rear  = K_W_rear / {motionRatioR}²  = {rearKS_no:.2f} N/m = {rearKS_no/1000:.3f} kN/m")

# ============ 40 m/s Aero Case ============
print("\n" + "=" * 70)
print("40 M/S AERO CASE")
print("=" * 70)

# Aero loads at 40 m/s (user-provided breakdown)
frontAero_aero = 1930.0
rearAero_aero = 1608.0
totalAero_aero = frontAero_aero + rearAero_aero

# Axle loads (static + aero)
frontAW_aero = frontWD * weight + frontAero_aero
rearAW_aero = rearWD * weight + rearAero_aero
totalAW_aero = frontAW_aero + rearAW_aero

print(f"\nStatic weight distribution:")
print(f"  Front: {frontWD * weight:.2f} N")
print(f"  Rear:  {rearWD * weight:.2f} N")
print(f"  Aero contribution: {totalAero_aero:.2f} N (front={frontAero_aero:.2f} N, rear={rearAero_aero:.2f} N)")
print(f"\nTotal axle loads at v=40 m/s, ay=0:")
print(f"  Front: {frontAW_aero:.2f} N")
print(f"  Rear:  {rearAW_aero:.2f} N")
print(f"  Total: {totalAW_aero:.2f} N")

# Roll moments
frontRM_aero = frontAW_aero * (CGHeight - RCFront) * masterAy
rearRM_aero = rearAW_aero * (CGHeight - RCRear) * masterAy
M_roll_aero = frontRM_aero + rearRM_aero

print(f"\nRoll moment calculation at ay={masterAy}g:")
print(f"  Front arm (CG - RC_F): {CGHeight - RCFront:.5f} m")
print(f"  Rear arm (CG - RC_R):  {CGHeight - RCRear:.5f} m")
print(f"  M_roll_front = {frontAW_aero:.2f} N × {CGHeight - RCFront:.5f} m × {masterAy} = {frontRM_aero:.2f} Nm")
print(f"  M_roll_rear  = {rearAW_aero:.2f} N × {CGHeight - RCRear:.5f} m × {masterAy} = {rearRM_aero:.2f} Nm")
print(f"  Total M_roll: {M_roll_aero:.2f} Nm")

# Roll stiffness
frontRS_aero = frontRM_aero / TRG
rearRS_aero = rearRM_aero / TRG
Kphi_aero = frontRS_aero + rearRS_aero

print(f"\nRoll stiffness (from target roll gradient TRG={TRG:.8f} rad/g):")
print(f"  K_phi_front = M_roll_front / TRG = {frontRM_aero:.2f} / {TRG:.8f} = {frontRS_aero:.2f} N⋅m/rad")
print(f"  K_phi_rear  = M_roll_rear / TRG  = {rearRM_aero:.2f} / {TRG:.8f} = {rearRS_aero:.2f} N⋅m/rad")
print(f"  Total K_phi: {Kphi_aero:.2f} N⋅m/rad")

# Wheel rates
frontKW_aero = frontRS_aero / (2 * halfTrackSq)
rearKW_aero = rearRS_aero / (2 * halfTrackSq)

print(f"\nWheel rates (from roll stiffness; track={trackWidth} m):")
print(f"  (track/2)² = {halfTrackSq:.5f} m²")
print(f"  K_W_front = K_phi_front / (2 × {halfTrackSq:.5f}) = {frontKW_aero:.2f} N/m = {frontKW_aero/1000:.3f} kN/m")
print(f"  K_W_rear  = K_phi_rear / (2 × {halfTrackSq:.5f})  = {rearKW_aero:.2f} N/m = {rearKW_aero/1000:.3f} kN/m")

# Spring rates
frontKS_aero = frontKW_aero / (motionRatioF**2)
rearKS_aero = rearKW_aero / (motionRatioR**2)

print(f"\nSpring rates (from wheel rates; motion ratios F={motionRatioF}, R={motionRatioR}):")
print(f"  K_S_front = K_W_front / {motionRatioF}² = {frontKS_aero:.2f} N/m = {frontKS_aero/1000:.3f} kN/m")
print(f"  K_S_rear  = K_W_rear / {motionRatioR}²  = {rearKS_aero:.2f} N/m = {rearKS_aero/1000:.3f} kN/m")

# ============ Comparison ============
print("\n" + "=" * 70)
print("DIFFERENCE (Aero - No-Aero)")
print("=" * 70)

delta_frontAW = frontAW_aero - frontAW_no
delta_rearAW = rearAW_aero - rearAW_no
delta_totalAW = totalAW_aero - totalAW_no

print(f"\nAxle load delta:")
print(f"  ΔFront AW: {delta_frontAW:+.2f} N")
print(f"  ΔRear AW:  {delta_rearAW:+.2f} N")
print(f"  ΔTotal AW: {delta_totalAW:+.2f} N")

delta_M_roll = M_roll_aero - M_roll_no
print(f"\nRoll moment delta:")
print(f"  ΔM_roll: {delta_M_roll:+.2f} Nm ({delta_M_roll/M_roll_no*100:+.1f}%)")

delta_Kphi = Kphi_aero - Kphi_no
print(f"\nRoll stiffness delta:")
print(f"  ΔK_phi: {delta_Kphi:+.2f} N⋅m/rad ({delta_Kphi/Kphi_no*100:+.1f}%)")

delta_frontKW = frontKW_aero - frontKW_no
delta_rearKW = rearKW_aero - rearKW_no
print(f"\nWheel rate delta:")
print(f"  ΔK_W_front: {delta_frontKW:+.2f} N/m = {delta_frontKW/1000:+.3f} kN/m")
print(f"  ΔK_W_rear:  {delta_rearKW:+.2f} N/m = {delta_rearKW/1000:+.3f} kN/m")

delta_frontKS = frontKS_aero - frontKS_no
delta_rearKS = rearKS_aero - rearKS_no
print(f"\nSpring rate delta:")
print(f"  ΔK_S_front: {delta_frontKS:+.2f} N/m = {delta_frontKS/1000:+.3f} kN/m")
print(f"  ΔK_S_rear:  {delta_rearKS:+.2f} N/m = {delta_rearKS/1000:+.3f} kN/m")
print(f"  Average ΔK_S: {(delta_frontKS + delta_rearKS)/2/1000:+.3f} kN/m")

# ============ Sensitivity Analysis ============
print("\n" + "=" * 70)
print("SENSITIVITY: Spring Rate per N of Total Axle Load")
print("=" * 70)

sens_no = (frontKS_no + rearKS_no) / (2 * totalAW_no)
sens_aero = (frontKS_aero + rearKS_aero) / (2 * totalAW_aero)

print(f"\nNo-aero:  {sens_no:.6f} (N/m) per (N of total load)")
print(f"40 m/s:   {sens_aero:.6f} (N/m) per (N of total load)")
print(f"\nNote: These are nearly equal because spring rate is defined to keep")
print(f"roll gradient constant. The ~{delta_M_roll/M_roll_no*100:.0f}% jump in moment is matched")
print(f"by a ~{delta_Kphi/Kphi_no*100:.0f}% jump in roll stiffness to maintain fixed TRG.")
