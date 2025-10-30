def stepSuspension(state, inputs, vehicleParams, dt):
    zS, zSDot, theta, thetaDot, phi, phiDot, psi, psiDot, u, v = state[0:10]
    zU1, zU1Dot, zU2, zU2Dot, zU3, zU3Dot, zU4, zU4Dot = state[10:18]
    alpha1, alpha2, alpha3, alpha4 = state[18:22]
    
    steerAngle, driveTorque, brakeTorque, zRoad1, zRoad2, zRoad3, zRoad4 = inputs
    
    mS = vehicleParams['mS']
    mU = vehicleParams['mU']
    Ixx, Iyy, Izz, Ixz = vehicleParams['Ixx'], vehicleParams['Iyy'], vehicleParams['Izz'], vehicleParams['Ixz']
    a, b = vehicleParams['a'], vehicleParams['b']
    trackF, trackR = vehicleParams['trackF'], vehicleParams['trackR']
    hCG, hRoll, hPitch = vehicleParams['hCG'], vehicleParams['hRoll'], vehicleParams['hPitch']
    kW = vehicleParams['kW']
    cJounce, cRebound = vehicleParams['cJounce'], vehicleParams['cRebound']
    kARBf, kARBr = vehicleParams['kARBf'], vehicleParams['kARBr']
    MR0 = vehicleParams['MR0']
    kMR = vehicleParams['kMR']
    rideRate, rollRate = vehicleParams['rideRate'], vehicleParams['rollRate']
    staticCamber = vehicleParams['staticCamber']
    sigma = vehicleParams['relaxationLength']
    
    r = psiDot
    
    xWheel = np.array([-a, -a, b, b])
    yWheel = np.array([-trackF/2, trackF/2, -trackR/2, trackR/2])
    
    zRel = np.zeros(4)
    zRel[0] = zU1 - (zS - theta*xWheel[0] + phi*yWheel[0]) - zRoad1
    zRel[1] = zU2 - (zS - theta*xWheel[1] + phi*yWheel[1]) - zRoad2
    zRel[2] = zU3 - (zS - theta*xWheel[2] + phi*yWheel[2]) - zRoad3
    zRel[3] = zU4 - (zS - theta*xWheel[3] + phi*yWheel[3]) - zRoad4
    
    zRelDot = np.zeros(4)
    zRelDot[0] = zU1Dot - (zSDot - thetaDot*xWheel[0] + phiDot*yWheel[0])
    zRelDot[1] = zU2Dot - (zSDot - thetaDot*xWheel[1] + phiDot*yWheel[1])
    zRelDot[2] = zU3Dot - (zSDot - thetaDot*xWheel[2] + phiDot*yWheel[2])
    zRelDot[3] = zU4Dot - (zSDot - thetaDot*xWheel[3] + phiDot*yWheel[3])
    MR = MR0 * (1 + kMR * zRel)
    
    # Springs
    fSpring = kW * zRel
    bumpStopEngaged = zRel > vehicleParams['bumpStopGap']
    fSpring[bumpStopEngaged] += vehicleParams['bumpStopRate'] * (zRel[bumpStopEngaged] - vehicleParams['bumpStopGap'])

    # Damper
    cDamper = np.where(zRelDot > 0, cJounce, cRebound)
    fDamper = cDamper * zRelDot * MR**2
    
    # ARB
    fARB = np.zeros(4)
    fARB[0] = -kARBf * (zRel[0] - zRel[1]) / trackF
    fARB[1] = kARBf * (zRel[0] - zRel[1]) / trackF
    fARB[2] = -kARBr * (zRel[2] - zRel[3]) / trackR
    fARB[3] = kARBr * (zRel[2] - zRel[3]) / trackR
    
