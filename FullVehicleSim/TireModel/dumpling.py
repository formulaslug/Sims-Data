# This (hopefully) implements MF 6.x (I think 6.2) as described in Tire and Vehicle Dynamics 3rd edition on page 179
import math

class Tire:


    def __init__(self, normalForce, slipRatio, slipAngle, velocityX, pressure, temperature, camber, mechanicalParams, magicParams):
        self.magic = magicParams
        self.mechanical = mechanicalParams

        self.normalForce = normalForce
        self.velocityX = velocityX
        self.slipRatioInit = slipRatio
        self.slipRatio = self.slipRatioInit
        self.slipAngle = slipAngle
        self.tirePressure = pressure
        self.tireTemperature = temperature
        self.actPressure = pressure # Actual PSI
        self.camber = camber # Radians




        #if(lat):
        self.normDeltaLoadLat = self.normalizeLoadLat()
        self.normDeltaPressureLat = self.normalizePressureLat()
        #if(long):
        self.normDeltaLoadLong = self.normalizeLoadLong()
        self.normDeltaPressureLong = self.normalizePressureLong()

        self.normalForce = self.getNormalLoad(self.normalForce)


    def getNormalLoad(self, inputNormalForce):
        # Neglecting last force
        # I intentionally neglect the last Fx and Fy because that would involve a large rewrite of this.
        sqrt_term = math.sqrt(9.81 * self.mechanical["unloaded-radius"])
        term1 = (1 + self.magic["q_v2"] * abs(self.magic["Omega"]) * self.mechanical["unloaded-radius"]/sqrt_term - self.magic["q_Fcx"] - self.magic["q_Fcy"])
        # We assume the deflection is 1 because idk how to do that
        term2 = (self.magic["q_Fz1"] + self.magic["q_Fz2"] * self.camber**2) / self.mechanical["unloaded-radius"]
        term3 = (1 + self.magic["P_pFz1"] * self.normDeltaPressureLong) * inputNormalForce

        return term1 * term2 * term3

    ##### ********************************
    ##### LATERAL SLIP FUNCTION
    ##### ********************************


    def getLateralForce(self):
        Alphas = self.magic["lambda_alphastar"] * self.slipAngle * math.copysign(1, self.velocityX)
        Byk = self.magic["r_by1"]# + self.magic["r_by4"] * math.sin(self.camber) ** 2) * math.cos(math.atan(self.magic["r_by2"] * (Alphas - self.magic["r_by3"]))) * self.magic["lambda_yk"]
        Cyk = self.magic["r_cy1"]
        Eyk = self.magic["r_ey1"] + self.magic["r_ey2"] * self.normDeltaLoadLat
        Shyk = self.magic["r_hy1"] + self.magic["r_hy2"] * self.normDeltaLoadLat


        Ks = self.slipRatio + Shyk
        BykKs = Byk * Ks
        BykShyk = Byk * Shyk
        Gykappa = math.cos(Cyk * math.atan(BykKs - Eyk * (BykKs - math.atan(BykKs))))
        Gykappazero =  math.cos(Cyk * math.atan(BykShyk - Eyk * (BykShyk - math.atan(BykShyk))))


        Dvyk = self.mechanical["friction-coeff-lat"] * self.normalForce * (self.magic["r_vy1"] + self.magic["r_vy2"] * self.normDeltaLoadLat + self.magic["r_vy3"] * math.sin(self.camber)) * math.cos(math.atan(self.magic["r_vy4"] * math.sin(Alphas)))  * self.magic["zeta_2"]
        Svyk = Dvyk * math.sin(self.magic["r_vy5"] * math.atan(self.magic["r_vy6"] * self.slipRatio)) * self.magic["lambda_vyk"]

        #print(Byk, Cyk, Eyk, Shyk)

        return Gykappa/Gykappazero * self.getLateralForcePure() #+ Svyk # + self.magic["Svyk"]

    def getLateralForcePure(self):
        Alphas = self.magic["lambda_alphastarypure"] * self.slipAngle * math.copysign(1,self.velocityX)

        loadDependentPeak = self.magic["loadA"] * self.normalForce * self.normalForce + self.magic["loadB"] * self.normalForce + self.magic["loadC"]

        self.Cy = self.magic["p_cy1"]
        self.Dy = loadDependentPeak * self.getLateralCoefficientOfFriction() * self.normalForce * (self.magic["tempYAPure"] * self.tireTemperature ** 2 + self.magic["tempYBPure"] * self.tireTemperature + self.magic["tempYCPure"])
        self.By = self.magic["By_pure"]
        self.Ey = self.getLateralE(Alphas)

        Svy = self.magic["Svy"]
        return self.stdCurveSine(self.By, self.Cy, self.Dy, self.Ey, self.slipAngle) + Svy

    def getLateralB(self):
        Kyalpha = self.magic["p_ky1"] * self.normDeltaLoadLat * (1 + self.magic["p_py1"] * self.normDeltaPressureLat) * (1 - self.magic["p_ky3"] * abs(math.sin(self.camber))) * math.sin(self.magic["p_ky4"] * math.atan(1/(self.magic["lambda_nominalload"] * (self.magic["p_ky2"] + self.magic["p_ky5"] * math.sin(self.camber)**2) * (1+ self.magic["p_py2"] * self.normDeltaPressureLat) ) )) * self.magic["zeta3"] * self.magic["lambda_kyalpha"]
        By = Kyalpha / (self.Cy * self.Dy + self.magic["epsilon_y"])
        return By
    def getLateralCoefficientOfFriction(self):
        return (self.magic["p_dy1"] + self.magic["p_dy2"] * self.normDeltaLoadLat) * (1 + self.magic["p_py3"] * self.normDeltaPressureLat + self.magic["p_py4"] * self.normDeltaPressureLat ** 2) * (1 - self.magic["p_dy3"] * math.sin(self.camber) ** 2) * self.magic["lambda_coeffscalary"]
    def getLateralE(self, Alphas):
        term1 = (self.magic["p_ey1"] + self.magic["p_ey2"] * self.normDeltaLoadLat)
        term2 = (1 + self.magic["p_ey5"] * math.sin(self.camber) ** 2 - (self.magic["p_ey3"] + self.magic["p_ey4"] * math.sin(self.camber)) * Alphas)
        return term1 * term2 * self.magic["lambda_ey"]

    ##### ********************************
    ##### LONGITUDINAL COMBINED SLIP FUNCTIONS
    ##### ********************************

    def getLongForce(self):
        tempScalar = self.magic["tempXA"] * self.tireTemperature ** 2 + self.magic["tempXB"] * self.tireTemperature + self.magic["tempXC"]

        #if self.slipAngle < 0.1 and self.slipAngle > -0.1:
        #    return self.getLongForcePureSlip()

        #if (self.getLongForceCombinedSlip() * tempScalar)/self.normalForce > 5:
        #    return 5 * self.normalForce
        #elif (self.getLongForceCombinedSlip() * tempScalar)/self.normalForce < -5:
        #    return -5 * self.normalForce
        return self.getLongForceCombinedSlip() * tempScalar


    def getLongForceCombinedSlip(self):
        fCoefficient = self.getGxalpha()
        #if fCoefficient > 1:
        #    fCoefficient = 1
        force = self.getLongForcePureSlip()
        #print(force * fCoefficient)
        #print(fCoefficient)
        return force * (fCoefficient + self.magic["combined_long_offset"])


    def getGxalpha(self):
        Cxalpha = self.magic["r_cx1"]
        Bxalpha = (self.magic["r_bx1"] + self.magic["r_bx3"] * math.sin(self.camber) ** 2) * math.cos(math.atan(self.magic["r_bx2"] * self.slipRatio))  * self.magic["lambda_xalpha"]
        Exalpha = self.magic["r_ex1"] + self.magic["r_ex2"] * self.normDeltaLoadLong
        Shxalpha = self.magic["r_hx1"]
        #print("LAMBDA ALPHA STAR", self.magic["lambda_alphastar"], "SA", self.slipAngle)
        Alphas = self.magic["lambda_alphastar"] * self.slipAngle * math.copysign(1, self.velocityX) + Shxalpha

        Gxalpha_init = math.cos(Cxalpha * math.atan( Bxalpha * Alphas - Exalpha * ( Bxalpha * Alphas - math.atan(Bxalpha * Alphas) ) ) )
        Gxalphanaught = math.cos(Cxalpha * math.atan( Bxalpha * Shxalpha - Exalpha * ( Bxalpha * Shxalpha - math.atan(Bxalpha * Shxalpha) ) ) )

        #print(Bxalpha, Cxalpha, Exalpha, Shxalpha)

        return Gxalpha_init / Gxalphanaught * self.magic["lambda_combinedslipcoeff"]
    ##### ********************************
    ##### Lateral PURE SLIP FUNCTIONS
    ##### ********************************

    def getLatForcePureSlip(self):
        return self.mechanical["friction-coeff-lat"] * self.normalForce

    ##### ********************************
    ##### LONGITUDINAL PURE SLIP FUNCTIONS
    ##### ********************************


    def getLongForcePureSlip(self):

        #return self.slipRatio * 100

        tempScalarPure = self.magic["tempXAPure"] * self.tireTemperature ** 2 + self.magic["tempXBPure"] * self.tireTemperature + self.magic["tempXCPure"]

        self.Cx = self.magic["shape-factor"] # Shape Factor (this thing is entirely magic. I think.) P_cx1
        self.Dx = self.getMaxLongFriction() * tempScalarPure # Peak Factor
        self.Ex = self.getCurvatureFactor() # Curvature Factor
        self.Bx = self.getStiffnessFactorLong() # Stifness Factor

        Svx = self.getVerticalShift()

        longForce = self.stdCurveSine(self.Bx, self.Cx, self.Dx, self.Ex, self.slipRatio) + Svx

        #print(self.Bx, self.Cx, self.Dx, self.Ex, Svx)
        # Safety
        #longForce = max(longForce, self.mechanical["friction-coeff-long"] * self.normalForce)
        self.longforce = longForce

        return self.longforce
    def getMaxLongFriction(self):
        # Here there needs to be the mystery camber parameter zeta sub 1.
        return self.getLongFrictionCoefficient() * self.normalForce * self.magic["zeta_1"]

    def getStiffnessFactorLong(self):
        self.Kx = self.normalForce * (self.magic["p_kx1"] + self.magic["p_kx2"] * self.normDeltaLoadLong) * (2.71828182845905 ** (self.magic["p_kx3"] * self.normDeltaLoadLong)) * (1 + self.magic["p_px1"] * self.normDeltaPressureLong + self.magic["p_px2"] * self.normDeltaPressureLong ** 2)
        return self.Kx / (self.Cx  * self.Dx)

    def getLongFrictionCoefficient(self):
        term1 = (self.magic["p_dx1"] + self.magic["p_dx2"] * self.normDeltaLoadLong)
        term2 = (1 + self.magic["p_px3"] * self.normDeltaPressureLong + self.magic["p_px4"] * self.normDeltaPressureLong ** 2)
        term3 = (1 - self.magic["p_dx3"] * self.camber ** self.calculateLongCompositeLongFrictionScalingFactor())
        return term1 * term2 * term3

    def calculateLongCompositeLongFrictionScalingFactor(self):
        return self.mechanical["friction-coeff-long"] / (1 + self.mechanical["friction-coeff-long"] * ((self.velocityX ) / math.sqrt(9.81 * self.mechanical["unloaded-radius"])))

    def getCurvatureFactor(self):
        term1 = (self.magic["p_ex1"] + self.magic["p_ex2"] * self.normDeltaLoadLong + self.magic["p_ex3"] * self.normDeltaLoadLong ** 2)
        #print("---------")
        #print(self.slipRatio)
        #print(self.getHorizontalShift())
        #print("---------")
        term2 = (1 - self.magic["p_ex4"] * math.copysign(1,self.slipRatio + self.getHorizontalShift()))
        return term1 * term2 * self.magic["curvature-scaling-factor"]
    def specialDegressiveFrictionFactor(self):
        # A_μ is 10 by what the book suggests so that's what 10 is
        return 10 * self.calculateLongCompositeLongFrictionScalingFactor() / (1 + (9) * self.calculateLongCompositeLongFrictionScalingFactor())

    def getHorizontalShift(self):
        return (self.magic["p_hx1"] + self.magic["p_hx2"] * self.normDeltaLoadLong) * self.magic["horizontal-shift-factor"]

    def getVerticalShift(self):
        return self.normalForce * (self.magic["p_vx1"] + self.magic["p_vx2"] * self.normDeltaLoadLong) * self.magic["vertical-shift-factor"] * self.specialDegressiveFrictionFactor() * self.magic["zeta_1"]



    ##### ********************************
    ##### Standard Functioms
    ##### ********************************

    def stdCurveSine(self, Bx, Cx, Dx, Ex, slip):
        BxSlip = Bx * slip
        return Dx * math.sin( Cx * math.atan( BxSlip - Ex * (BxSlip - math.atan(BxSlip) ) ) )

    #def trainStdCurveSine(self, Bx, Cx, Dx, Ex, slip):
    #    BxSlip = Bx * slip
    #    return Dx * torch.sin( Cx * torch.atan( BxSlip - Ex * (BxSlip - torch.atan(BxSlip) ) ) )

    def normalizeLoadLong(self):
        return (self.normalForce - self.magic["lambda_loadscalarlong"] * self.normalForce) / (self.magic["lambda_loadscalarlong"] * self.normalForce)

    def normalizeLoadLat(self):
        return (self.normalForce - self.magic["lambda_loadscalarlat"] * self.normalForce) / (self.magic["lambda_loadscalarlat"] * self.normalForce)

    def normalizePressureLong(self):
        # Only long because lat doesn't use it
        return (self.tirePressure - self.magic["lambda_pressurescalarlong"] * self.tirePressure) / (self.magic["lambda_pressurescalarlong"] * self.tirePressure)

    def normalizePressureLat(self):
        # Only long because lat doesn't use it
        return (self.tirePressure - self.magic["lambda_pressurescalarlat"] * self.tirePressure) / (self.magic["lambda_pressurescalarlat"] * self.tirePressure)

    def updateParams(self, normalForce=-1, slipRatio=-1, velocityX=-1):
        if self.normalForce != -1:
            self.normalForce = normalForce
        if slipRatio != -1:
            self.slipRatio = slipRatio
        if velocityX != -1:
            self.velocityX = velocityX
