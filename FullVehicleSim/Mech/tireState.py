# This implements MF 6.x (I think 6.2) as described in Tire and Vehicle Dynamics 3rd edition on page 179
# copy pasted from dumping in July 21
import math
from paramLoader import Parameters, Magic

class Tire:
    def __init__(self, normalForce, slipRatio, slipAngle, velocityX, pressure, temperature):
        self.normalForce = normalForce * -1
        self.velocityX = velocityX
        self.slipRatioInit = slipRatio
        self.slipRatio = self.slipRatioInit
        self.slipAngle = slipAngle
        self.tirePressure = pressure
        self.tireTemperature = temperature
        self.actPressure = 12 # Actual PSI
        self.camber = 0 # Radians

        #if(lat):
        self.normDeltaLoadLat = self.normalizeLoadLat()
        self.normDeltaPressureLat = self.normalizePressureLat()
        #if(long):
        self.normDeltaLoadLong = self.normalizeLoadLong()
        self.normDeltaPressureLong = self.normalizePressureLong()

    ##### ********************************
    ##### LATERAL SLIP FUNCTION
    ##### ********************************


    def getLateralForce(self):
        Alphas = Magic["lambda_alphastar"] * self.slipAngle * math.copysign(1, self.velocityX)
        Byk = Magic["r_by1"]# + self.magic["r_by4"] * math.sin(self.camber) ** 2) * math.cos(math.atan(self.magic["r_by2"] * (Alphas - self.magic["r_by3"]))) * self.magic["lambda_yk"]
        Cyk = Magic["r_cy1"]
        Eyk = Magic["r_ey1"] + Magic["r_ey2"] * self.normDeltaLoadLat
        Shyk = Magic["r_hy1"] + Magic["r_hy2"] * self.normDeltaLoadLat

        Ks = self.slipRatio + Shyk
        BykKs = Byk * Ks
        BykShyk = Byk * Shyk
        Gykappa = math.cos(Cyk * math.atan(BykKs - Eyk * (BykKs - math.atan(BykKs))))
        Gykappazero =  math.cos(Cyk * math.atan(BykShyk - Eyk * (BykShyk - math.atan(BykShyk))))


        Dvyk = Parameters["friction-coeff-lat"] * self.normalForce * (Magic["r_vy1"] + Magic["r_vy2"] * self.normDeltaLoadLat + Magic["r_vy3"] * math.sin(self.camber)) * math.cos(math.atan(Magic["r_vy4"] * math.sin(Alphas)))  * Magic["zeta_2"]
        Svyk = Dvyk * math.sin(Magic["r_vy5"] * math.atan(Magic["r_vy6"] * self.slipRatio)) * Magic["lambda_vyk"]

        #print(Byk, Cyk, Eyk, Shyk)

        return Gykappa/Gykappazero * self.getLateralForcePure() #+ Svyk # + self.magic["Svyk"]

    def getLateralForcePure(self):
        Alphas = Magic["lambda_alphastarypure"] * self.slipAngle * math.copysign(1,self.velocityX)

        self.Cy = Magic["p_cy1"]
        self.Dy = self.getLateralCoefficientOfFriction() * self.normalForce * (Magic["tempYAPure"] * self.tireTemperature ** 2 + Magic["tempYBPure"] * self.tireTemperature + Magic ["tempYCPure"])
        self.By = Magic["By_pure"]
        self.Ey = self.getLateralE(Alphas)

        Svy = Magic["Svy"]
        return self.stdCurveSine(self.By, self.Cy, self.Dy, self.Ey, self.slipAngle) + Svy

    def getLateralB(self):
        Kyalpha = Magic["p_ky1"] * self.normDeltaLoadLat * (1 + Magic["p_py1"] * self.normDeltaPressureLat) * (1 - Magic["p_ky3"] * abs(math.sin(self.camber))) * math.sin(Magic["p_ky4"] * math.atan(1/(Magic["lambda_nominalload"] * (Magic["p_ky2"] + Magic["p_ky5"] * math.sin(self.camber)**2) * (1+ Magic["p_py2"] * self.normDeltaPressureLat) ) )) * Magic["zeta3"] * Magic["lambda_kyalpha"]
        By = Kyalpha / (self.Cy * self.Dy + Magic["epsilon_y"])
        return By
    def getLateralCoefficientOfFriction(self):
        return (Magic["p_dy1"] + Magic["p_dy2"] * self.normDeltaLoadLat) * (1 + Magic["p_py3"] * self.normDeltaPressureLat + Magic["p_py4"] * self.normDeltaPressureLat ** 2) * (1 - Magic["p_dy3"] * math.sin(self.camber) ** 2) * Magic["lambda_coeffscalary"]
    def getLateralE(self, Alphas):
        term1 = (Magic["p_ey1"] + Magic["p_ey2"] * self.normDeltaLoadLat)
        term2 = (1 + Magic["p_ey5"] * math.sin(self.camber) ** 2 - (Magic["p_ey3"] + Magic["p_ey4"] * math.sin(self.camber)) * Alphas)
        return term1 * term2 * Magic["lambda_ey"]

    ##### ********************************
    ##### LONGITUDINAL COMBINED SLIP FUNCTIONS
    ##### ********************************

    def getLongForce(self):
        tempScalar = Magic["tempXA"] * self.tireTemperature ** 2 + Magic["tempXB"] * self.tireTemperature + Magic["tempXC"]

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
        return force * min(1,(fCoefficient + Magic["combined_long_offset"]))


    def getGxalpha(self):
        Cxalpha = Magic["r_cx1"]
        Bxalpha = (Magic["r_bx1"] + Magic["r_bx3"] * math.sin(self.camber) ** 2) * math.cos(math.atan(Magic["r_bx2"] * self.slipRatio))  * Magic["lambda_xalpha"]
        Exalpha = Magic["r_ex1"] + Magic["r_ex2"] * self.normDeltaLoadLong
        Shxalpha = Magic["r_hx1"]
        #print("LAMBDA ALPHA STAR", self.magic["lambda_alphastar"], "SA", self.slipAngle)
        Alphas = Magic["lambda_alphastar"] * self.slipAngle * math.copysign(1, self.velocityX) + Shxalpha

        Gxalpha_init = math.cos(Cxalpha * math.atan( Bxalpha * Alphas - Exalpha * ( Bxalpha * Alphas - math.atan(Bxalpha * Alphas) ) ) )
        Gxalphanaught = math.cos(Cxalpha * math.atan( Bxalpha * Shxalpha - Exalpha * ( Bxalpha * Shxalpha - math.atan(Bxalpha * Shxalpha) ) ) )

        #print(Bxalpha, Cxalpha, Exalpha, Shxalpha)

        return Gxalpha_init / Gxalphanaught * Magic["lambda_combinedslipcoeff"]
    ##### ********************************
    ##### Lateral PURE SLIP FUNCTIONS
    ##### ********************************

    def getLatForcePureSlip(self):
        return Parameters["friction-coeff-lat"] * self.normalForce

    ##### ********************************
    ##### LONGITUDINAL PURE SLIP FUNCTIONS
    ##### ********************************


    def getLongForcePureSlip(self):

        #return self.slipRatio * 100

        tempScalarPure = Magic["tempXAPure"] * self.tireTemperature ** 2 + Magic["tempXBPure"] * self.tireTemperature + Magic["tempXCPure"]

        self.Cx = Magic["shape-factor"] # Shape Factor (this thing is entirely magic. I think.) P_cx1
        self.Dx = self.getMaxLongFriction() * tempScalarPure # Peak Factor
        self.Ex = self.getCurvatureFactor() # Curvature Factor
        self.Bx = self.getStiffnessFactorLong() # Stifness Factor

        Svx = self.getVerticalShift()

        longForce = self.stdCurveSine(self.Bx, self.Cx, self.Dx, self.Ex, self.slipRatio) + Svx

        #print(self.Bx, self.Cx, self.Dx, self.Ex, Svx)
        # Safety
        #longForce = max(longForce, Parameters["friction-coeff-long"] * self.normalForce)
        self.longforce = longForce

        return self.longforce
    def getMaxLongFriction(self):
        # Here there needs to be the mystery camber parameter zeta sub 1.
        return self.getLongFrictionCoefficient() * self.normalForce * Magic["zeta_1"]

    def getStiffnessFactorLong(self):
        self.Kx = self.normalForce * (Magic["p_kx1"] + Magic["p_kx2"] * self.normDeltaLoadLong) * (2.71828182845905 ** (Magic["p_kx3"] * self.normDeltaLoadLong)) * (1 + Magic["p_px1"] * self.normDeltaPressureLong + Magic["p_px2"] * self.normDeltaPressureLong ** 2)
        return self.Kx / (self.Cx  * self.Dx)

    def getLongFrictionCoefficient(self):
        term1 = (Magic["p_dx1"] + Magic["p_dx2"] * self.normDeltaLoadLong)
        term2 = (1 + Magic["p_px3"] * self.normDeltaPressureLong + Magic["p_px4"] * self.normDeltaPressureLong ** 2)
        term3 = (1 - Magic["p_dx3"] * self.camber ** self.calculateLongCompositeLongFrictionScalingFactor())
        return term1 * term2 * term3

    def calculateLongCompositeLongFrictionScalingFactor(self):
        return Parameters["friction-coeff-long"] / (1 + Parameters["friction-coeff-long"] * ((self.velocityX ) / math.sqrt(9.81 * Parameters["unloaded-radius"])))

    def getCurvatureFactor(self):
        term1 = (Magic["p_ex1"] + Magic["p_ex2"] * self.normDeltaLoadLong + Magic["p_ex3"] * self.normDeltaLoadLong ** 2)
        #print("---------")
        #print(self.slipRatio)
        #print(self.getHorizontalShift())
        #print("---------")
        term2 = (1 - Magic["p_ex4"] * math.copysign(1,self.slipRatio + self.getHorizontalShift()))
        return term1 * term2 * Magic["curvature-scaling-factor"]
    def specialDegressiveFrictionFactor(self):
        # A_μ is 10 by what the book suggests so that's what 10 is
        return 10 * self.calculateLongCompositeLongFrictionScalingFactor() / (1 + (9) * self.calculateLongCompositeLongFrictionScalingFactor())

    def getHorizontalShift(self):
        return (Magic["p_hx1"] + Magic["p_hx2"] * self.normDeltaLoadLong) * Magic["horizontal-shift-factor"]

    def getVerticalShift(self):
        return self.normalForce * (Magic["p_vx1"] + Magic["p_vx2"] * self.normDeltaLoadLong) * Magic["vertical-shift-factor"] * self.specialDegressiveFrictionFactor() * Magic["zeta_1"]


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
        return (self.normalForce - Magic["lambda_loadscalarlong"] * self.normalForce) / (Magic["lambda_loadscalarlong"] * self.normalForce)

    def normalizeLoadLat(self):
        return (self.normalForce - Magic["lambda_loadscalarlat"] * self.normalForce) / (Magic["lambda_loadscalarlat"] * self.normalForce)

    def normalizePressureLong(self):
        # Only long because lat doesn't use it
        return (self.tirePressure - Magic["lambda_pressurescalarlong"] * self.tirePressure) / (Magic["lambda_pressurescalarlong"] * self.tirePressure)

    def normalizePressureLat(self):
        # Only long because lat doesn't use it
        return (self.tirePressure - Magic["lambda_pressurescalarlat"] * self.tirePressure) / (Magic["lambda_pressurescalarlat"] * self.tirePressure)
    def updateParams(self, normalForce=-1, slipRatio=-1, velocityX=-1):
        if self.normalForce != -1:
            self.normalForce = normalForce
        if slipRatio != -1:
            self.slipRatio = slipRatio
        if velocityX != -1:
            self.velocityX = velocityX
