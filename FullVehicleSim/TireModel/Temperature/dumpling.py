# This (hopefully) implements MF 6.x (I think 6.2) as described in Tire and Vehicle Dynamics 3rd edition on page 179
import math
import torch

class Tire:


    def __init__(self, normalForce, slipRatio, slipAngle, velocityX, pressure, temperature, constants, mechanicalParams, magicParams, lat = False, long = False):
        self.normalForce = normalForce * -1 # Difference in TTC Data vs Pacejka
        self.velocityX = velocityX
        self.slipRatioInit = slipRatio
        self.slipRatio = self.slipRatioInit
        self.slipAngle = slipAngle
        self.tirePressure = pressure
        self.tireTemperature = temperature
        self.actPressure = torch.tensor(12) # Actual PSI
        self.camber = torch.tensor(0) # Radians


        self.magic = magicParams
        self.mechanical = mechanicalParams
        self.constants = constants

        if(lat):
            self.normDeltaLoadLat = self.normalizeLoadLat()
            self.normDeltaPressureLat = self.normalizePressureLat()
        if(long):
            self.normDeltaLoadLong = self.normalizeLoadLong()
            self.normDeltaPressureLong = self.normalizePressureLong()

    ##### ********************************
    ##### LATERAL SLIP FUNCTION
    ##### ********************************


    def getLateralForce(self):
        Alphas = self.magic["lambda_alphastar"] * self.slipAngle * torch.sign(self.velocityX)
        Byk = self.magic["r_by1"]# + self.magic["r_by4"] * torch.sin(self.camber) ** 2) * torch.cos(torch.atan(self.magic["r_by2"] * (Alphas - self.magic["r_by3"]))) * self.magic["lambda_yk"]
        Cyk = self.magic["r_cy1"]
        Eyk = self.magic["r_ey1"] + self.magic["r_ey2"] * self.normDeltaLoadLat
        Shyk = self.magic["r_hy1"] + self.magic["r_hy2"] * self.normDeltaLoadLat


        Ks = self.slipRatio + Shyk
        BykKs = Byk * Ks
        BykShyk = Byk * Shyk
        Gykappa = torch.cos(Cyk * torch.atan(BykKs - Eyk * (BykKs - torch.atan(BykKs))))
        Gykappazero =  torch.cos(Cyk * torch.atan(BykShyk - Eyk * (BykShyk - torch.atan(BykShyk))))


        Dvyk = self.mechanical["friction-coeff-lat"] * self.normalForce * (self.magic["r_vy1"] + self.magic["r_vy2"] * self.normDeltaLoadLat + self.magic["r_vy3"] * torch.sin(self.camber)) * torch.cos(torch.atan(self.magic["r_vy4"] * torch.sin(Alphas)))  * self.magic["zeta_2"]
        Svyk = Dvyk * torch.sin(self.magic["r_vy5"] * torch.atan(self.magic["r_vy6"] * self.slipRatio)) * self.magic["lambda_vyk"]

        #print(Byk, Cyk, Eyk, Shyk)

        return -1 * (Gykappa/Gykappazero * self.getLateralForcePure()) #+ Svyk) # + self.magic["Svyk"]




    def getLateralForcePure(self):
        Alphas = self.magic["lambda_alphastarypure"] * self.slipAngle * torch.sign(self.velocityX)

        self.Cy = self.magic["p_cy1"]
        self.Dy = self.getLateralCoefficientOfFriction() * self.normalForce * (-1 * torch.abs(self.magic["tempYAPure"]) * self.tireTemperature ** 2 + self.magic["tempYBPure"] * self.tireTemperature + self.magic["tempYCPure"])
        self.By = self.magic["By_pure"]
        self.Ey = self.getLateralE(Alphas)

        Svy = self.magic["Svy"]
        return self.trainStdCurveSine(self.By, self.Cy, self.Dy, self.Ey, self.slipAngle) + Svy

    def getLateralB(self):
        Kyalpha = self.magic["p_ky1"] * self.normDeltaLoadLat * (1 + self.magic["p_py1"] * self.normDeltaPressureLat) * (1 - self.magic["p_ky3"] * torch.abs(torch.sin(self.camber))) * torch.sin(self.magic["p_ky4"] * torch.atan(1/(self.magic["lambda_nominalload"] * (self.magic["p_ky2"] + self.magic["p_ky5"] * torch.sin(self.camber)**2) * (1+ self.magic["p_py2"] * self.normDeltaPressureLat) ) )) * self.magic["zeta3"] * self.magic["lambda_kyalpha"]
        By = Kyalpha / (self.Cy * self.Dy + self.magic["epsilon_y"])
        return By
    def getLateralCoefficientOfFriction(self):
        return (self.magic["p_dy1"] + self.magic["p_dy2"] * self.normDeltaLoadLat) * (1 + self.magic["p_py3"] * self.normDeltaPressureLat + self.magic["p_py4"] * self.normDeltaPressureLat ** 2) * (1 - self.magic["p_dy3"] * torch.sin(self.camber) ** 2) * self.magic["lambda_coeffscalary"]
    def getLateralE(self, Alphas):
        term1 = (self.magic["p_ey1"] + self.magic["p_ey2"] * self.normDeltaLoadLat)
        term2 = (1 + self.magic["p_ey5"] * torch.sin(self.camber) ** 2 - (self.magic["p_ey3"] + self.magic["p_ey4"] * torch.sin(self.camber)) * Alphas)
        return term1 * term2 * self.magic["lambda_ey"]

    ##### ********************************
    ##### LONGITUDINAL COMBINED SLIP FUNCTIONS
    ##### ********************************


    def getLongForce(self):
        tempScalar = self.magic["tempXA"] * self.tireTemperature ** 2 + self.magic["tempXB"] * self.tireTemperature + self.magic["tempXC"]

        #if self.slipAngle < 0.1 and self.slipAngle > -0.1:
        #    return self.getLongForcePureSlip()

        lF = self.getLongForceCombinedSlip() * tempScalar / self.normalForce

        return torch.where(lF > 5, 5 * self.normalForce, torch.where(lF < -5, -5 * self.normalForce, lF * self.normalForce))


    def getLongForceCombinedSlip(self):
        fCoefficient: torch.Tensor = self.getGxalpha()
        fCoefficient = torch.where(fCoefficient > 1, 1, fCoefficient)
        force = self.getLongForcePureSlip()
        #print(force * fCoefficient)
        #print(fCoefficient)
        return force * (fCoefficient + self.magic["combined_long_offset"])


    def getGxalpha(self):
        Cxalpha = self.magic["r_cx1"]
        Bxalpha = (self.magic["r_bx1"] + self.magic["r_bx3"] * torch.sin(self.camber) ** 2) * torch.cos(torch.atan(self.magic["r_bx2"] * self.slipRatio))  * self.magic["lambda_xalpha"]
        Exalpha = self.magic["r_ex1"] + self.magic["r_ex2"] * self.normDeltaLoadLong
        Shxalpha = self.magic["r_hx1"]
        Alphas = self.magic["lambda_alphastar"] * self.slipAngle * torch.sign(self.velocityX) + Shxalpha
        Alphas = self.magic["lambda_alphastar"] * self.slipAngle * torch.sign(self.velocityX) + Shxalpha

        Gxalpha_init = torch.cos(Cxalpha * torch.atan( Bxalpha * Alphas - Exalpha * ( Bxalpha * Alphas - torch.atan(Bxalpha * Alphas) ) ) )
        Gxalphanaught = torch.cos(Cxalpha * torch.atan( Bxalpha * Shxalpha - Exalpha * ( Bxalpha * Shxalpha - torch.atan(Bxalpha * Shxalpha) ) ) )
        Gxalpha_init = torch.cos(Cxalpha * torch.atan( Bxalpha * Alphas - Exalpha * ( Bxalpha * Alphas - torch.atan(Bxalpha * Alphas) ) ) )
        Gxalphanaught = torch.cos(Cxalpha * torch.atan( Bxalpha * Shxalpha - Exalpha * ( Bxalpha * Shxalpha - torch.atan(Bxalpha * Shxalpha) ) ) )

        return Gxalpha_init / Gxalphanaught  * self.magic["lambda_combinedslipcoeff"]
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

        tempScalarPure = -1 * torch.abs(self.magic["tempXAPure"]) * self.tireTemperature ** 2 + self.magic["tempXBPure"] * self.tireTemperature + self.magic["tempXCPure"]

        self.Cx = self.magic["shape-factor"] # Shape Factor (this thing is entirely magic. I think.) P_cx1
        self.Dx = self.getMaxLongFriction() * tempScalarPure # Peak Factor
        self.Ex = self.getCurvatureFactor() # Curvature Factor
        self.Bx = self.getStiffnessFactorLong() # Stifness Factor

        Svx = self.getVerticalShift()

        longForce = self.trainStdCurveSine(self.Bx, self.Cx, self.Dx, self.Ex, self.slipRatio) + Svx

        #print(self.Bx, self.Cx, self.Dx, self.Ex, Svx)
        # Safety
        #longForce = max(longForce, self.mechanical["friction-coeff-long"] * self.normalForce)
        self.longforce = longForce

        return self.longforce
    def getMaxLongFriction(self):
        # Here there needs to be the mystery camber parameter zeta sub 1.
        return self.getLongFrictionCoefficient() * self.normalForce * self.magic["zeta_1"]

    def getStiffnessFactorLong(self):
        self.Kx = self.normalForce * (self.magic["p_kx1"] + self.magic["p_kx2"] * self.normDeltaLoadLong) * (self.constants["e"] ** (self.magic["p_kx3"] * self.normDeltaLoadLong)) * (1 + self.magic["p_px1"] * self.normDeltaPressureLong + self.magic["p_px2"] * self.normDeltaPressureLong ** 2)
        return self.Kx / (self.Cx  * self.Dx)

    def getLongFrictionCoefficient(self):
        term1 = (self.magic["p_dx1"] + self.magic["p_dx2"] * self.normDeltaLoadLong)
        term2 = (1 + self.magic["p_px3"] * self.normDeltaPressureLong + self.magic["p_px4"] * self.normDeltaPressureLong ** 2)
        term3 = (1 - self.magic["p_dx3"] * self.camber ** self.calculateLongCompositeLongFrictionScalingFactor())
        return term1 * term2 * term3

    def calculateLongCompositeLongFrictionScalingFactor(self):
        return self.mechanical["friction-coeff-long"] / (1 + self.mechanical["friction-coeff-long"] * ((self.velocityX ) / math.sqrt(self.constants["g"] * self.mechanical["unloaded-radius"])))

    def getCurvatureFactor(self):
        term1 = (self.magic["p_ex1"] + self.magic["p_ex2"] * self.normDeltaLoadLong + self.magic["p_ex3"] * self.normDeltaLoadLong ** 2)
        #print("---------")
        #print(self.slipRatio)
        #print(self.getHorizontalShift())
        #print("---------")
        term2 = (1 - self.magic["p_ex4"] * torch.sign(self.slipRatio + self.getHorizontalShift()))
        return term1 * term2 * self.magic["curvature-scaling-factor"]
    def specialDegressiveFrictionFactor(self):
        # A_μ is 10 by what the book suggests so that's what 10 is
        return 10 * self.calculateLongCompositeLongFrictionScalingFactor() / (1 + (9) * self.calculateLongCompositeLongFrictionScalingFactor())

    def getHorizontalShift(self):
        return (self.magic["p_hx1"] + self.magic["p_hx2"] * self.normDeltaLoadLong) * self.magic["horizontal-shift-factor"]

    def getVerticalShift(self):
        return self.normalForce * (self.magic["p_vx1"] + self.magic["p_vx2"] * self.normDeltaLoadLong) * self.magic["vertical-shift-factor"] * self.specialDegressiveFrictionFactor() * self.magic["zeta_1"]




    ##### ********************************
    ##### Standard Functios
    ##### ********************************

    def stdCurveSine(self, Bx, Cx, Dx, Ex, slip):
        BxSlip = Bx * slip
        return Dx * torch.sin( Cx * torch.atan( BxSlip - Ex * (BxSlip - torch.atan(BxSlip) ) ) )

    def trainStdCurveSine(self, Bx, Cx, Dx, Ex, slip):
        BxSlip = Bx * slip
        return Dx * torch.sin( Cx * torch.atan( BxSlip - Ex * (BxSlip - torch.atan(BxSlip) ) ) )

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
