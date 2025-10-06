# This (hopefully) implements MF 6.x (I think 6.2) as described in Tire and Vehicle Dynamics 3rd edition on page 179
import torch

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

        self.normalForce = self.getNormalLoad(normalForce)


    ##### ********************************
    ##### NORMAL LOAD FUNCTION
    ##### ********************************

    def getNormalLoad(self, inputNormalForce):
        # Neglecting last force
        # I intentionally neglect the last Fx and Fy because that would involve a large rewrite of this.
        sqrt_term = torch.sqrt(torch.tensor(9.81) * self.mechanical["unloaded-radius"])
        term1 = (1 + self.magic["q_v2"] * torch.abs(self.magic["Omega"]) * self.mechanical["unloaded-radius"]/sqrt_term - self.magic["q_Fcx"] - self.magic["q_Fcy"])
        # We assume the deflection is 1 because idk how to do that
        term2 = (self.magic["q_Fz1"] + self.magic["q_Fz2"] * self.camber**2) / self.mechanical["unloaded-radius"]
        term3 = (1 + self.magic["P_pFz1"] * self.normDeltaPressureLong) * inputNormalForce

        return term1 * term2 * term3


    ##### ********************************
    ##### LATERAL SLIP FUNCTION
    ##### ********************************


    def getLateralForce(self):
        Alphas = self.magic["lambda_alphastar"] * self.slipAngle * torch.sign(self.velocityX)
        Byk = self.magic["r_by1"]
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

        return -1 * (Gykappa / Gykappazero * self.getLateralForcePure() + Svyk )# + self.magic["Svyk"]

    def getLateralForcePure(self):

        C_y = self.magic["p_Cy1"]  *self.magic["lambda_Cy"]

        mu_yTerm1 = (self.magic["p_Dy1"] + self.magic["p_Dy2"] * self.normDeltaLoadLat)
        mu_yTerm2 = (1 + self.magic["p_py3"] * self.normDeltaPressureLat + self.magic["p_py4"] * self.normDeltaPressureLat ** 2)
        mu_yTerm3 = (1 - self.magic["p_Dy3"] * torch.sin(self.camber) ** 2) * self.magic["lambda_muy"]
        mu_y = mu_yTerm1 * mu_yTerm2 * mu_yTerm3

        Dy = mu_y * self.normalForce

        E_yTerm1 = (self.magic["p_Ey1"] + self.magic["P_Ey2"] * self.normDeltaLoadLat)
        E_yTerm2 = (1 + self.magic["p_Ey5"] * torch.sin(self.camber) ** 2 - (self.magic["p_Ey3"] + self.magic["p_Ey4"] * torch.sin(self.camber)))
        E_y = E_yTerm1 * E_yTerm2 * self.magic["lambda_Ey"]

        K_yalphaTerm1 = self.magic["p_Ky1"]


    ##### ********************************
    ##### LONGITUDINAL COMBINED SLIP FUNCTIONS
    ##### ********************************

    def getLongForce(self):
        tempScalar = self.magic["tempXA"] * self.tireTemperature ** 2 + self.magic["tempXB"] * self.tireTemperature + self.magic["tempXC"]

        return self.getLongForceCombinedSlip() * tempScalar * -1


    def getLongForceCombinedSlip(self):
        fCoefficient = self.getGxalpha()
        force = self.getLongForcePureSlip()
        coefficient = fCoefficient + self.magic["combined_long_offset"]
        return force * coefficient


    def getGxalpha(self):
        Cxalpha = self.magic["r_cx1"]
        Bxalpha = (self.magic["r_bx1"] + self.magic["r_bx3"] * torch.sin(self.camber) ** 2) * torch.cos(torch.atan(self.magic["r_bx2"] * self.slipRatio))  * self.magic["lambda_xalpha"]
        Exalpha = self.magic["r_ex1"] + self.magic["r_ex2"] * self.normDeltaLoadLong
        Shxalpha = self.magic["r_hx1"]
        Alphas = self.magic["lambda_alphastar"] * self.slipAngle * torch.sign(self.velocityX) + Shxalpha

        Gxalpha_init = torch.cos(Cxalpha * torch.atan( Bxalpha * Alphas - Exalpha * ( Bxalpha * Alphas - torch.atan(Bxalpha * Alphas) ) ) )
        Gxalphanaught = torch.cos(Cxalpha * torch.atan( Bxalpha * Shxalpha - Exalpha * ( Bxalpha * Shxalpha - torch.atan(Bxalpha * Shxalpha) ) ) )

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

        self.longforce = longForce

        return self.longforce
    def getMaxLongFriction(self):
        return self.getLongFrictionCoefficient() * self.normalForce * self.magic["zeta_1"]

    def getStiffnessFactorLong(self):
        exp_term = self.magic["p_kx3"] * self.normDeltaLoadLong
        self.Kx = self.normalForce * (self.magic["p_kx1"] + self.magic["p_kx2"] * self.normDeltaLoadLong) * torch.exp(exp_term) * (torch.tensor(1.0) + self.magic["p_px1"] * self.normDeltaPressureLong + self.magic["p_px2"] * self.normDeltaPressureLong ** 2)
        denominator = self.Cx * self.Dx
        # Add epsilon to prevent division by zero
        eps = torch.tensor(1e-8)
        return self.Kx / (denominator + eps)

    def getLongFrictionCoefficient(self):
        term1 = (self.magic["p_dx1"] + self.magic["p_dx2"] * self.normDeltaLoadLong)
        term2 = (torch.tensor(1.0) + self.magic["p_px3"] * self.normDeltaPressureLong + self.magic["p_px4"] * self.normDeltaPressureLong ** 2)
        scaling_factor = self.calculateLongCompositeLongFrictionScalingFactor()
        camber_term = torch.abs(self.camber) ** scaling_factor
        term3 = (torch.tensor(1.0) - self.magic["p_dx3"] * camber_term)
        return term1 * term2 * term3

    def calculateLongCompositeLongFrictionScalingFactor(self):
        sqrt_term = torch.sqrt(torch.tensor(9.81) * self.mechanical["unloaded-radius"])
        velocity_ratio = abs(self.velocityX) / sqrt_term
        denominator = torch.tensor(1.0) + self.mechanical["friction-coeff-long"] * velocity_ratio
        # Add epsilon to prevent division by zero
        eps = torch.tensor(1e-8)
        return self.mechanical["friction-coeff-long"] / (denominator + eps)

    def getCurvatureFactor(self):
        term1 = (self.magic["p_ex1"] + self.magic["p_ex2"] * self.normDeltaLoadLong + self.magic["p_ex3"] * self.normDeltaLoadLong ** 2)
        term2 = (1 - self.magic["p_ex4"] * torch.sign(self.slipRatio + self.getHorizontalShift()))
        return term1 * term2 * self.magic["curvature-scaling-factor"]
    def specialDegressiveFrictionFactor(self):
        # A_μ is 10 by what the book suggests so that's what 10 is
        return torch.tensor(10.0) * self.calculateLongCompositeLongFrictionScalingFactor() / (torch.tensor(1.0) + torch.tensor(9.0) * self.calculateLongCompositeLongFrictionScalingFactor())

    def getHorizontalShift(self):
        return (self.magic["p_hx1"] + self.magic["p_hx2"] * self.normDeltaLoadLong) * self.magic["horizontal-shift-factor"]

    def getVerticalShift(self):
        return self.normalForce * (self.magic["p_vx1"] + self.magic["p_vx2"] * self.normDeltaLoadLong) * self.magic["vertical-shift-factor"] * self.specialDegressiveFrictionFactor() * self.magic["zeta_1"]


    ##### ********************************
    ##### Standard Functioms
    ##### ********************************

    def stdCurveSine(self, Bx, Cx, Dx, Ex, slip):
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
