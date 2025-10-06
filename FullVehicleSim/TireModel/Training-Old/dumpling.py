# This (hopefully) implements MF 6.x (I think 6.2) as described in Tire and Vehicle Dynamics 3rd edition on page 179
import math
import torch 

class Tire:


    def __init__(self, normalForce, slipRatio, slipAngle, velocityX, constants, mechanicalParams, magicParams):
        self.normalForce = normalForce
        self.velocityX = velocityX
        self.slipRatioInit = slipRatio
        self.slipRatio = self.slipRatioInit
        self.slipAngle = slipAngle
        self.nomPressure = 14 # Nominal PSI
        self.actPressure = 12 # Actual PSI 
        self.camber = 0 # Radians


        self.magic = magicParams
        self.mechanical = mechanicalParams
        self.constants = constants
        
        self.normDeltaLoad = self.normalize(self.mechanical["load_0"], self.normalForce)
        self.normDeltaPressure = self.normalize(self.mechanical["p_0"], self.nomPressure)


    def getLongForce(self):
        if abs(self.slipAngle) < 0.25:
            return self.getLongForcePureSlip() 
        return self.getLongForceCombinedSlip() 

    ##### ********************************
    ##### LATERAL SLIP FUNCTION
    ##### ********************************
    def getLateralForce(self):
        return self.normalForce * self.stdCurveSine(self.magic["By1"], self.magic["Cy1"], self.magic["Dy1"], self.magic["Ey1"], self.slipAngle) + (self.magic["e_one"] * self.slipRatio + self.magic["e_two"]) + self.magic["e_three"]

    ##### ********************************
    ##### LATERAL COMBINED SLIP FUNCTIONS
    ##### ********************************
    """
    # Archived function
    def getLatForceCombinedSlip(self):
        Cyk = 1.9 #self.magic["r_cy1"]
        Byk = 10 #(self.magic["r_by1"] + self.magic["r_by4"] * math.sin(self.camber) ** 2 ) * math.cos( math.atan( self.magic["r_by2"] * (math.tan(self.slipAngle) * math.copysign(1, self.velocityX)) - self.magic["r_by3"]) ) * self.magic["lambda_yk"]
        Shyk = self.magic["r_hy1"] + self.magic["r_hy2"] * self.normDeltaLoad 
        Ks = self.slipRatio# + Shyk
        Eyk = 0.97 #self.magic["r_ey1"] + self.magic["r_ey2"] * self.normDeltaLoad

        numerator = math.cos( Cyk * math.atan( Byk * Ks - Eyk * ( Byk * Ks - math.atan( Byk * Ks ) ) ) )
        denominator = 1 #math.cos( Cyk * math.atan( Byk * Shyk - Eyk * ( Byk * Shyk - math.atan( Byk * Shyk ) ) ) )

        Svyk = 0 # shift is 0 for now

        return ( self.getLatForcePureSlip() * numerator / denominator + Svyk ) * self.magic["lambda_getlatforcecombinedscalar"]
    """



    ##### ********************************
    ##### LONGITUDINAL COMBINED SLIP FUNCTIONS
    ##### ********************************

    def getLongForceCombinedSlip(self):
        fCoefficient = self.getGxalpha()
        force = self.getLongForcePureSlip()
        #print(force * fCoefficient)
        return force * (fCoefficient * force + self.magic["combined_long_offset"]) # This thing is broken idk why i have to multiply by force again
    

    
    
    #**** This version is designed for training purposes ****
    
    def getGxalpha(self):
        Cxalpha = self.magic["r_cx1"]
        Bxalpha = (self.magic["r_bx1"] + self.magic["r_bx3"] * math.sin(self.camber) ** 2) * torch.cos(torch.atan(self.magic["r_bx2"] * self.slipRatio))  * self.magic["lambda_xalpha"]
        Exalpha = self.magic["r_ex1"] + self.magic["r_ex2"] * self.normDeltaLoad
        Shxalpha = self.magic["r_hx1"]
        Alphas =  self.magic["lambda_alphastar"] * self.slipAngle * torch.sign(self.velocityX) + Shxalpha
        
        Gxalpha_init = torch.cos(Cxalpha * torch.atan( Bxalpha * Alphas - Exalpha * ( Bxalpha * Alphas - torch.atan(Bxalpha * Alphas) ) ) )
        Gxalphanaught = torch.cos(Cxalpha * torch.atan( Bxalpha * Shxalpha - Exalpha * ( Bxalpha * Shxalpha - torch.atan(Bxalpha * Shxalpha) ) ) )

        return Gxalpha_init / Gxalphanaught * self.magic["lambda_combinedslipcoeff"]
    """
    
    def getGxalpha(self):
        Cxalpha = self.magic["r_cx1"]
        Bxalpha = (self.magic["r_bx1"] + self.magic["r_bx3"] * math.sin(self.camber) ** 2) * math.cos(math.atan(self.magic["r_bx2"] * self.slipRatio))  * self.magic["lambda_xalpha"]
        Exalpha = self.magic["r_ex1"] + self.magic["r_ex2"] * self.normDeltaLoad
        Shxalpha = self.magic["r_hx1"]
        Alphas =  self.magic["lambda_alphastar"] * self.slipAngle * math.copysign(1, self.velocityX) + Shxalpha
        
        Gxalpha_init = math.cos(Cxalpha * math.atan( Bxalpha * Alphas - Exalpha * ( Bxalpha * Alphas - math.atan(Bxalpha * Alphas) ) ) )
        Gxalphanaught = math.cos(Cxalpha * math.atan( Bxalpha * Shxalpha - Exalpha * ( Bxalpha * Shxalpha - math.atan(Bxalpha * Shxalpha) ) ) )

        return Gxalpha_init / Gxalphanaught * self.magic["lambda_combinedslipcoeff"]
    """

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

        
        self.Cx = self.magic["shape-factor"] # Shape Factor (this thing is entirely magic. I think.)
        self.Dx = self.getMaxLongFriction() # Peak Factor
        self.Ex = self.getCurvatureFactor() # Curvature Factor
        self.Bx = self.magic["stiffness-factor"] #self.getStiffnessFactorLong() # Stifness Factor

        Svx = self.getVerticalShift()

        longForce = self.trainStdCurveSine(self.Bx, self.Cx, self.Dx, self.Ex, self.slipRatio) + Svx
        # Safety
        #longForce = max(longForce, self.mechanical["friction-coeff-long"] * self.normalForce)
        self.longforce = longForce

        #print(self.slipRatio)
        return  longForce  * self.normalForce # Returns force in Newtons

    def getMaxLongFriction(self):
        # Here there needs to be the mystery camber parameter zeta sub 1. 
        return self.getLongFrictionCoefficient() * self.magic["zeta_1"]
    
    def getStiffnessFactorLong(self):
        self.Kx = self.normalForce * (self.magic["p_kx1"] + self.magic["p_kx2"] * self.normDeltaLoad) * (self.constants["e"] ** (self.magic["p_kx3"] * self.normDeltaLoad)) * (1 + self.magic["p_px1"] * self.normDeltaPressure + self.magic["p_px2"] * self.normDeltaPressure ** 2)
        return self.Kx / (self.Cx  * self.Dx)
    
    def getLongFrictionCoefficient(self):
        term1 = (self.magic["p_dx1"] + self.magic["p_dx2"] * self.normDeltaLoad)
        term2 = (1 + self.magic["p_dx3"] * self.normDeltaPressure + self.magic["p_dx4"] * self.normDeltaPressure ** 2)
        term3 = (1 - self.magic["p_dx3"] * self.camber ** 2) 
        return term1 * term2 * term3# * self.calculateLongCompositeLongFrictionScalingFactor()

    def calculateLongCompositeLongFrictionScalingFactor(self):
        return self.mechanical["friction-coeff-long"] / (1 + self.mechanical["friction-coeff-long"] * ((self.slipRatio * self.velocityX * -1) / math.sqrt(self.constants["g"] * self.mechanical["unloaded-radius"])))

    def getCurvatureFactor(self):
        term1 = (self.magic["p_ex1"] + self.magic["p_ex2"] * self.normDeltaLoad + self.magic["p_ex3"] * self.normDeltaLoad ** 2)
        #print("---------")
        #print(self.slipRatio)
        #print(self.getHorizontalShift())
        #print("---------")
        term2 = (1 - self.magic["p_ex4"] * torch.sign(self.slipRatio + self.getHorizontalShift()))
        #term2 = (1 - self.magic["p_ex4"] * math.copysign(1, self.slipRatio + self.getHorizontalShift()))
        return term1 * term2 * self.magic["curvature-scaling-factor"]
    def specialDegressiveFrictionFactor(self):
        # A_μ is 10 by what the book suggests so that's what 10 is
        return 10 * self.calculateLongCompositeLongFrictionScalingFactor() / (1 + (9) * self.calculateLongCompositeLongFrictionScalingFactor())

    def getHorizontalShift(self):
        return (self.magic["p_hx1"] + self.magic["p_hx2"] * self.normDeltaLoad) * self.magic["horizontal-shift-factor"]
    
    def getVerticalShift(self):
        return self.normalForce * (self.magic["p_vx1"] + self.magic["p_vx2"] * self.normDeltaLoad) * self.magic["vertical-shift-factor"] * self.specialDegressiveFrictionFactor() * self.magic["zeta_1"]



    ##### ********************************
    ##### Standard Functios
    ##### ********************************

    def stdCurveSine(self, Bx, Cx, Dx, Ex, slip):
        BxSlip = Bx * slip
        return Dx * math.sin( Cx * math.atan( BxSlip - Ex * (BxSlip - math.atan(BxSlip) ) ) )

    def trainStdCurveSine(self, Bx, Cx, Dx, Ex, slip):
        BxSlip = Bx * slip
        return Dx * torch.sin( Cx * torch.atan( BxSlip - Ex * (BxSlip - torch.atan(BxSlip) ) ) )

    def normalize(self, init, new):
        return 0 #(new - init)/init
    
    def updateParams(self, normalForce=-1, slipRatio=-1, velocityX=-1):
        if self.normalForce != -1:
            self.normalForce = normalForce
        if slipRatio != -1:
            self.slipRatio = slipRatio
        if velocityX != -1:
            self.velocityX = velocityX