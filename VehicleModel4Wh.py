# Important Packages
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import integrate
from scipy.interpolate import interp1d


# Class Vehicle with all the functions
class Vehicle:
    
    """The Vehicle class encapsulates a variety of methods and properties related to vehicle dynamics, steering, wheel loads, and geometric calculations.
     It leverages numerical methods, regressions, and physical equations to model and analyze vehicle behavior under different conditions.
     The class is designed to provide comprehensive insights into the performance and characteristics of a vehicle's steering and suspension systems.
    """
    # --- Constuctor for Inputs ---

    def __init__(self,    
             r_A: np.array,
             r_B: np.array,
             r_C: np.array,
             r_O: np.array,
             r_K: np.array,
             slr: float,
             dlr: float,
             initial_camber: float,
             toe_in: float,
             tw: float,
             wb: float,
             GVW: float,
             b: float,
             CG_height: float,
             wheel_rate_f: float,
             wheel_rate_r: float,
             tire_stiffness_f: float,
             tire_stiffness_r: float,
             pinion: float,
             tirep: float,
             dila: float,
             assumed_rack_stroke: float,
             r_La: np.array,
             r_Lb: np.array,
             r_strut: np.array = np.array([0, 0, 0]),
             r_Ua: np.array = np.array([0, 0, 0]),
             r_Ub: np.array = np.array([0, 0, 0]),
             mu : float = 0.5,
             g : float = 9.8,
             speed: float = 10.0,
             linkage_effort: float = 1.36, # Nm
             linkage_kpm: float = 7.0, # Nm/deg
             I_w: float = 0.34,
             I_ss: float = 0.03,
             tiredata: np.array =  np.array([0.5094636099593582, 0.1120749440478134, 17.8337673155644, 0.4054933824758519, 0.25184969239087557, 5.904032519832173, 0.5968391994177625, 0.309857379732586 ]),
             CF_Loads: np.array = np.array([0, 150, 200, 250, 500]),
             CF_Stiffnessrad: np.array = np.array([0, 20234.57749,	23031.75745, 24629.16378, 24629.16378 + 250*(24629.16378-23031.75745)/50]),
             CF_pneumatictrail: np.array = np.array([0, 0.011909253,	0.018484467, 0.023331694, 0.023331694 + 250*(0.023331694-0.018484467)/50])): # Continental R13

        # Create static object
        self.mu = mu
        self.g = g
        self.tw = tw
        self.wb = wb
        self.b = b
        self.a = wb - b
        self.GVW = GVW
        self.FAW = GVW*b/wb
        self.RAW = GVW*self.a/wb
        self.CF_Factor = 1
        self.align_factor = 1
        self.thetaforcamber = 0
        self.assumed_rack_stroke = assumed_rack_stroke
        self.pinion = pinion
        self.speed = speed*5/18 #m/s
        self.static = Vehicle.create_object(r_A, r_B, r_C, r_O, r_K, slr, initial_camber, toe_in, 
                                          CG_height, wheel_rate_f, wheel_rate_r, tire_stiffness_f, tire_stiffness_r,
                                            tirep, r_La, r_Lb, r_strut, r_Ua, r_Ub, tiredata, speed)

        # Create dynamic object 
        self.dynamic = Vehicle.create_object(r_A, r_B, r_C, r_O, r_K, dlr, initial_camber, toe_in,
                                           CG_height, wheel_rate_f, wheel_rate_r, tire_stiffness_f, tire_stiffness_r,
                                           tirep, r_La, r_Lb, r_strut, r_Ua, r_Ub, tiredata, speed)
        # Initialize common parameters
        self.I_w = I_w
        self.I_ss = I_ss
        self.CF_Loads = CF_Loads
        self.CF_Stiffnessrad = CF_Stiffnessrad
        self.CF_pneumatictrail = CF_pneumatictrail
        self.dynamic_analysis = 1
        self.model = self.regression_model()
        self.dynamic_analysis = 0
        reference = self.reference()
        self.model = self.regression_model()
        self.rack_stroke = self.rack_vs_road_steer(dila - toe_in)
        
        self.slipangles = np.zeros((50, 2))
        self.slipangles[0] = np.array([0,0])
        self.Flguess = np.zeros((50))
        self.Frguess = np.zeros((50))
        self.Rlguess = np.zeros((50))
        self.Rrguess = np.zeros((50))
        self.Flguess[0] = self.FAW/self.FAW/2
        self.Frguess[0] = self.Flguess[0]
        self.Rlguess[0] = self.RAW/self.FAW/2
        self.Rrguess[0] = self.Rlguess[0]
        self.patch_radius_left = 0
        self.patch_radius_right = 0
        self.tempdynamicsolution = np.zeros(12)
        self.tempdynamictheta = 0
        self.move = 0.01
        self.trainslipangles()
        self.linkage_friction_contribution_on_steering = linkage_effort   
    @classmethod
    def create_object(cls, r_A, r_B, r_C, r_O, r_K, tire_radius, initial_camber, toe_in, CG_height, 
                    wheel_rate_f, wheel_rate_r, tire_stiffness_f, tire_stiffness_r, tirep,
                    r_La, r_Lb, r_strut, r_Ua, r_Ub, tiredata,speed):
        
        obj = type('VehicleState', (), {})()
        
        # Assign instance variables
        obj.r_A = r_A
        obj.r_B = r_B 
        obj.r_C = r_C
        obj.r_O = r_O
        obj.r_K = r_K
        obj.tire_radius = tire_radius
        obj.initial_camber = initial_camber
        obj.Kf = wheel_rate_f * tire_stiffness_f / (wheel_rate_f + tire_stiffness_f)
        obj.Kr = wheel_rate_r * tire_stiffness_r / (wheel_rate_r + tire_stiffness_r)
        obj.tiredata = tiredata
        obj.tirep = tirep
        obj.r_La = r_La
        obj.r_Lb = r_Lb
        obj.r_strut = r_strut
        obj.r_Ua = r_Ua
        obj.r_Ub = r_Ub

        # Calculate additional points
        obj.r_D = np.array([obj.r_C[0], 0.00, obj.r_C[2]])
        obj.r_T = np.array([obj.r_O[0], obj.r_O[1] - obj.tire_radius * np.sin(np.radians(obj.initial_camber)),
                            obj.r_O[2] - obj.tire_radius])
        obj.r_O[2] = obj.r_O[2] - obj.tire_radius + obj.tire_radius * np.cos(np.radians(obj.initial_camber))
        obj.r_W = obj.r_O - np.array([-np.sin(np.radians(toe_in)), np.cos(np.radians(toe_in)), 0])

        # Calculate KPA
        obj.KPA = (r_A - r_K) / cls.magnitude(r_A - r_K)
        obj.currKPA = (r_A - r_K) / cls.magnitude(r_A - r_K)

        # Initialize arrays
        obj.mindp = 50
        obj.maxdp = 50
        obj.step = 0.1
        obj.dpK = np.zeros((int(obj.mindp / obj.step + obj.maxdp / obj.step + 1), 3))
        obj.dpT = np.zeros((int(obj.mindp / obj.step + obj.maxdp / obj.step + 1), 3))
        obj.dpO = np.zeros((int(obj.mindp / obj.step + obj.maxdp / obj.step + 1), 3))
        obj.dpW = np.zeros((int(obj.mindp / obj.step + obj.maxdp / obj.step + 1), 3))
        obj.dpA = np.zeros((int(obj.mindp / obj.step + obj.maxdp / obj.step + 1), 3))
        obj.dpB = np.zeros((int(obj.mindp / obj.step + obj.maxdp / obj.step + 1), 3))
        obj.dpnewB = np.zeros((int(obj.mindp / obj.step + obj.maxdp / obj.step + 1), 3))
        obj.dpC = np.zeros((int(obj.mindp / obj.step + obj.maxdp / obj.step + 1), 3))
        obj.dpdz = np.zeros((int(obj.mindp / obj.step + obj.maxdp / obj.step + 1)))
        obj.dpfvsa = np.zeros((int(obj.mindp / obj.step + obj.maxdp / obj.step + 1), 3))
        obj.dpsvsa = np.zeros((int(obj.mindp / obj.step + obj.maxdp / obj.step + 1), 3))
        obj.zeropos = int(obj.mindp / obj.step)

        # Set initial positions
        obj.dpK[obj.zeropos] = obj.r_K
        obj.dpO[obj.zeropos] = obj.r_O  
        obj.dpT[obj.zeropos] = obj.r_T
        obj.dpW[obj.zeropos] = obj.r_W
        obj.dpA[obj.zeropos] = obj.r_A
        obj.dpB[obj.zeropos] = obj.r_B
        obj.dpnewB[obj.zeropos] = obj.r_B
        obj.dpC[obj.zeropos] = obj.r_C
        obj.dpdz[obj.zeropos] = 0

        # Calculate angles
        h1 = (obj.KPA - np.dot(obj.KPA, np.array([0, 1, 0])) * np.array([0, 1, 0])) / cls.magnitude(
            obj.KPA - np.dot(obj.KPA, np.array([0, 1, 0])) * np.array([0, 1, 0]))
        h2 = (obj.KPA - np.dot(obj.KPA, np.array([1, 0, 0])) * np.array([1, 0, 0])) / cls.magnitude(
            obj.KPA - np.dot(obj.KPA, np.array([1, 0, 0])) * np.array([1, 0, 0]))
        obj.caster = np.degrees(np.arccos(np.dot(h1, np.array([0, 0, 1]))))
        obj.kpi = np.degrees(np.arccos(np.dot(h2, np.array([0, 0, 1]))))

        # Calculate projection points
        t = (obj.r_T[2] - obj.r_K[2]) / obj.KPA[2]
        obj.r_I = obj.r_K + t * obj.KPA
        obj.r_Aprime = cls.projection(obj.r_A, obj.KPA, obj.r_B)
        obj.r_Iprime = cls.projection(obj.r_A, obj.KPA, obj.r_T)
        obj.r_Ioprime = cls.projection(obj.r_A, obj.KPA, obj.r_O)

        obj.maxdecimal = int(-np.log10(obj.step))
        obj.conversionstep = int(10**obj.maxdecimal)
        # Initializing additional helper variables and methods
        obj.CG_height = CG_height



        return obj        
    def reference(self):
        if(self.dynamic_analysis == 0):
            reference = self.static
        else:
            reference = self.dynamic
        return reference
    # --- Calculation of instantaneous axis for suspension travel ---
    def fvsa_equations(self, values):
        """
        Calculates the equations for Front View Swing Arm (FVSA) optimization.

        Computes the difference between two vectors based on vehicle geometry and steering parameters
        to find optimal values of `la` and `mu`. Depending on whether `r_strut` is defined, calculates
        equations for suspension parameters affecting FVSA optimization.

        Args:
        values (list or tuple): Contains two float values representing:
            - `la`: Parameter affecting the vector calculation based on current_A and current_K.
            - `mu`: Parameter affecting the vector calculation based on current_K and average of r_La and r_Lb.

        Returns:
        list: A list containing two equations (`eq1` and `eq2`) representing the difference between `l2` and `l1`.
            - `eq1`: Difference in the y-component between `l2` and `l1`.
            - `eq2`: Difference in the z-component between `l2` and `l1`.

        Notes:
        - If `r_strut` is not defined (equal to [0, 0, 0]), calculates `a2` based on average of r_Ua and r_Ub.
        - If `r_strut` is defined, calculates `a2` based on current_A and cross product of r_strut-a1 and [1, 0, 0].
        - `current_A`, `current_K`, and `current_O` are calculated using `self.curr_A`, `self.curr_K`, and `self.curr_O`
        methods respectively, with `self.curr_KPA_angle_for_fvsa` as input.
        """
        reference = self.reference()
        if(reference.r_strut[0] == 0): # No strut present
            la = values[0]
            mu = values[1]
            current_A = self.curr_A(self.curr_KPA_angle_for_fvsa)
            current_K = self.curr_K(self.curr_KPA_angle_for_fvsa)
            if reference.r_strut[0] == 0:
                # No strut present
                if(np.abs(reference.r_Ua[0] - reference.r_Ub[0])<1 and np.abs(reference.r_Ua[2] - reference.r_Ub[2])<1) :
                    a1 = reference.r_Ua
                    a2 = reference.r_Ub
                    b1 = reference.r_La
                    b2 = reference.r_Lb
                    l1 = a1 + la * (a1 - a2)
                    l2 = b1 + mu * (b1 - b2)
                else:
                    a1 = current_A
                    a2 = (reference.r_Ua + reference.r_Ub) / 2
                    b1 = current_K
                    b2 = (reference.r_La + reference.r_Lb) / 2
                    l1 = a1 + la * (a1 - a2)
                    l2 = b1 + mu * (b1 - b2)  
            l1 = a1 + la*(a1-a2)
            l2 = b1 + mu*(b1-b2)
            eq1 = (l2-l1)[1]
            eq2 = (l2-l1)[2]
        else:
            la = values[0]
            mu = values[1]
            current_A = self.curr_A(self.curr_KPA_angle_for_fvsa)
            current_K = self.curr_K(self.curr_KPA_angle_for_fvsa)
            current_O = self.curr_O(self.curr_KPA_angle_for_fvsa)
            a1 = current_A
            a2 = a1+np.cross(reference.r_strut-a1, np.array([1,0,0]))
            b1 = current_K
            b2 = (reference.r_La+reference.r_Lb)/2
            a2 += 1e-9
            b2 += 1e-9
            l1 = a1 + la*(a1-a2)
            l2 = b1 + mu*(b1-b2)
            eq1 = (l2-l1)[1]
            eq2 = (l2-l1)[2]
        return [eq1,eq2]
    def fvsa_ic(self, curr_KPA_angle):
        """
        Computes the Instantaneous Centers (IC) for the Front View Swing Arm (FVSA) suspension.

        This method calculates the IC based on the current KPA angle and the geometry of the suspension.

        Args:
        curr_KPA_angle (float): Current Kingpin Axis (KPA) angle in degrees.

        Returns:
        ndarray: Coordinates of the Instantaneous Center (IC) in 3D space.

        Notes:
        - Uses numerical root-finding (fsolve) to staticsolve the FVSA equations for la and mu.
        - Handles different configurations based on the presence of a strut.
        """
        reference = self.reference()
        self.curr_KPA_angle_for_fvsa = curr_KPA_angle
        position_to_add = reference.zeropos + int(np.round(curr_KPA_angle, reference.maxdecimal) * reference.conversionstep)
        if(np.abs(reference.dpfvsa[position_to_add][0])>np.abs(reference.dpfvsa[reference.zeropos][0]/10000)):
            return reference.dpfvsa[position_to_add]
        if reference.r_strut[0] == 0:
            # No strut present
            
            current_A = self.curr_A(curr_KPA_angle)
            current_K = self.curr_K(curr_KPA_angle)
            if(np.abs(reference.r_Ua[0] - reference.r_Ub[0])<1 and np.abs(reference.r_Ua[2] - reference.r_Ub[2])<1) :
                reference.dpfvsa[position_to_add] = self.svsa_ic(curr_KPA_angle) + np.array([0,1,0])
                return reference.dpfvsa[position_to_add]
            else:
                la, mu = fsolve(self.fvsa_equations, [0.01, 0.01])
                a1 = current_A
                a2 = (reference.r_Ua + reference.r_Ub) / 2
                b1 = current_K
                b2 = (reference.r_La + reference.r_Lb) / 2
                l1 = a1 + la * (a1 - a2)
                l2 = b1 + mu * (b1 - b2)   
        else:
            # Strut present
            la, mu = fsolve(self.fvsa_equations, [0.01, 0.01])
            current_A = self.curr_A(curr_KPA_angle)
            current_K = self.curr_K(curr_KPA_angle)
            current_O = self.curr_O(curr_KPA_angle)
            a1 = current_A
            a2 = a1 + np.cross(reference.r_strut - a1, np.array([1, 0, 0]))
            b1 = current_K
            b2 = (reference.r_La + reference.r_Lb) / 2
            l1 = a1 + la * (a1 - a2)
            l2 = b1 + mu * (b1 - b2)
        reference.dpfvsa[position_to_add] = (l1 + l2) / 2

        return reference.dpfvsa[position_to_add]
    def svsa_equations(self, values):
        """
        Calculates the Side View Swing Arm (SVSA) suspension equations for finding la and mu.

        This method computes the equations based on the current configuration of the suspension.
        For configurations without a strut, it uses the upper (Ua, Ub) and lower (La, Lb) control arm pivot points.
        For configurations with a strut, it adjusts the calculation based on the strut position relative to the upper pivot.

        Args:
        values (list): List containing la and mu values to staticsolve the equations.

        Returns:
        list: Equations [eq1, eq2] representing the difference between computed lengths l2 and l1 along x and z axes.

        Notes:
        - Uses current KPA angle for calculating current_A.
        - Handles different suspension configurations based on the presence of a strut (r_strut).
        """
        reference = self.reference()
        if reference.r_strut[0] == 0:
            la = values[0]
            mu = values[1]
            # No strut present
            current_A = self.curr_A(self.curr_KPA_angle_for_svsa)
            current_K = self.curr_K(self.curr_KPA_angle_for_svsa)
            if(np.abs(reference.r_Ua[0] - reference.r_Ub[0])<1 and np.abs(reference.r_Ua[2] - reference.r_Ub[2])<1) :
                a1 = current_A
                a2 = (reference.r_Ua + reference.r_Ub) / 2
                b1 = current_K
                b2 = (reference.r_La + reference.r_Lb) / 2
                l1 = a1 + la * (a1 - a2)
                l2 = b1 + mu * (b1 - b2)
            else:
                a1 = reference.r_Ua
                a2 = reference.r_Ub
                b1 = reference.r_La
                b2 = reference.r_Lb
                l1 = a1 + la * (a1 - a2)
                l2 = b1 + mu * (b1 - b2)
            eq1 = (l2 - l1)[0]
            eq2 = (l2 - l1)[2]
        else:
            # Strut present
            la = values[0]
            mu = values[1]
            current_A = self.curr_A(self.curr_KPA_angle_for_svsa)
            a1 = current_A
            a2 = a1 + np.cross(reference.r_strut - a1, np.array([0, 1, 0]))
            b1 = reference.r_La
            b2 = reference.r_Lb
            a2 += 1e-9
            b2 += 1e-9
            l1 = a1 + la * (a1 - a2)
            l2 = b1 + mu * (b1 - b2)
            eq1 = (l2 - l1)[0]
            eq2 = (l2 - l1)[2]
        
        return [eq1, eq2]
    def svsa_ic(self, curr_KPA_angle):
        """
        Computes the Instantaneous Centers (IC) for the Side View Swing Arm (SVSA) suspension.

        This method calculates the IC based on the current configuration of the SVSA suspension.
        If no strut is present, it uses the upper (Ua, Ub) and lower (La, Lb) control arm pivot points.
        If a strut is present, it adjusts the calculation based on the strut position relative to the upper pivot.

        Args:
        curr_KPA_angle (float): Current Kingpin Axis (KPA) angle in radians.

        Returns:
        ndarray: Coordinates of the IC (Instantaneous Center) calculated as the midpoint of lengths l1 and l2.

        Notes:
        - Uses fsolve to staticsolve the svsa_equations for la and mu.
        - Handles different suspension configurations based on the presence of a strut (r_strut).
        """
        reference = self.reference()
        self.curr_KPA_angle_for_svsa = curr_KPA_angle
        position_to_add = reference.zeropos + int(np.round(curr_KPA_angle, reference.maxdecimal) * reference.conversionstep)
        if(reference.dpsvsa[position_to_add][0]>reference.dpsvsa[reference.zeropos][0]/10):
            return reference.dpsvsa[position_to_add]
        if reference.r_strut[0] == 0:
            # No strut present            
            [la, mu] = fsolve(self.svsa_equations, [0.01, 0.01])
            current_A = self.curr_A(self.curr_KPA_angle_for_svsa)
            current_K = self.curr_K(curr_KPA_angle)
            if(np.abs(reference.r_Ua[0] - reference.r_Ub[0])<1 and np.abs(reference.r_Ua[2] - reference.r_Ub[2])<1) :
                a1 = current_A
                a2 = (reference.r_Ua + reference.r_Ub) / 2
                b1 = current_K
                b2 = (reference.r_La + reference.r_Lb) / 2
                l1 = a1 + la * (a1 - a2)
                l2 = b1 + mu * (b1 - b2)
            else:
                a1 = reference.r_Ua
                a2 = reference.r_Ub
                b1 = reference.r_La
                b2 = reference.r_Lb
                l1 = a1 + la * (a1 - a2)
                l2 = b1 + mu * (b1 - b2)
        else:
            # Strut present
            [la, mu] = fsolve(self.svsa_equations, [0.01, 0.01])
            current_A = self.curr_A(self.curr_KPA_angle_for_svsa)
            a1 = current_A
            a2 = a1 + np.cross(reference.r_strut - a1, np.array([0, 1, 0]))
            b1 = reference.r_La
            b2 = reference.r_Lb
            l1 = a1 + la * (a1 - a2)
            l2 = b1 + mu * (b1 - b2)
        reference.dpsvsa[position_to_add] = (l1 + l2) / 2
        
        return reference.dpsvsa[position_to_add]
    def curr_K(self, curr_KPA_angle):
        """
        Computes the position of point K based on the current KPA angle.

        This method calculates the position of point K along the suspension axis based on the provided current
        Kingpin Axis (KPA) angle. If the angle is zero, it returns the initial position of point K. Otherwise,
        it adjusts the position using the stored positions and angles of various suspension components and ICs.

        Args:
        curr_KPA_angle (float): Current Kingpin Axis (KPA) angle in degrees.

        Returns:
        ndarray: The current position of point K in 3D space.

        Notes:
        - If `curr_KPA_angle` is zero, returns `reference.r_K`.
        - Uses the stored positions (`dpK`) and ICs (`fvsa_ic` and `svsa_ic`) to compute the current position of K.
        - Adjusts positions based on the sign of `curr_KPA_angle` for accuracy.
        """
        reference = self.reference()
        if curr_KPA_angle == 0:
            return reference.r_K
        position_to_add = reference.zeropos + int(np.round(curr_KPA_angle, reference.maxdecimal) * reference.conversionstep)

        # Adjust position based on stored data and IC calculations
        if reference.dpK[position_to_add][0] < reference.r_K[0] / 10:
            self.curr_KPA_angle_for_T = curr_KPA_angle
            self.old_O = Vehicle.rotation(reference.dpO[position_to_add-int(np.sign(curr_KPA_angle))].tolist(), self.curr_A(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(),self.curr_K(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(), np.sign(curr_KPA_angle)*reference.step)
            self.old_W = Vehicle.rotation(reference.dpW[position_to_add-int(np.sign(curr_KPA_angle))].tolist(), self.curr_A(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(),self.curr_K(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(), np.sign(curr_KPA_angle)*reference.step)
            
            [t] = fsolve(self.solveO, [0.01], xtol = 0.001)
            reference.dpK[position_to_add] = Vehicle.rotation(reference.dpK[position_to_add - int(np.sign(curr_KPA_angle))].tolist(),
                                                        self.fvsa_ic(curr_KPA_angle - np.sign(curr_KPA_angle) * reference.step).tolist(),
                                                        self.svsa_ic(curr_KPA_angle - np.sign(curr_KPA_angle) * reference.step).tolist(), t)

        return reference.dpK[position_to_add]
    def curr_KPA(self, curr_KPA_angle):
        """
        Computes the current Kingpin Axis (KPA) based on the given KPA angle.

        This method calculates the current Kingpin Axis (KPA) by determining the vector difference between
        the current position of point A (upper ball joint) and point K (lower ball joint) at the given
        Kingpin Axis angle. It normalizes this vector to obtain the direction of the KPA.

        Args:
        curr_KPA_angle (float): The current Kingpin Axis (KPA) angle in degrees.

        Returns:
        ndarray: The normalized direction vector of the current Kingpin Axis (KPA).

        Notes:
        - The method updates the instance variable `reference.currKPA` to the computed KPA direction.
        - Uses the methods `curr_A` and `curr_K` to obtain the current positions of points A and K.
        """
        reference = self.reference()
        t = self.curr_A(curr_KPA_angle) - self.curr_K(curr_KPA_angle)
        m = t / Vehicle.magnitude(t)
        reference.currKPA = m
        return m
    def curr_A(self, curr_KPA_angle):
        """
        Determines the current position of point A (upper ball joint) at the specified Kingpin Axis (KPA) angle.

        This method calculates the position of point A by solving for the rotation parameter `t` if the position
        has not been previously computed for the given KPA angle. It updates the position in the dpA array.

        Args:
        curr_KPA_angle (float): The current Kingpin Axis (KPA) angle in degrees.

        Returns:
        ndarray: The position of point A at the specified KPA angle.

        Notes:
        - If the `curr_KPA_angle` is zero, it returns the initial position `r_A`.
        - The method checks if the position for the given angle has already been computed.
        - If not, it solves for the rotation parameter `t` using the `solveO` method.
        - The position is then updated in the `dpA` array using the `rotation` method.
        """
        reference = self.reference()
        
        if np.abs(curr_KPA_angle) < 10e-4:    
            return reference.r_A
        position_to_add = reference.zeropos + int(np.round(curr_KPA_angle, reference.maxdecimal) * reference.conversionstep)
        
        if reference.dpA[position_to_add][0] < reference.r_A[0] / 100000:
            self.curr_KPA_angle_for_T = curr_KPA_angle
            self.old_O = Vehicle.rotation(reference.dpO[position_to_add-int(np.sign(curr_KPA_angle))].tolist(), self.curr_A(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(),self.curr_K(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(), np.sign(curr_KPA_angle)*reference.step)
            self.old_W = Vehicle.rotation(reference.dpW[position_to_add-int(np.sign(curr_KPA_angle))].tolist(), self.curr_A(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(),self.curr_K(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(), np.sign(curr_KPA_angle)*reference.step)
            
            [t] = fsolve(self.solveO, [0.01], xtol = 0.001)
            reference.dpA[position_to_add] = Vehicle.rotation(
                reference.dpA[position_to_add - int(np.sign(curr_KPA_angle))].tolist(),
                self.fvsa_ic(curr_KPA_angle - np.sign(curr_KPA_angle) * reference.step).tolist(),
                self.svsa_ic(curr_KPA_angle - np.sign(curr_KPA_angle) * reference.step).tolist(),
                t
            )
        return reference.dpA[position_to_add]
    def curr_I(self, curr_KPA_angle):
        reference = self.reference()
        currK = self.curr_K(curr_KPA_angle)
        currT = self.curr_T(curr_KPA_angle)
        currKPA =  self.curr_KPA(curr_KPA_angle)
        t = (currT[2] - currK[2]) / currKPA[2]
        r_I = currK + t * currKPA
        return r_I
# --- Projection of point (x,y,z) on the plane a*x + b*y + c*z = 1 --- 
    def project_points(x, y, z, a, b, c):
        """
        Projects the points with coordinates x, y, z onto the plane
        defined by a*x + b*y + c*z = 1
        """
        vector_norm = a*a + b*b + c*c
        normal_vector = np.array([a, b, c]) / np.sqrt(vector_norm)
        point_in_plane = np.array([a, b, c]) / vector_norm

        points = np.column_stack((x, y, z))
        points_from_point_in_plane = points - point_in_plane
        proj_onto_normal_vector = np.dot(points_from_point_in_plane,
                                        normal_vector)
        proj_onto_plane = (points_from_point_in_plane -
                        proj_onto_normal_vector[:, None]*normal_vector)

        return point_in_plane + proj_onto_plane
    # --- Magnitude of a vector ---
    def magnitude(vector):
        return np.sqrt(sum(pow(element, 2) for element in vector))
    # --- Matrix Multiplication ---
    @staticmethod
    def safe_normalize(U):
        norm = np.linalg.norm(U)
        if norm == 0:
            return np.zeros_like(U)
        return U / norm

    @staticmethod
    def matrix_multiply(*matrices):
        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.dot(result, matrix)
        return result
    # --- Rotaion of point p at an angle t about the axis defined by points x1,x2 ---
    def rotation(p, x1, x2, t):
        theta = np.radians(t)
        p = np.array([[pp] for pp in p] + [[1]])
        x1, y1, z1 = x1
        x2, y2, z2 = x2

        # Define the unit vector U along the axis of rotation
        U = np.array([x2 - x1, y2 - y1, z2 - z1])
        U = Vehicle.safe_normalize(U)
        a, b, c = U

        d = np.sqrt(b**2 + c**2)
        if d == 0:
            d = 1e-9  # Handle case where b and c are both zero to avoid division by zero

        # Translation matrices
        T = np.array([
            [1, 0, 0, -x1],
            [0, 1, 0, -y1],
            [0, 0, 1, -z1],
            [0, 0, 0, 1]
        ])
        T_inv = np.array([
            [1, 0, 0, x1],
            [0, 1, 0, y1],
            [0, 0, 1, z1],
            [0, 0, 0, 1]
        ])

        # Rotation matrices around x, y, and z axes
        R_x = np.array([
            [1, 0, 0, 0],
            [0, c / d, -b / d, 0],
            [0, b / d, c / d, 0],
            [0, 0, 0, 1]
        ])
        R_x_inv = np.array([
            [1, 0, 0, 0],
            [0, c / d, b / d, 0],
            [0, -b / d, c / d, 0],
            [0, 0, 0, 1]
        ])

        R_y = np.array([
            [d, 0, -a, 0],
            [0, 1, 0, 0],
            [a, 0, d, 0],
            [0, 0, 0, 1]
        ])
        R_y_inv = np.array([
            [d, 0, a, 0],
            [0, 1, 0, 0],
            [-a, 0, d, 0],
            [0, 0, 0, 1]
        ])

        # Rotation matrix around z-axis
        ct = np.cos(theta)
        st = np.sin(theta)
        R_z = np.array([
            [ct, st, 0, 0],
            [-st, ct, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Composite transformation
        p_transformed = Vehicle.matrix_multiply(T_inv, R_x_inv, R_y_inv, R_z, R_y, R_x, T, p)

        return p_transformed[:3, 0]
    # --- Projection of a point given normal and point on plane ---
    def projection(point, normal, point_on_plane):
        """
        Projects the vector point on the plane with normal vector and point_on_plane vector
        """
        x=point[0]
        y=point[1]
        z=point[2]
        a=normal[0]/np.dot(normal,point_on_plane)
        b=normal[1]/np.dot(normal,point_on_plane)
        c=normal[2]/np.dot(normal,point_on_plane)
        return Vehicle.project_points(x,y,z,a,b,c)[0]
    # --- Local X and Y axes for the given centre, point and normal ---
    # --- Current Coordinates of points B,C,W,T and wheel travel in Z ---

    def curr_B(self, curr_KPA_angle):
        reference = self.reference()
        self.curr_KPA_angle_for_T = curr_KPA_angle
        if np.abs(curr_KPA_angle) < 10e-4:
            return reference.r_B
        rounded_value = np.round(curr_KPA_angle,reference.maxdecimal)
        shift = curr_KPA_angle - rounded_value
        position_to_add = reference.zeropos+int(np.round(curr_KPA_angle,reference.maxdecimal)*reference.conversionstep)
        if (np.abs(shift) < 10e-4):
            if(reference.dpB[position_to_add][0]<reference.r_B[0]/1000000000):
                self.old_B = Vehicle.rotation(reference.dpB[position_to_add-int(np.sign(curr_KPA_angle))].tolist(), self.curr_A(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(),self.curr_K(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(), np.sign(curr_KPA_angle)*reference.step)
                self.old_O = Vehicle.rotation(reference.dpO[position_to_add-int(np.sign(curr_KPA_angle))].tolist(), self.curr_A(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(),self.curr_K(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(), np.sign(curr_KPA_angle)*reference.step)
                self.old_W = Vehicle.rotation(reference.dpW[position_to_add-int(np.sign(curr_KPA_angle))].tolist(), self.curr_A(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(),self.curr_K(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(), np.sign(curr_KPA_angle)*reference.step)
                
                [t] = fsolve(self.solveO, [0.01], xtol = 0.001)
                reference.dpB[position_to_add] = Vehicle.rotation(
                    self.old_B.tolist(),
                    self.fvsa_ic(curr_KPA_angle - np.sign(curr_KPA_angle) * reference.step).tolist(),
                    self.svsa_ic(curr_KPA_angle - np.sign(curr_KPA_angle) * reference.step).tolist(),
                    t
                )
            return reference.dpB[position_to_add]
        self.old_O = Vehicle.rotation(reference.dpT[position_to_add].tolist(), self.curr_A(rounded_value).tolist(),self.curr_K(rounded_value).tolist(), shift)
        self.old_W = Vehicle.rotation(reference.dpW[position_to_add].tolist(), self.curr_A(rounded_value).tolist(),self.curr_K(rounded_value).tolist(), shift)
        self.old_B = Vehicle.rotation(reference.dpB[position_to_add].tolist(), self.curr_A(rounded_value).tolist(),self.curr_K(rounded_value).tolist(), shift)
        [t] = fsolve(self.solveO, [0.01], xtol = 0.001)
        temp = Vehicle.rotation(
            self.old_B.tolist(),
            self.fvsa_ic(rounded_value).tolist(),
            self.svsa_ic(rounded_value).tolist(),
            t
        )
        return temp     
    def curr_C(self, curr_KPA_angle):
        reference = self.reference()

        if curr_KPA_angle==0:
            return reference.r_C
        temp = self.curr_B(curr_KPA_angle)
        length = Vehicle.magnitude(self.tierod(0))
        # print(curr_KPA_angle)
        return np.array([reference.r_C[0],temp[1]-np.sqrt(length**2-(reference.r_C[0]-temp[0])**2-(reference.r_C[2]-temp[2])**2), reference.r_C[2]])
    def solveO(self, inputval):
        reference = self.reference()
        t = inputval[0]
        theta = self.curr_KPA_angle_for_T
        # position_to_add = reference.zeropos + int(np.round(theta, reference.maxdecimal) * reference.conversionstep)
        spindle = self.old_O - self.old_W
        heading = np.cross(np.array([0,0,1]), spindle)
        dir = np.cross(heading,spindle)
        inclination = dir/Vehicle.magnitude(dir)
        oldT = self.old_O + reference.tire_radius*inclination
        tempO = Vehicle.rotation(
            self.old_O.tolist(),
            self.fvsa_ic(theta - np.sign(theta) * reference.step).tolist(),
            self.svsa_ic(theta - np.sign(theta) * reference.step).tolist(),
            t
        )
        tempW = Vehicle.rotation(
            self.old_W.tolist(),
            self.fvsa_ic(theta - np.sign(theta) * reference.step).tolist(),
            self.svsa_ic(theta - np.sign(theta) * reference.step).tolist(),
            t
        )
        spindle = tempO - tempW
        heading = np.cross(np.array([0,0,1]), spindle)
        dir = np.cross(heading,spindle)
        inclination = dir/Vehicle.magnitude(dir)
        tempT = tempO + reference.tire_radius*inclination
        eq1 = self.delta_z(theta) - self.delta_z(theta - reference.step * np.sign(theta)) + (tempT - oldT)[2]
        return [eq1]
    def curr_W(self, curr_KPA_angle):
        reference = self.reference()
        self.curr_KPA_angle_for_T = curr_KPA_angle
        if np.abs(curr_KPA_angle) < 10e-4:
            return reference.r_W
        position_to_add = reference.zeropos+int(np.round(curr_KPA_angle,reference.maxdecimal)*reference.conversionstep)
        rounded_value = np.round(curr_KPA_angle,reference.maxdecimal)
        shift = curr_KPA_angle - rounded_value
        if (np.abs(shift) < 10e-4):
            if(reference.dpW[position_to_add][0]<reference.r_W[0]/1000000000):
                self.old_O = Vehicle.rotation(reference.dpO[position_to_add-int(np.sign(curr_KPA_angle))].tolist(), self.curr_A(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(),self.curr_K(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(), np.sign(curr_KPA_angle)*reference.step)
                self.old_W = Vehicle.rotation(reference.dpW[position_to_add-int(np.sign(curr_KPA_angle))].tolist(), self.curr_A(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(),self.curr_K(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(), np.sign(curr_KPA_angle)*reference.step)
                
                [t] = fsolve(self.solveO, [0.01], xtol = 0.001)
                
                tempW = Vehicle.rotation(
                    self.old_W.tolist(),
                    self.fvsa_ic(curr_KPA_angle - np.sign(curr_KPA_angle) * reference.step).tolist(),
                    self.svsa_ic(curr_KPA_angle - np.sign(curr_KPA_angle) * reference.step).tolist(),
                    t
                )
                
                reference.dpW[position_to_add] = tempW
            return reference.dpW[position_to_add]
        self.old_O = Vehicle.rotation(reference.dpO[position_to_add].tolist(), self.curr_A(rounded_value).tolist(),self.curr_K(rounded_value).tolist(), shift)
        self.old_W = Vehicle.rotation(reference.dpW[position_to_add].tolist(), self.curr_A(rounded_value).tolist(),self.curr_K(rounded_value).tolist(), shift)
        [t] = fsolve(self.solveO, [0.01], xtol = 0.001)
        tempW = Vehicle.rotation(
            self.old_W.tolist(),
            self.fvsa_ic(rounded_value).tolist(),
            self.svsa_ic(rounded_value).tolist(),
            t
        )
        return tempW
    def curr_T(self, curr_KPA_angle):
        reference = self.reference()
        self.curr_KPA_angle_for_T = curr_KPA_angle
        if np.abs(curr_KPA_angle) < 10e-4:
            return reference.r_T
        # rounded_value = np.round(curr_KPA_angle,reference.maxdecimal)
        # shift = curr_KPA_angle - rounded_value
        # position_to_add = reference.zeropos+int(rounded_value*reference.conversionstep)
        inclination = self.wheel_inclination(curr_KPA_angle)
        return self.curr_O(curr_KPA_angle) + reference.tire_radius*inclination

        # if (np.abs(shift) < 10e-4):
        #     if(reference.dpT[position_to_add][0]<reference.r_T[0]/1000000000):
                
                
        #         self.old_O = Vehicle.rotation(reference.dpO[position_to_add-int(np.sign(curr_KPA_angle))].tolist(), self.curr_A(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(),self.curr_K(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(), np.sign(curr_KPA_angle)*reference.step)
        #         self.old_T =  Vehicle.rotation(reference.dpT[position_to_add-int(np.sign(curr_KPA_angle))].tolist(), self.curr_A(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(),self.curr_K(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(), np.sign(curr_KPA_angle)*reference.step)
                
        #         self.old_W = Vehicle.rotation(reference.dpW[position_to_add-int(np.sign(curr_KPA_angle))].tolist(), self.curr_A(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(),self.curr_K(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(), np.sign(curr_KPA_angle)*reference.step)
                
        #         print(self.old_T)
        #         [t] = fsolve(self.solveO, [0.01])
        #         tempO = Vehicle.rotation(
        #             self.old_O.tolist(),
        #             self.fvsa_ic(curr_KPA_angle - np.sign(curr_KPA_angle) * reference.step).tolist(),
        #             self.svsa_ic(curr_KPA_angle - np.sign(curr_KPA_angle) * reference.step).tolist(),
        #             t
        #         )
        #         tempW = Vehicle.rotation(
        #             self.old_W.tolist(),
        #             self.fvsa_ic(curr_KPA_angle - np.sign(curr_KPA_angle) * reference.step).tolist(),
        #             self.svsa_ic(curr_KPA_angle - np.sign(curr_KPA_angle) * reference.step).tolist(),
        #             t
        #         )
                
        #         self.wheel_inclination(curr_KPA_angle)
        #         mag = reference.tire_radius # Vehicle.magnitude(tempO - np.array([0,0, tempO[2]]) - tempT + np.array([0,0, tempT[2]]))
        #         dir1 = tempO - tempW
        #         dir1 = dir1/Vehicle.magnitude(dir1)
        #         dir2 = np.cross(np.array([0,0,1]), dir1)
        #         dir = np.cross(dir2, dir1)
        #         reference.dpT[position_to_add] = tempO + reference.tire_radius*dir
        #     return reference.dpT[position_to_add]
        
        self.old_T = Vehicle.rotation(reference.dpT[position_to_add].tolist(), self.curr_A(rounded_value).tolist(),self.curr_K(rounded_value).tolist(), shift)
        self.old_O = Vehicle.rotation(reference.dpO[position_to_add].tolist(), self.curr_A(rounded_value).tolist(),self.curr_K(rounded_value).tolist(), shift)
        self.old_W = Vehicle.rotation(reference.dpO[position_to_add].tolist(), self.curr_A(rounded_value).tolist(),self.curr_K(rounded_value).tolist(), shift)
        
        [t] = fsolve(self.solveO, [0.01])
        tempT = Vehicle.rotation(
            self.old_T.tolist(),
            self.fvsa_ic(rounded_value).tolist(),
            self.svsa_ic(rounded_value).tolist(),
            t
        )
        tempO = Vehicle.rotation(
            self.old_O.tolist(),
            self.fvsa_ic(rounded_value).tolist(),
            self.svsa_ic(rounded_value).tolist(),
            t
        )

        return tempT # np.array([tempO[0], tempT[1], tempT[2]])
    def delta_z(self, curr_KPA_angle):
        reference = self.reference()
        if curr_KPA_angle == 0:
            return 0
        position_to_add = reference.zeropos+int(np.round(curr_KPA_angle,reference.maxdecimal)*reference.conversionstep)
        if(np.abs(reference.dpdz[position_to_add])<1e-4):
            old_T = Vehicle.rotation(self.curr_T(curr_KPA_angle - np.sign(curr_KPA_angle)*reference.step).tolist(), self.curr_A(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(),self.curr_K(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(), np.sign(curr_KPA_angle)*reference.step)
            reference.dpdz[position_to_add] = reference.dpdz[position_to_add-int(np.sign(curr_KPA_angle))] + old_T[2] - reference.r_T[2]
           
        return reference.dpdz[position_to_add]
    
    def solvebump(self, inputval):
        reference = self.reference()
        t = inputval[0]
        theta = self.curr_KPA_angle_for_bump_steer
        tempT = Vehicle.rotation(
            self.curr_T(theta).tolist(),
            self.fvsa_ic(theta).tolist(),
            self.svsa_ic(theta).tolist(),
            t
        )
        eq1 = self.bump + (tempT - self.curr_T(theta))[2]
        return [eq1]
    def solveB(self, inputval):
        reference = self.reference()
        x = inputval[0]
        y = inputval[1]
        theta = self.curr_KPA_angle_for_bump_steer
        z = self.curr_B(theta)[2] + self.bump
        [x1,y1,z1] = self.curr_C(theta)
        temp = np.array([x,y,z])
        [t] = fsolve(self.solvebump, [0.01])
        currK = Vehicle.rotation(
            self.curr_K(theta).tolist(),
            self.fvsa_ic(theta).tolist(),
            self.svsa_ic(theta).tolist(),
            t
        )
        currA = Vehicle.rotation(
            self.curr_A(theta).tolist(),
            self.fvsa_ic(theta).tolist(),
            self.svsa_ic(theta).tolist(),
            t
        )
        currKPA = (currA - currK)
        currKPA = currKPA/Vehicle.magnitude(currKPA)
        steeringarm = temp-Vehicle.projection(currA, currKPA, temp)
        eq1 = (x-x1)**2 + (y-y1)**2 + (z-z1)**2 - Vehicle.magnitude(self.tierod(0))**2
        eq2 = Vehicle.magnitude(steeringarm) - Vehicle.magnitude(self.steering_arm(0))
        return [eq1, eq2]
    def bump_steer(self, curr_KPA_angle, bump):
        theta = curr_KPA_angle
        self.curr_KPA_angle_for_bump_steer = theta
        self.bump = bump
        [x,y] = fsolve(self.solveB, [0,0])
        z = self.curr_B(theta)[2] + self.bump
        currB = np.array([x,y,z])
        [t] = fsolve(self.solvebump, [0.01])
        currK = Vehicle.rotation(
            self.curr_K(theta).tolist(),
            self.fvsa_ic(theta).tolist(),
            self.svsa_ic(theta).tolist(),
            t
        )
        currA = Vehicle.rotation(
            self.curr_A(theta).tolist(),
            self.fvsa_ic(theta).tolist(),
            self.svsa_ic(theta).tolist(),
            t
        )
        currKPA = (currA - currK)
        currKPA = currKPA/Vehicle.magnitude(currKPA)
        steeringarm = currB-Vehicle.projection(currA, currKPA, currB)
        currT = Vehicle.rotation(
            self.curr_T(theta).tolist(),
            self.fvsa_ic(theta).tolist(),
            self.svsa_ic(theta).tolist(),
            t
        )
        currW = Vehicle.rotation(
            self.curr_W(theta).tolist(),
            self.fvsa_ic(theta).tolist(),
            self.svsa_ic(theta).tolist(),
            t
        )
        currO = Vehicle.rotation(
            self.curr_O(theta).tolist(),
            self.fvsa_ic(theta).tolist(),
            self.svsa_ic(theta).tolist(),
            t
        )
        currW =  Vehicle.projection(currW, currO - currT, currT)
        dot = np.dot(steeringarm,self.steering_arm(0))/Vehicle.magnitude(steeringarm)/Vehicle.magnitude(self.steering_arm(0))
        angle = np.degrees(np.arccos(dot))
        # print(currB)
        currT = Vehicle.rotation(
            currT.tolist(),
            currA.tolist(),
            currK.tolist(),
            angle
        )
        currO = Vehicle.rotation(
            currO.tolist(),
            currA.tolist(),
            currK.tolist(),
            angle
        )
        currW = Vehicle.rotation(
            currW.tolist(),
            currA.tolist(),
            currK.tolist(),
            angle
        )
        currW =  Vehicle.projection(currW, currO - currT, currT)
        currTW = currW-currT
        currTW = currTW/Vehicle.magnitude(currTW)
        bump_steer = np.degrees(np.arccos(np.dot(currTW, self.wheel_heading(curr_KPA_angle))))
        return bump_steer

    def curr_O(self, curr_KPA_angle):
        reference = self.reference()
        self.curr_KPA_angle_for_T = curr_KPA_angle
        if np.abs(curr_KPA_angle)<=10e-4:
            return reference.r_O
        rounded_value = np.round(curr_KPA_angle,reference.maxdecimal)
        shift = curr_KPA_angle - rounded_value
        position_to_add = reference.zeropos+int(rounded_value*reference.conversionstep)
        if (np.abs(shift) < 10e-4):
            if(reference.dpO[position_to_add][0]<reference.r_O[0]/1000000000):
                
                self.old_O = Vehicle.rotation(reference.dpO[position_to_add-int(np.sign(curr_KPA_angle))].tolist(), self.curr_A(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(),self.curr_K(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(), np.sign(curr_KPA_angle)*reference.step)
                self.old_W = Vehicle.rotation(reference.dpW[position_to_add-int(np.sign(curr_KPA_angle))].tolist(), self.curr_A(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(),self.curr_K(curr_KPA_angle-reference.step*np.sign(curr_KPA_angle)).tolist(), np.sign(curr_KPA_angle)*reference.step)
                [t] = fsolve(self.solveO, [0.01], xtol = 0.001)
                reference.dpO[position_to_add] = Vehicle.rotation(
                    self.old_O.tolist(),
                    self.fvsa_ic(curr_KPA_angle - np.sign(curr_KPA_angle) * reference.step).tolist(),
                    self.svsa_ic(curr_KPA_angle - np.sign(curr_KPA_angle) * reference.step).tolist(),
                    t
                )
            return reference.dpO[position_to_add]
        self.old_O = Vehicle.rotation(reference.dpO[position_to_add].tolist(), self.curr_A(rounded_value).tolist(),self.curr_K(rounded_value).tolist(), shift)
        self.old_W = Vehicle.rotation(reference.dpW[position_to_add].tolist(), self.curr_A(rounded_value).tolist(),self.curr_K(rounded_value).tolist(), shift)
        [t] = fsolve(self.solveO, [0.01], xtol = 0.001)
        tempO = Vehicle.rotation(
            self.old_O.tolist(),
            self.fvsa_ic(rounded_value).tolist(),
            self.svsa_ic(rounded_value).tolist(),
            t
        )
        return tempO
    # --- Current Tangent Motion of the Tire Contact Patch, returns the direction ---
    def curr_tangent(self, point):
        reference = self.reference()
        currA = self.curr_A(self.curr_KPA_angle)
        currKPA = self.curr_KPA(self.curr_KPA_angle)
        temp = Vehicle.projection(currA,currKPA,point) - point
        product = np.cross(temp, currKPA)
        ans = np.array([product[0], product[1], 0])
        ans = ans/Vehicle.magnitude(ans)
        return ans
    # --- Rack Displacement, Steering Arm and Tie Rod ---
    def rack_displacement(self, curr_KPA_angle):
        reference = self.reference()
        return self.curr_C(curr_KPA_angle)[1]-reference.r_C[1]
    def steering_arm(self, curr_KPA_angle):
        reference = self.reference()
        temp = self.curr_B(curr_KPA_angle)
        currKPA = self.curr_KPA(curr_KPA_angle)
        currA = self.curr_A(curr_KPA_angle)
        return temp-Vehicle.projection(currA, currKPA, temp)
    def tierod(self, curr_KPA_angle):
        return self.curr_C(curr_KPA_angle)-self.curr_B(curr_KPA_angle)
    # --- Inclination and Heading ---
    def wheel_inclination(self, curr_KPA_angle):
        spindle = self.spindle(curr_KPA_angle)
        heading = np.cross(np.array([0,0,1]), spindle)
        dir = np.cross(heading,spindle)
        dir = dir/Vehicle.magnitude(dir)
        return dir
    def spindle(self, curr_KPA_angle):
        return self.curr_O(curr_KPA_angle)-self.curr_W(curr_KPA_angle)
    def wheel_centre_axis(self, curr_KPA_angle):
        val = self.spindle(curr_KPA_angle)
        val = np.array([val[0],val[1],0])  
        mag = Vehicle.magnitude(val)
        return val/mag
    def wheel_heading(self, curr_KPA_angle):
        val = np.cross(np.array([0,0,1]), self.wheel_centre_axis(curr_KPA_angle))
        mag = Vehicle.magnitude(val)
        # val = self.curr_W(curr_KPA_angle)-self.curr_O(curr_KPA_angle)
        # val = val - np.array([0,0,val[2]])
        # mag = Vehicle.magnitude(val)
        # coordinate = self.curr_O(curr_KPA_angle)
        # dx = 1e-2
        # coordinate1 = self.curr_O(curr_KPA_angle-np.sign(curr_KPA_angle)*1e-2)
        # dely = coordinate-coordinate1                
        return val/mag
    # --- Caster Trail, Scrub Radius and Spindle Length ---
    def trails(self, curr_KPA_angle):
        currT = self.curr_T(curr_KPA_angle)
        currI = self.curr_I(curr_KPA_angle)
        return currT - currI 
    def caster_trail(self, curr_KPA_angle):
        head = self.wheel_heading(curr_KPA_angle)
        mag = Vehicle.magnitude(head)
        return -np.dot(self.trails(curr_KPA_angle),head/mag)
    def scrub_radius(self, curr_KPA_angle):
        head = self.wheel_heading(curr_KPA_angle)
        mag  = Vehicle.magnitude(head)
        return np.sign(np.dot(self.trails(curr_KPA_angle),
                                np.cross(head,np.array([0,0,1]))))*Vehicle.magnitude(self.trails(curr_KPA_angle) + 
                                                                                     self.caster_trail(curr_KPA_angle)*head/mag)
    # --- Camber and Road Steer ---
    def camber(self, curr_KPA_angle):
        camber_angle  = -np.arcsin(np.dot(self.spindle(curr_KPA_angle), np.array([0,0,1])))
        return np.degrees(camber_angle)
        inclination = self.wheel_inclination(curr_KPA_angle)
        heading = self.wheel_heading(curr_KPA_angle)
        triple_product = np.cross(np.cross(heading, inclination), heading)
        mag = Vehicle.magnitude(triple_product)
        return np.sign(np.cross(inclination,heading)[2])*np.degrees(np.arccos(np.dot(triple_product, np.array([0,0,1]))/mag))
        # projected_wc = Vehicle.projection(self.curr_O(curr_KPA_angle),self.wheel_heading(curr_KPA_angle),self.curr_T(curr_KPA_angle))
        # projected_wheel_inclination = projected_wc - self.curr_T(curr_KPA_angle)
        # projected_wheel_inclination = projected_wheel_inclination/Vehicle.magnitude(projected_wheel_inclination)
        # return np.sign(np.cross(projected_wheel_inclination, self.wheel_heading(curr_KPA_angle))[2])*np.degrees(np.arccos(np.dot(projected_wheel_inclination, np.array([0,0,1]))))
    def road_steer(self, curr_KPA_angle):
        return np.degrees(np.sign(curr_KPA_angle)*np.arccos(np.dot(self.wheel_heading(curr_KPA_angle),
                                                                   self.wheel_heading(0))/(Vehicle.magnitude(self.wheel_heading(curr_KPA_angle))*Vehicle.magnitude(self.wheel_heading(0)))))
    def wheel_angle(self, curr_KPA_angle):
        return -np.sign(self.wheel_heading(curr_KPA_angle)[1])*np.degrees(np.arccos(np.dot(self.wheel_heading(curr_KPA_angle),
                                                                   np.array([-1,0,0]))/(Vehicle.magnitude(self.wheel_heading(curr_KPA_angle))*Vehicle.magnitude(self.wheel_heading(0)))))
    # --- Steering Ratio ---
    def steering_ratio(self, curr_KPA_angle):
        """Steering Ratio is the Ratio of Steering Wheel Angle to the Road Steer Angle

        Args:
            curr_KPA_angle (float): Angle rotated by the Kingpin Axis 

        Returns:
            float: Steering Ratio
        """
        if np.abs(curr_KPA_angle)<0.2:
            return self.steering_ratio(0.2)
        reference = self.reference()
        return -1/(self.road_steer(curr_KPA_angle)/self.rack_displacement(curr_KPA_angle)*2*np.pi*self.pinion/360)
    def caster(self, curr_KPA_angle):
        CurrentKPA = self.curr_KPA(curr_KPA_angle)
        currx = np.array([1,0,0])
        curry = np.cross(np.array([0,0,1]), currx)
        h1 = (CurrentKPA - np.dot(CurrentKPA,curry)*curry)/Vehicle.magnitude(CurrentKPA - np.dot(CurrentKPA,curry)*curry)
        h2 = (CurrentKPA - np.dot(CurrentKPA,currx)*currx)/Vehicle.magnitude(CurrentKPA - np.dot(CurrentKPA,currx)*currx)
        return np.degrees(np.arccos(np.dot(h1,np.array([0,0,1]))))
    def kpi(self, curr_KPA_angle):
        CurrentKPA = self.curr_KPA(curr_KPA_angle)
        currx = np.array([1,0,0])
        curry = np.cross(np.array([0,0,1]), currx)
        h1 = (CurrentKPA - np.dot(CurrentKPA,curry)*curry)/Vehicle.magnitude(CurrentKPA - np.dot(CurrentKPA,curry)*curry)
        h2 = (CurrentKPA - np.dot(CurrentKPA,currx)*currx)/Vehicle.magnitude(CurrentKPA - np.dot(CurrentKPA,currx)*currx)
        return np.degrees(np.arccos(np.dot(h2,np.array([0,0,1])))) 
    #  --- Regression Modeling for Inverse Functions, Kingpin Rotation Angle, Road Steering and Rack Displacement Correlated ---
    def helperroadsteer(self,x):
        # Ensure x is a scalar
        x = x[0] if isinstance(x, (list, np.ndarray)) else x
        return self.road_steer(x)
    def helperrack(self,x):
        # Ensure x is a scalar
        x = x[0] if isinstance(x, (list, np.ndarray)) else x
        return self.rack_displacement(x)

    def regression_model(self):
        reference = self.reference()
        X = np.array([])
        y = np.array([])
        t = 0#-5self.step
        for i in range(0,50*reference.conversionstep):
            t = np.round(t + reference.step,2)
            X = np.append(X,self.road_steer(t))
           
            y = np.append(y,t)
        t = 0
        for i in range(0,50*reference.conversionstep):
            t = np.round(t - reference.step,2)
            X = np.append(X,self.road_steer(t))
            y = np.append(y,t)
        X = X.reshape(-1,1)
        poly_features = PolynomialFeatures(degree=6, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        X1 = np.array([])
        y1 = np.array([])
        t1 = 0 # -5self.step
        for i in range(0,50*reference.conversionstep):
            t1 = t1 + reference.step
            X1 = np.append(X1,self.rack_displacement(t1))
            y1 = np.append(y1,t1)
        t1 = 0
        for i in range(0,50*reference.conversionstep):
            t1 = t1 - reference.step
            X1 = np.append(X1,self.rack_displacement(t1))
            y1 = np.append(y1,t1)
        X1 = X1.reshape(-1,1)
        poly_features1 = PolynomialFeatures(degree=6, include_bias=False)
        X_poly1 = poly_features1.fit_transform(X1)
        model1 = LinearRegression()
        model1.fit(X_poly1, y1)
        return model,poly_features, model1,poly_features1
    def KPA_rotation_angle(self, input_road_steer):
        # input_road_steer = np.array([input_road_steer]).reshape(-1, 1)
        # input_road_steer = reference.model[1].fit_transform(input_road_steer)
        # return (reference.model[0].predict(input_road_steer))[0]
        if np.abs(input_road_steer)<1e-3:
                return 0
        val = input_road_steer
        input_road_steer = np.array([input_road_steer]).reshape(-1, 1)
        input_road_steer = self.model[1].fit_transform(input_road_steer)
        guess = (self.model[0].predict(input_road_steer))[0]
        return fsolve(lambda x: self.helperroadsteer(x) - val, x0=[guess])[0] 
    def rack_vs_road_steer(self, input_road_steer):
        return self.rack_displacement(self.KPA_rotation_angle(input_road_steer))

    def KPA_rotation_angle_vs_rack(self, input_rack_stroke):
        try:
            if np.abs(input_rack_stroke)<1e-3:
                return 0
            val = input_rack_stroke
            input_rack_stroke = np.array([input_rack_stroke]).reshape(-1, 1)
            input_rack_stroke = self.model[3].fit_transform(input_rack_stroke)
            guess = (self.model[2].predict(input_rack_stroke))[0]
            return fsolve(lambda x: self.helperrack(x) - val, x0=[guess], xtol = 0.01)[0]
        except Exception as error:
            # Log the error and adjust theta by adding 0.01
            print(f"[Ignore] Error encountered at rack displacement = {val}: {error}. Retrying with rack displacement = {val - 0.01}")
            return self.KPA_rotation_angle_vs_rack(val - 0.01)
        return (reference.model[2].predict(input_rack_stroke))[0]
       
    def road_steer_vs_rack(self, input_rack_stroke):
        return self.road_steer(self.KPA_rotation_angle_vs_rack(input_rack_stroke))
    
    # --- Ackerman Calculations ---
    def ackerman_radius(self, angle):
        return self.wb/np.tan(np.radians(angle))
    def ackerman_radius_from_inner(self, inner_angle):
        return self.ackerman_radius(inner_angle) + self.tw/2
    # def ackerman_radius_from_outer(self, outer_angle):
    #     return self.ackerman_radius(self.ackerman(outer_angle)) - self.tw/2
    def ackerman_ideal_inner(self, inner_angle, ackerman_radius):
        reference = self.reference()
        R = ackerman_radius
        curr_KPA_angle = self.KPA_rotation_angle(-inner_angle)
        return np.degrees(np.arctan(self.wb/(R - self.tw/2 - Vehicle.magnitude(self.curr_T(curr_KPA_angle)-self.curr_T(0)))))
    def ackerman_ideal_outer(self, outer_angle, ackerman_radius):
        reference = self.reference()
        R = ackerman_radius
        curr_KPA_angle = self.KPA_rotation_angle(outer_angle)
        return np.degrees(np.arctan(self.wb/(R + self.tw/2 - Vehicle.magnitude(self.curr_T(curr_KPA_angle)-self.curr_T(0)))))
    def ackerman_percentage(self, inner, outer):
        R = self.ackerman_radius_from_inner(inner)
        return (inner - outer)/(inner - self.ackerman_ideal_outer(outer, R))*100
    def ackerman_vs_KPA(self, curr_KPA_angle):
        return self.ackerman_percentage(np.maximum(np.abs(self.wheel_angle(curr_KPA_angle)),
                                                   np.abs(self.wheel_angle(self.KPA_rotation_angle_vs_rack(np.round(-self.rack_displacement(curr_KPA_angle),1))))),
                                                   np.minimum(np.abs(self.wheel_angle(curr_KPA_angle)),
                                                              np.abs(self.wheel_angle(self.KPA_rotation_angle_vs_rack(np.round(-self.rack_displacement(curr_KPA_angle),1))))))
    def tcr(self, outer_angle, inner_angle):
        reference = self.reference()
        a = self.a
        t = self.tw
        theta1 = np.radians(outer_angle)
        theta2 = np.radians(inner_angle)
        OP1 = np.sin(theta2)*t/np.sin(theta2 - theta1)
        OP2 = np.sin(theta1)*t/np.sin(theta2 - theta1)
        OG = np.sqrt(t**2/4 + OP2**2 + 2*t/2*OP2*np.cos(theta2))
        sin_tau = np.sin(theta2)/OG*OP2
        R = np.sqrt(a**2 + OG**2 - 2*a*OG*sin_tau)
        return R/1000 
    def ccr(self, outer_angle, inner_angle):
        reference = self.reference()
        a = self.a
        t = self.tw
        theta1 = np.radians(outer_angle)
        theta2 = np.radians(inner_angle)
        OP1 = np.sin(theta2)*t/np.sin(theta2 - theta1)
        return OP1/1000
    def ideal_ccr_inner(self, inner_angle): # For 100% Ackermann, Inmner Angle
        wb = self.wb
        tw = self.tw
        theta = np.radians(inner_angle)
        inner_rear_radus = wb/np.tan(theta)
        outer_rear_radius = inner_rear_radus + tw
        outer_front_radius = np.sqrt(wb**2 + outer_rear_radius**2)
        return outer_front_radius
    def ideal_ccr_outer(self, outer_angle): # For 100% Ackermann, Outer Angle
        wb = self.wb
        tw = self.tw
        theta = np.radians(outer_angle)
        outer_rear_radius = wb/np.tan(theta)
        outer_front_radius = np.sqrt(wb**2 + outer_rear_radius**2)
        return outer_front_radius 
    # --- Tire Contact Patch positions: x_L, x_R, y_L, y_R ---
    def delta_T(self, curr_KPA_angle):
        reference = self.reference()
        return self.curr_T(curr_KPA_angle)-reference.r_T
    def x_R(self, curr_KPA_angle):
        return self.delta_T(curr_KPA_angle)[0]
    def y_R(self, curr_KPA_angle):
        return self.delta_T(curr_KPA_angle)[1]
    def x_L(self, curr_KPA_angle):
        return self.delta_T(self.KPA_rotation_angle_vs_rack(np.round(-self.rack_displacement(curr_KPA_angle),1)))[0]
    def y_L(self, curr_KPA_angle):
        return -self.delta_T(self.KPA_rotation_angle_vs_rack(np.round(-self.rack_displacement(curr_KPA_angle),1)))[1]
    def z_R(self, curr_KPA_angle):
        return self.delta_z(curr_KPA_angle)
    def z_L(self, curr_KPA_angle):
        return self.delta_z(self.KPA_rotation_angle_vs_rack(np.round(-self.rack_displacement(curr_KPA_angle),1)))
    
    # --- Wheel Loads ----
    def F_Lz(self, curr_KPA_angle):
        """
        Curent KPA angle
        """
         # --- Wheel Loads Object ----
        return self.staticsolve(curr_KPA_angle)[0]
    def F_Rz(self, curr_KPA_angle):
        """
        Curent KPA angle
        """
        return self.staticsolve(curr_KPA_angle)[1]
    def R_Lz(self, curr_KPA_angle):
        """
        Curent KPA angle
        """
        return self.staticsolve(curr_KPA_angle)[2]
    def R_Rz(self, curr_KPA_angle):
        """
        Curent KPA angle
        """
        return self.staticsolve(curr_KPA_angle)[3]
    def FrontLoad(self, curr_KPA_angle):
        return self.F_Lz(curr_KPA_angle)+self.F_Rz(curr_KPA_angle)
    def RearLoad(self, curr_KPA_angle):
        reference = self.reference()
        return self.GVW - self.FrontLoad(curr_KPA_angle)
    def FLRR(self,curr_KPA_angle):
        return self.F_Lz(curr_KPA_angle)+self.R_Rz(curr_KPA_angle)
    def FRRL(self,curr_KPA_angle):
        reference = self.reference()
        return self.GVW - self.FLRR(curr_KPA_angle)
    def CF_L(self, curr_KPA_angle):
        return self.dynamicsolve(curr_KPA_angle)[8]
    def CF_R(self, curr_KPA_angle):
        return self.dynamicsolve(curr_KPA_angle)[9]
    def CR_L(self, curr_KPA_angle):
        return self.dynamicsolve(curr_KPA_angle)[10]
    def CR_R(self, curr_KPA_angle):
        return self.dynamicsolve(curr_KPA_angle)[11]
    def NF_L(self, curr_KPA_angle):
        return self.dynamicsolve(curr_KPA_angle)[0]
    def NF_R(self, curr_KPA_angle):
        return self.dynamicsolve(curr_KPA_angle)[1]
    def NR_L(self, curr_KPA_angle):
        return self.dynamicsolve(curr_KPA_angle)[2]
    def NR_R(self, curr_KPA_angle):
        return self.dynamicsolve(curr_KPA_angle)[3]
    # --- Set of Equations for Wheel Load Calculations ---
    def staticequation(self, x):
        self.dynamic_analysis = 0
        reference = self.reference()
        zfl = x[0]
        zfr = x[1]
        zrl = x[2]
        zrr = x[3]
        Fl = reference.Kf*zfl
        Fr = reference.Kf*zfr
        Rl = reference.Kr*zrl
        Rr = reference.Kr*zrr
        
        t = self.tw
        a = self.a
        b = self.b
        W = self.GVW
        FL = np.array([self.x_L(self.curr_KPA_angle), self.y_L(self.curr_KPA_angle), -self.z_L(self.curr_KPA_angle)-zfl])
        FR = np.array([self.x_R(self.curr_KPA_angle), self.tw+self.y_R(self.curr_KPA_angle), -self.z_R(self.curr_KPA_angle)-zfr])
        RL = np.array([self.a+self.b, 0, -zrl])
        RR = np.array([self.a+self.b, self.tw, -zrr])

        eq1 = Fl*(self.y_L(self.curr_KPA_angle)) + Rr*t +Fr*(t+self.y_R(self.curr_KPA_angle)) - W*t/2
        eq2 = Fl*(a+b-self.x_L(self.curr_KPA_angle)) + Fr*(a+b-self.x_R(self.curr_KPA_angle)) - W*b
        eq3 = Fl + Fr + Rl + Rr - W
        eq4 = np.dot(np.cross(FR - FL,RL - FL), FR - RR)
        return [eq1,eq2,eq3,eq4]
    def staticsolve(self, theta):
        """
        Solves for the static wheel loads (front left, front right, rear left, rear right) 
        based on the current Kingpin Axis (KPA) angle.

        Args:
        theta (float): Current KPA angle.

        Returns:
        tuple: Static wheel loads (Fl, Fr, Rl, Rr).
        """
        self.dynamic_analysis = 0  # Set to static analysis mode
        reference = self.reference()  # Get the reference object (static or dynamic)
        a = self.a  # Distance from CG to rear axle
        b = self.b  # Distance from CG to front axle
        W = self.GVW  # Gross Vehicle Weight
        self.curr_KPA_angle = theta  # Set the current KPA angle

        # Initial guesses for wheel loads based on weight distribution
        F = W * b / (a + b) * 0.5  # Front axle load (half distributed to each wheel)
        R = W * a / (a + b) * 0.5  # Rear axle load (half distributed to each wheel)

        # Solve the static equations for wheel displacements (zfl, zfr, zrl, zrr)
        [zfl, zfr, zrl, zrr] = fsolve(
            self.staticequation, [F / reference.Kf, F / reference.Kf, R / reference.Kr, R / reference.Kr]
        )

        # Calculate the wheel loads using the spring stiffness values
        Fl = reference.Kf * zfl  # Front left wheel load
        Fr = reference.Kf * zfr  # Front right wheel load
        Rl = reference.Kr * zrl  # Rear left wheel load
        Rr = reference.Kr * zrr  # Rear right wheel load

        return Fl, Fr, Rl, Rr  # Return the calculated wheel loads

    def dynamicequation(self, x):
        self.dynamic_analysis = 1
        reference = self.reference()
        g = self.g
        xFl = x[0]
        xFr = x[1]
        xRl = x[2]
        xRr = x[3]
        alphafL = x[4]
        alphafR = x[5]
        # R = x[6]
        Fl = xFl * self.FAW
        Fr = xFr * self.FAW
        Rl = xRl * self.FAW
        Rr = xRr * self.FAW
        # Fl = reference.Kfce.Kfcerence.Kf*zfl
        # Fr = reference.Kfce.Kf*zfr
        # Rl = reference.Krce.Krce.Krce.Kr*zrl
        # Rr = reference.Krce.Kr*zrr
        theta = self.curr_KPA_angle
        thetaL = np.abs(self.road_steer(self.KPA_rotation_angle_vs_rack(np.round(-self.rack_displacement(theta),1))))
        
        thetaR = np.abs(self.road_steer(theta))
       
        zfl = Fl/reference.Kf
        zfr = Fr/reference.Kf
        zrl = Rl/reference.Kr
        zrr = Rr/reference.Kr
        a = self.a
        b = self.b
        t = self.tw
        FL = np.array([self.x_L(theta), self.y_L(theta), -self.z_L(theta)-zfl])
        FR = np.array([self.x_R(theta), t+self.y_R(theta), -self.z_R(theta)-zfr])
        RL = np.array([a+b, 0, -zrl])
        RR = np.array([a+b, t, -zrr])

        h = reference.CG_height + self.curr_O(theta)[2]-self.curr_T(theta)[2]
        M = self.GVW
        W = M*g
        V = self.speed
        yL = self.y_L(theta)
        yR = self.y_R(theta)
        xL = self.x_L(theta)
        xR = self.x_R(theta)

        theta2 = np.radians(thetaR - alphafR)
        theta1 = np.radians(thetaL - alphafL)

        OP1 = np.sin(theta2)*t/np.sin(theta2 - theta1)
        OP2 = np.sin(theta1)*t/np.sin(theta2 - theta1)
        OG = np.sqrt(t**2/4 + OP2**2 + 2*t/2*OP2*np.cos(theta2))
        sin_tau = np.sin(theta2)/OG*OP2
        cos_tau = np.sqrt(1-sin_tau**2)
        R = np.sqrt(a**2 + OG**2 - 2*a*OG*sin_tau)
        # print(R)
        tan_alpharL = ((a+b)/OP1 - np.sin(theta1))/np.cos(theta1)
        tan_alpharR = ((a+b)/OP2 - np.sin(theta2))/np.cos(theta2)
        alpharL = np.rad2deg(np.atan(tan_alpharL))
        alpharR = np.rad2deg(np.atan(tan_alpharR))
        theta3 = np.radians(alpharL)
        theta4 = np.radians(alpharR)
        # print(alpharR)

        gamma = np.acos(OG*cos_tau/R)
        B = reference.tiredata[0]
        C = reference.tiredata[1]
        D = reference.tiredata[2]
        E = reference.tiredata[3]
        CF_Loads = self.CF_Loads # np.array([0, 150, 200, 250, 500])
        CF_Stiffnessrad= self.CF_Stiffnessrad*self.CF_Factor # np.array([0, 20234.57749,	23031.75745, 24629.16378, 24629.16378 + 250*(24629.16378-23031.75745)/50])
        interpolator = interp1d(CF_Loads, CF_Stiffnessrad, kind='linear')
        if (Fl<0 or Fr<0 or Rl<0 or Rr<0 or Fl>CF_Loads[-1] or Fr>CF_Loads[-1] or Rl>CF_Loads[-1] or Rr>CF_Loads[-1]):
            eq1 = -1
            eq2 = -1
            eq3 = -1
            eq4 = -1
            eq5 = -1
            eq6 = -1
        Cfl = np.sign(Fl)*interpolator(np.abs(Fl))
        Cfr = np.sign(Fr)*interpolator(np.abs(Fr))
        Crl = np.sign(Rl)*interpolator(np.abs(Rl))
        Crr = np.sign(Rr)*interpolator(np.abs(Rr))
        mufl = 1.0*0.2*np.sqrt(2*Fl/reference.tirep)
        mufr = 1.0*0.2*np.sqrt(2*Fr/reference.tirep)
        murl = 1.0*0.2*np.sqrt(2*Rl/reference.tirep)
        murr = 1.0*0.2*np.sqrt(2*Rr/reference.tirep)
        alphafLprime = Cfl/g*np.tan(np.radians(alphafL))/mufl/Fl
        alphafRprime = Cfr/g*np.tan(np.radians(alphafR))/mufr/Fr
        alpharLprime = Crl/g*np.tan(np.radians(alpharL))/murl/Rl
        alpharRprime = Crr/g*np.tan(np.radians(alpharR))/murr/Rr

        CFL = mufl*1.00*D*Fl*np.sin(C*np.atan(B*((alphafLprime) - E*(alphafLprime) +E/B*np.atan(B*(alphafLprime)))))
        CFR = mufr*1.00*D*Fr*np.sin(C*np.atan(B*((alphafRprime) - E*(alphafRprime) +E/B*np.atan(B*(alphafRprime)))))
        CRL = murl*1.00*D*Rl*np.sin(C*np.atan(B*((alpharLprime) - E*(alpharLprime) +E/B*np.atan(B*(alpharLprime)))))
        CRR = murr*1.00*D*Rr*np.sin(C*np.atan(B*((alpharRprime) - E*(alpharRprime) +E/B*np.atan(B*(alpharRprime)))))

        dlr = reference.tire_radius

        # eq1 = (CFL+CFR+CRL+CRR)*h - (Fl-Fr+Rl-Rr)*t/2# due to radial force
        eq1 = Rr*t + Fr*(t+yR)+Fl*yL + (CFL*np.cos(theta1) + CFR*np.cos(theta2)+CRL*np.cos(theta3) + CRR*np.cos(theta4))*(h+dlr) - M*t/2  # Fl*yL + Rr*t + Fr*(t+yR) + (CFL*np.cos(theta1) + CFR*np.cos(theta2)+CRL*np.cos(theta3) + CRR*np.cos(theta4))*h - M*t/2 
        eq2 = Fl*(a + b -xL) + Fr*(a +b -xR) + (CRL*np.sin(theta3) + CRR*np.sin(theta4))*(h+dlr) - (M*b + (CFL*np.sin(theta1) + CFR*np.sin(theta2))*(h+dlr)) # Fl*(a + b - xL) + Fr*(a +b - xR) + (CRL*np.sin(theta3) + CRR*np.sin(theta4))*h - (M*b + (CFL*np.sin(theta1) + CFR*np.sin(theta2))*h)
        eq3 = (Fl + Fr + Rl + Rr - M)
        eq4 = np.dot(np.cross(FR - FL,RL - FL), FR - RR)
        eq5 = CFL*np.cos(theta1)*yL+CFR*np.cos(theta2)*(t+yR) + CFL*np.sin(theta1)*(a+b-xL) + CFR*np.sin(theta2)*(a+b-xR) - (CRR*np.cos(theta4)*t + M/g*V**2/R*1000*np.sin(gamma)*t/2 + M/g*V**2/R*1000*np.cos(gamma)*b) # CFL*np.cos(theta1)*(yL) + CFR*np.cos(theta2)*(t+yR) + CFL*np.sin(theta1)*(a+b-xL) + CFR*np.sin(theta2)*(a+b-xR) - (CRR*np.cos(theta4)*t + M/g*V**2/R*1000*np.sin(gamma)*t/2 + M/g*V**2/R*1000*np.cos(gamma)*b)
        # eq6 = CFL*np.sin(theta1) + CFR*np.sin(theta2) - CRL*np.sin(theta3) - CRR*np.sin(theta4) - M/g*V**2/R*1000*np.sin(gamma)
        # eq4 = CFL*np.cos(theta1) + CFR*np.cos(theta2) + CRL*np.cos(theta3) + CRR*np.cos(theta4) - M/g*V**2/R*1000*np.cos(gamma)
        eq6 = CFL + CFR + CRL + CRR - M/g*V**2/R*1000
        return [eq1,eq2,eq3,eq4,eq5,eq6]     
    def dynamicsolve(self, theta):
        try:
            self.dynamic_analysis = 1
            reference = self.reference()
            g = self.g
            a = self.a
            b = self.b
            W = self.GVW
            t = self.tw
            if(theta<=-1):
                loc = -int(int(theta))
                limits = self.slipangles[loc-1]
                Flguess = self.Flguess[loc-1]
                Frguess = self.Frguess[loc-1]
                Rlguess = self.Rlguess[loc-1]
                Rrguess = self.Rrguess[loc-1]
                self.curr_KPA_angle = theta
                [Fl,Fr,Rl,Rr, alphafL, alphafR] = (fsolve(self.dynamicequation, [Flguess, Frguess, Rlguess, Rrguess, limits[0], limits[1]], xtol=0.1))
                thetaL = np.abs(self.road_steer(self.KPA_rotation_angle_vs_rack(np.round(-self.rack_displacement(theta),1))))
                thetaR = np.abs(self.road_steer(theta))
                theta2 = np.radians(thetaR - alphafR)
                theta1 = np.radians(thetaL - alphafL)
                
                OP1 = np.sin(theta2)*t/np.sin(theta2 - theta1)
                OP2 = np.sin(theta1)*t/np.sin(theta2 - theta1)
                OG = np.sqrt(t**2/4 + OP2**2 + 2*t/2*OP2*np.cos(theta2))
                sin_tau = np.sin(theta2)/OG*OP2
                cos_tau = np.sqrt(1-sin_tau**2)
                R = np.sqrt(a**2 + OG**2 - 2*a*OG*sin_tau)
                
                tan_alpharL = ((a+b)/OP1 - np.sin(theta1))/np.cos(theta1)
                tan_alpharR = ((a+b)/OP2 - np.sin(theta2))/np.cos(theta2)
                alpharL = np.rad2deg(np.atan(tan_alpharL))
                alpharR = np.rad2deg(np.atan(tan_alpharR))
            else:
                Fhalf = W*b/(a+b)*0.5
                Rhalf = W*a/(a+b)*0.5
                tempsol = self.dynamicsolve(-4)
                Fl,Fr,Rl,Rr, alphafL, alphafR, alpharL, alpharR, CFL, CFR, CRL, CRR, SAT = tempsol[0], tempsol[1], tempsol[2], tempsol[3], tempsol[4], tempsol[5], tempsol[6], tempsol[7], tempsol[8], tempsol[9], tempsol[10], tempsol[11], tempsol[12]
                Fl = self.FAW * Fl
                Fr = self.FAW * Fr
                Rl = self.FAW * Rl
                Rr = self.FAW * Rr
                Fl = Fhalf + np.abs(theta)*(Fl - Fhalf)/4
                Fr = Fhalf + np.abs(theta)*(Fr - Fhalf)/4
                Rl = Rhalf + np.abs(theta)*(Rl - Rhalf)/4
                Rr = Rhalf + np.abs(theta)*(Rr - Rhalf)/4
                alphafL = np.abs(theta)*(alphafL)/4
                alphafR = np.abs(theta)*(alphafR)/4
                alpharL = np.abs(theta)*(alpharL)/4
                alpharR = np.abs(theta)*(alpharR)/4
                CFL = np.abs(theta)*(CFL)/4
                CFR = np.abs(theta)*(CFR)/4
                CRL = np.abs(theta)*(CRL)/4
                CRR = np.abs(theta)*(CRR)/4
                SAT = (np.abs(theta)*SAT[0]/4, np.abs(theta)*SAT[1]/4)
                return Fl,Fr,Rl,Rr, alphafL, alphafR, alpharL, alpharR, CFL, CFR, CRL, CRR, SAT
            B = reference.tiredata[0]
            C = reference.tiredata[1]
            D = reference.tiredata[2]
            E = reference.tiredata[3]
            CF_Loads = self.CF_Loads # np.array([0, 150, 200, 250, 500])
            CF_Stiffnessrad = self.CF_Stiffnessrad*self.CF_Factor # np.array([0, 20234.57749,	23031.75745, 24629.16378, 24629.16378 + 250*(24629.16378-23031.75745)/50])
            interpolator = interp1d(CF_Loads, CF_Stiffnessrad, kind='linear')
            Fl = self.FAW * Fl
            Fr = self.FAW * Fr
            Rl = self.FAW * Rl
            Rr = self.FAW * Rr
            if (Fl<0 or Fr<0 or Rl<0 or Rr<0 or Fl>CF_Loads[-1] or Fr>CF_Loads[-1] or Rl>CF_Loads[-1] or Rr>CF_Loads[-1]):
                return self.dynamicsolve(theta + 0.01)
            Cfl = interpolator(Fl)
            Cfr = interpolator(Fr)
            Crl = interpolator(Rl)
            Crr = interpolator(Rr)
            mufl = 1.0*0.18*np.sqrt(2*Fl/reference.tirep) 
            mufr = 1.0*0.18*np.sqrt(2*Fr/reference.tirep) 
            murl = 1.0*0.18*np.sqrt(2*Rl/reference.tirep) 
            murr = 1.0*0.18*np.sqrt(2*Rr/reference.tirep) 
            alphafLprime = Cfl/g*np.tan(np.radians(alphafL))/mufl/Fl
            alphafRprime = Cfr/g*np.tan(np.radians(alphafR))/mufr/Fr
            alpharLprime = Crl/g*np.tan(np.radians(alpharL))/murl/Rl
            alpharRprime = Crr/g*np.tan(np.radians(alpharR))/murr/Rr

            CFL = mufl*1.00*D*Fl*np.sin(C*np.atan(B*((alphafLprime) - E*(alphafLprime) +E/B*np.atan(B*(alphafLprime)))))
            CFR = mufr*1.00*D*Fr*np.sin(C*np.atan(B*((alphafRprime) - E*(alphafRprime) +E/B*np.atan(B*(alphafRprime)))))
            CRL = murl*1.00*D*Rl*np.sin(C*np.atan(B*((alpharLprime) - E*(alpharLprime) +E/B*np.atan(B*(alpharLprime)))))
            CRR = murr*1.00*D*Rr*np.sin(C*np.atan(B*((alpharRprime) - E*(alpharRprime) +E/B*np.atan(B*(alpharRprime)))))
            SAT = self.sat(alphafL, alphafR, Fl, Fr)    
            return Fl,Fr,Rl,Rr, alphafL, alphafR, alpharL, alpharR, CFL, CFR, CRL, CRR, SAT
        except Exception as error:
            # Log the error and adjust theta by subtracting 0.01
            self.move = -(self.move+np.sign(self.move)*0.01)
            print(f"Error encountered at theta = {theta}: {error}. Retrying with theta = {theta + self.move}")
            return self.dynamicsolve(theta + self.move)

    def trainslipangles(self):
        self.dynamic_analysis = 1
        reference = self.reference()
        angle = 0
        for i in range(49 - np.abs(int(angle))):
            angle = angle - 1
            temp = self.dynamicsolve(angle)
            loc = np.abs(np.abs(int(angle)))
            print(f"Slip Angles Training at a Kingpin Rotation Angle of {angle} deg")
            self.Flguess[loc] = temp[0]/self.FAW
            self.Frguess[loc] = temp[1]/self.FAW
            self.Rlguess[loc] = temp[2]/self.FAW
            self.Rrguess[loc] = temp[3]/self.FAW
            self.slipangles[loc][0] = temp[4]
            self.slipangles[loc][1] = temp[5]
    # --- Kingpin Moment Calulations ---
    def kpm_circular(self, theta):
        self.dynamic_analysis = 0
        reference = self.reference()
        self.curr_KPA_angle = theta
        reference.currKPA = (self.curr_A(theta)-self.curr_K(theta))/Vehicle.magnitude(reference.r_A-reference.r_K)
        normal = self.F_Rz(self.curr_KPA_angle)
        pressure = np.array([30, 34])
        cvals = np.array([13.859,14.739])
        interpolator1 = interp1d(pressure, cvals, kind='linear')
        patch_radius = np.sqrt(normal*self.g/np.pi/reference.tirep/6894.75729)
        self.mu = 0.18*np.sqrt(2*normal/reference.tirep) # interpolator1(reference.tirep)*patch_radius
        # patch_radius = np.sqrt(normal*self.g/np.pi/reference.tirep/6894.75729)
        temp = integrate.dblquad(self.tire_twisting_moment_circular_static, 0, 2*np.pi, 0, 1000*patch_radius)[0]/10**9
        if 0==self.curr_KPA_angle:
            return temp
        return temp 
    def circular_contactpatch_element(self, r, phi):
        curr_point = self.curr_T(self.curr_KPA_angle) + np.array([r*np.cos(phi),r*np.sin(phi),0])
        return self.curr_tangent(curr_point)
    def linear_interpolation(input_value):
        """
        Linearly interpolates with increasing gradient.
        Maps 0 to 0.2 and 50 to 0.4.

        Parameters:
            input_value (float): The input value to scale.

        Returns:
            float: A scaled value between 0.2 and 0.4.
        """
        # Ensure input_value is within the valid range
        input_value = np.maximum(-11, np.minimum(input_value, 11))
        x = 0.75
        y = 0.05
        # Calculate the scaled value
        scaled_value = (x+y)/2 - (x-y) * (input_value / 22)
        return scaled_value
    def tire_twisting_moment_circular_static(self, r,phi):
        reference = self.reference()
        currA = self.curr_A(self.curr_KPA_angle)
        currT = self.curr_T(self.curr_KPA_angle)
        currKPA =  self.curr_KPA(self.curr_KPA_angle)
        currI = self.curr_I(self.curr_KPA_angle)
        distance = self.curr_T(self.curr_KPA_angle) + np.array([r*np.cos(phi),r*np.sin(phi),0]) - currI 
        theta2 = np.radians(self.wheel_angle(self.curr_KPA_angle))
        camber = np.radians(self.camber(self.curr_KPA_angle))
        right_dir = np.array([np.sin(theta2),np.cos(theta2),0])
        friction = self.mu*reference.tirep*6894.75729*r*self.circular_contactpatch_element(r,phi)
        normal_contribution = reference.tirep*6894.75729*r*np.array([0,0,1])
        camber_thrust = np.abs(reference.tirep*6894.75729*r*np.tan(camber))*right_dir
        force = friction*np.sign(-self.curr_KPA_angle) + normal_contribution + np.sign(-theta2)*camber_thrust 
        return np.dot(np.cross(distance,force), currKPA)
   
    def dynamic_element_moment_circular_right(self, r,phi):
        self.dynamic_analysis = 1
        reference = self.reference()
        theta = self.curr_KPA_angle
        currT = self.curr_T(theta)
        currI = self.curr_I(theta)
        currKPA = self.curr_KPA(theta)
        distance = currT+np.array([r*np.cos(phi),r*np.sin(phi),0]) - currI
        temp = self.tempdynamicsolution
        thetaL = np.abs(self.road_steer(self.KPA_rotation_angle_vs_rack(np.round(-self.rack_displacement(theta),1))))
        thetaR = np.abs(self.road_steer(theta))
        alphafL = temp[4]
        alphafR = temp[5]
        theta1 = np.radians(thetaL - alphafL)
        theta2 = np.radians(thetaR - alphafR)
        camber = np.radians(self.camber(self.curr_KPA_angle))
        left_dir = np.array([np.sin(theta1), np.sign(-theta)*np.cos(theta1),0])
        right_dir = np.array([np.sin(theta2), np.sign(-theta)* np.cos(theta2),0])
        CFL = temp[8]
        CFR = temp[9]
        patch_radius = self.patch_radius_right
        normal_contribution  = np.sign(-theta)*reference.tirep*6894.75729*r*np.array([0,0,1])
        cornering_contribution =  np.sign(-theta)*CFR/np.pi/(patch_radius**2)*self.g*r*right_dir
        camber_thrust = np.sign(-theta)*np.abs(reference.tirep*6894.75729*r*np.tan(camber))*right_dir
        force = cornering_contribution + normal_contribution + camber_thrust 
        return np.dot(np.cross(distance,force),currKPA)    
    def dynamic_element_moment_circular_left(self, r,phi):
        self.dynamic_analysis = 1
        reference = self.reference()
        theta = self.curr_KPA_angle
        opposite_theta = self.KPA_rotation_angle_vs_rack(np.round(-self.rack_displacement(theta),1))
        currT = self.curr_T(opposite_theta)
        currI = self.curr_I(opposite_theta)
        currKPA = self.curr_KPA(opposite_theta)
        distance = currT+np.array([r*np.cos(phi),r*np.sin(phi),0]) - currI 
        temp = self.tempdynamicsolution
        thetaL = np.abs(self.road_steer(self.KPA_rotation_angle_vs_rack(np.round(-self.rack_displacement(theta),1))))
        thetaR = np.abs(self.road_steer(theta))
        alphafL = temp[4]
        alphafR = temp[5]
        theta1 = np.radians(thetaL - alphafL)
        theta2 = np.radians(thetaR - alphafR)
        camber = np.radians(self.camber(opposite_theta))
        left_dir = np.array([np.sin(theta1), np.sign(-opposite_theta)*np.cos(theta1),0])
        right_dir = np.array([np.sin(theta2), np.sign(-opposite_theta)*np.cos(theta2),0])
        CFL = temp[8]
        CFR = temp[9]
        patch_radius = self.patch_radius_left
        normal_contribution = np.sign(-opposite_theta)*reference.tirep*6894.75729*r*np.array([0,0,1])
        camber_thrust = np.sign(-opposite_theta)*np.abs(reference.tirep*6894.75729*r*np.tan(camber))*left_dir
        cornering_contribution = np.sign(-opposite_theta)*CFL/np.pi/(patch_radius**2)*self.g*r*left_dir
        # print(cornering_contribution)
        force = cornering_contribution + normal_contribution + camber_thrust 
        return np.dot(np.cross(distance,force), currKPA)
    def kpm_circular_dynamic_left(self, theta):
        self.dynamic_analysis = 1
        reference = self.reference()
        # if (theta==0):
        #     return 0
        if (theta>0):
            return -self.kpm_circular_dynamic_right(-theta)
        if (self.tempdynamictheta != theta):
            self.tempdynamictheta = theta    
            self.tempdynamicsolution = self.dynamicsolve(theta)
        opposite_theta = self.KPA_rotation_angle_vs_rack(np.round(-self.rack_displacement(theta),1))
        currKPA = (self.curr_A(opposite_theta)-self.curr_K(opposite_theta))/Vehicle.magnitude(reference.r_A-reference.r_K)
        t = theta
        patch_radius = np.sqrt(self.tempdynamicsolution[0]*self.g/np.pi/reference.tirep/6894.75729)
        self.patch_radius_left = patch_radius
        temp = integrate.dblquad(self.dynamic_element_moment_circular_left, 0, 2*np.pi, 0, 1000*patch_radius)[0]/10**9
        # temp2 = integrate.dblquad(self.tire_twisting_moment_circular, 0, 2*np.pi, 0, 1000*patch_radius)[0]/10**9
        if 0==self.curr_KPA_angle:
            return temp
        sat_contribution = np.abs(np.dot(np.array([0,0,self.tempdynamicsolution[12][0]]),currKPA))
        return (temp + sat_contribution) 
    def kpm_circular_dynamic_right(self, theta):
        self.dynamic_analysis = 1
        reference = self.reference()
        # if (theta==0):
        #     return 0
        if (theta>0):
            return -self.kpm_circular_dynamic_left(-theta)
        if (self.tempdynamictheta != theta):  
            self.tempdynamictheta = theta     
            self.tempdynamicsolution = self.dynamicsolve(theta)
        self.curr_KPA_angle = theta
        currKPA = (self.curr_A(theta)-self.curr_K(theta))/Vehicle.magnitude(reference.r_A-reference.r_K)
        t = theta
        patch_radius = np.sqrt(self.tempdynamicsolution[1]*self.g/np.pi/reference.tirep/6894.75729)
        self.patch_radius_right = patch_radius
        # print('1')
        temp = integrate.dblquad(self.dynamic_element_moment_circular_right, 0, 2*np.pi, 0, 1000*patch_radius)[0]/10**9
        # print('2')
        if 0==self.curr_KPA_angle:
            return temp
        sat_contribution = np.abs(np.dot(np.array([0,0,self.tempdynamicsolution[12][1]]),currKPA))
        return temp + sat_contribution 
       
    def static_kingpin_moment(self, curr_KPA_angle):
        return self.kpm_circular(curr_KPA_angle)
    def sat(self, alphafL, alphafR, Fl, Fr):
        self.dynamic_analysis = 1
        reference = self.reference()
        g = self.g
        B = reference.tiredata[4]
        C = reference.tiredata[5]
        D = reference.tiredata[6]
        E = reference.tiredata[7]
        CF_Loads = self.CF_Loads # np.array([0, 150, 200, 250, 500])
        CF_Stiffnessrad = self.CF_Stiffnessrad*self.CF_Factor # np.array([0, 20234.57749,	23031.75745, 24629.16378, 24629.16378 + 250*(24629.16378-23031.75745)/50])
        CF_pneumatictrail = self.CF_pneumatictrail*self.align_factor # np.array([0, 0.011909253,	0.018484467, 0.023331694, 0.023331694 + 250*(0.023331694-0.018484467)/50])
        interpolator = interp1d(CF_Loads, CF_Stiffnessrad, kind='linear')
        pneumaticinterpolator = interp1d(CF_Loads, CF_pneumatictrail, kind='linear')
        Cfl = interpolator(Fl)
        Cfr = interpolator(Fr)
        mufl = 1.0*0.18*np.sqrt(2*Fl/reference.tirep)
        mufr = 1.0*0.18*np.sqrt(2*Fr/reference.tirep)
        alphafLprime = Cfl/g*np.tan(np.radians(alphafL))/mufl/Fl
        alphafRprime = Cfr/g*np.tan(np.radians(alphafR))/mufr/Fr
        satFLprime = D*np.sin(C*np.atan(B*((alphafLprime) - E*(alphafLprime) +E/B*np.atan(B*(alphafLprime)))))
        satFRprime = D*np.sin(C*np.atan(B*((alphafRprime) - E*(alphafRprime) +E/B*np.atan(B*(alphafRprime)))))
        TzL = pneumaticinterpolator(Fl)
        TzR = pneumaticinterpolator(Fr)
        satFL = TzL*satFLprime*mufl*1.00*Fl*g
        satFR = TzR*satFRprime*mufr*1.00*Fr*g
        return satFL, satFR
    # --- Steering Effort ---
    def tierod_force(self, curr_KPA_angle):
        self.dynamic_analysis = 0
        reference = self.reference()
        reference.currKPA = (self.curr_A(curr_KPA_angle)-self.curr_K(curr_KPA_angle))/Vehicle.magnitude(reference.r_A-reference.r_K)
        return self.static_kingpin_moment(curr_KPA_angle)*1000/np.dot(np.cross(self.steering_arm(curr_KPA_angle),
                                                                        self.tierod(curr_KPA_angle)/Vehicle.magnitude(self.tierod(curr_KPA_angle))),
                                                                        reference.currKPA)*self.tierod(curr_KPA_angle)/Vehicle.magnitude(self.tierod(curr_KPA_angle))
    def tierod_force_dynamic_right(self, curr_KPA_angle):
        self.dynamic_analysis = 1
        reference = self.reference()
        # if(curr_KPA_angle==0):
        #         return 0
        # if(curr_KPA_angle>0):
        #     curr_KPA_angle = self.KPA_rotation_angle_vs_rack(np.round(-self.rack_displacement(curr_KPA_angle),1))
        #     return self.tierod_force_dynamic_left(curr_KPA_angle)
        tierod = self.tierod(curr_KPA_angle)
        mag= Vehicle.magnitude(self.tierod(curr_KPA_angle))
        reference.currKPA = (self.curr_A(curr_KPA_angle)-self.curr_K(curr_KPA_angle))/Vehicle.magnitude(reference.r_A-reference.r_K)
        force = self.kpm_circular_dynamic_right(curr_KPA_angle)*1000/np.dot(np.cross(self.steering_arm(curr_KPA_angle),
                                                                        tierod/mag),
                                                                        reference.currKPA)*tierod/mag
        return force
    def tierod_force_dynamic_left(self, curr_KPA_angle):
        self.dynamic_analysis = 1
        reference = self.reference()
        tierod = self.tierod(curr_KPA_angle)
        mag= Vehicle.magnitude(self.tierod(curr_KPA_angle))
        reference.currKPA = (self.curr_A(curr_KPA_angle)-self.curr_K(curr_KPA_angle))/Vehicle.magnitude(reference.r_A-reference.r_K)
        force = self.kpm_circular_dynamic_left(curr_KPA_angle)*1000/np.dot(np.cross(self.steering_arm(curr_KPA_angle),
                                                                        tierod/mag),
                                                                        reference.currKPA)*tierod/mag
        return force
    def rack_force_dynamic(self, curr_KPA_angle):
        self.dynamic_analysis = 1
        reference = self.reference()
        # if(curr_KPA_angle==0):
        #         return 0
        # if(curr_KPA_angle>0):
        #     curr_KPA_angle = self.KPA_rotation_angle_vs_rack(np.round(-self.rack_displacement(curr_KPA_angle),1))
        right_tierod_force = self.tierod_force_dynamic_right(curr_KPA_angle)
        left_tierod_force = self.tierod_force_dynamic_left(curr_KPA_angle)
        return (np.dot(right_tierod_force,np.array([0,1,0]))) + (np.dot(left_tierod_force, np.array([0,1,0])))
    def mechanical_advantage_linkages_static(self, curr_KPA_angle):
        self.dynamic_analysis = 0
        reference = self.reference()
        reference.currKPA = (self.curr_A(curr_KPA_angle)-self.curr_K(curr_KPA_angle))/Vehicle.magnitude(reference.r_A-reference.r_K)
        tierodr = 1*1000/np.dot(np.cross(self.steering_arm(curr_KPA_angle),
                                                                        self.tierod(curr_KPA_angle)/Vehicle.magnitude(self.tierod(curr_KPA_angle))),
                                                                        reference.currKPA)*self.tierod(curr_KPA_angle)/Vehicle.magnitude(self.tierod(curr_KPA_angle))
        rackforce = (np.dot(tierodr,np.array([0,1,0])))
        return Vehicle.magnitude(tierodr)/rackforce
    def mechanical_advantage_static(self, curr_KPA_angle):
        """
        Calculates the mechanical advantage of the steering system in static conditions.

        This method determines the ratio of the input force applied to the steering wheel
        to the output force transmitted to the rack, providing insight into the ease of steering.
        """
        self.dynamic_analysis = 0
        reference = self.reference()
        reference.currKPA = (self.curr_A(curr_KPA_angle)-self.curr_K(curr_KPA_angle))/Vehicle.magnitude(reference.r_A-reference.r_K)
        tierodr =  1*1000/np.dot(np.cross(self.steering_arm(curr_KPA_angle),
                                                                        self.tierod(curr_KPA_angle)/Vehicle.magnitude(self.tierod(curr_KPA_angle))),
                                                                        reference.currKPA)*self.tierod(curr_KPA_angle)/Vehicle.magnitude(self.tierod(curr_KPA_angle))
        rackforce = (np.dot(tierodr,np.array([0,1,0])))
        return 1/np.abs(rackforce*self.pinion/1000)
    def mechanical_advantage_dynamic(self, curr_KPA_angle):
        self.dynamic_analysis = 1
        reference = self.reference()
        curr_A = self.curr_A(curr_KPA_angle)
        curr_K = self.curr_K(curr_KPA_angle)
        tierod = self.tierod(curr_KPA_angle)
        tierod_magnitude = Vehicle.magnitude(tierod)
        reference.currKPA = (curr_A - curr_K) / Vehicle.magnitude(reference.r_A - reference.r_K)
        tierodr = 1 * 1000 / np.dot(
            np.cross(self.steering_arm(curr_KPA_angle), tierod / tierod_magnitude),
            reference.currKPA
        ) * tierod / tierod_magnitude
        rackforce = np.dot(tierodr, np.array([0, 1, 0]))
        return 1 / np.abs(rackforce * self.pinion / 1000)
    def rack_force(self, curr_KPA_angle):
        current_tierod_force = self.tierod_force(curr_KPA_angle)
        opp_angle = np.round(self.KPA_rotation_angle_vs_rack(np.round(-self.rack_displacement(curr_KPA_angle),1)),1)
        opposite_tierod_force = -self.tierod_force(opp_angle)
        return (np.dot(current_tierod_force,
                          np.array([0,1,0]))) + np.dot(opposite_tierod_force,
                                                    np.array([0,1,0]))
 
    def static_steering_effort(self, curr_KPA_angle):
        self.dynamic_analysis = 0
        reference = self.reference()
        return np.abs(self.rack_force(curr_KPA_angle)*self.pinion/1000) + self.linkage_friction_contribution_on_steering
    def dynamic_steering_effort(self, curr_KPA_angle):
        self.dynamic_analysis = 1
        reference = self.reference()
        if(curr_KPA_angle==0):
            return 0
        if(curr_KPA_angle>0):
            curr_KPA_angle = self.KPA_rotation_angle_vs_rack(np.round(-self.rack_displacement(curr_KPA_angle),1))
        return np.abs(self.rack_force_dynamic(curr_KPA_angle)*self.pinion/1000) + self.linkage_friction_contribution_on_steering
    def returnability(self, lim_time):
        self.dynamic_analysis = 1
        reference = self.reference()
        # Parameters
        I_w = self.I_w  # Moment of Inertia of the wheel wrt the Kingpin Axis
        I_ss = self.I_ss  # Moment of Interia of the steering system wrt the steering column
        c_factor = 2*np.pi*self.pinion
        y0 = self.assumed_rack_stroke/c_factor*360 # self.rack_stroke/c_factor*360  # Initial condition for y
        v0 = 0.0  # Initial condition for y'
        # t_span = (0, 0.1)  # Time range
        # t_eval = np.linspace(t_span[0], t_span[1], 150)  # Time points to evaluate
        # Solve the ODE
        # wheel_solution = solve_ivp(self.wheel_system, t_span, [y0, v0], t_eval=t_eval, args=(I_w,))
        # steering_solution = solve_ivp(self.steering_system, t_span, [y0, v0], t_eval=t_eval, args=(I_ss,))
        # Extract solution
        system = self.wheel_system
        k = I_w
        # system = self.steering_system
        # print(wheel_solution.y[0][-1])
        # print(steering_solution.y[0][-1])
        # if wheel_solution.y[0][-1] > steering_solution.y[0][-1]:
        #     system = self.wheel_system
        #     k = I_w
        #     print('It is a wheel solution')
        
        t_span = (0, lim_time)  # Time range
        t_eval = np.linspace(t_span[0], t_span[1], 50)  # Time points to evaluate
        # Solve the ODE
        solution = solve_ivp(system, t_span, [y0, v0], method='BDF', t_eval=t_eval, args=(k,), rtol = 1e-3, atol=0.0001)
        # Extract solution
        t = solution.t
        y = solution.y[0]  # y corresponds to y1
        # Plot the solution
        plt.figure(figsize=(8, 5))
        plt.plot(t, y, label="y(t)")
        plt.title("Solution to Differential Equation $ky'' = f(y)$")
        plt.xlabel("Time (t)")
        plt.ylabel("y(t)")
        plt.grid()
        plt.legend()
        plt.show()
        return (y0-y[-1])/y0*100
    def steering_wheel_kpa_ratio(self, curr_KPA_angle):
        if np.abs(curr_KPA_angle)<0.2:
            return self.steering_wheel_kpa_ratio(0.2)
        reference = self.reference()
        rack_disp = self.rack_displacement(curr_KPA_angle)
        return -1/((curr_KPA_angle)/rack_disp*2*np.pi*self.pinion/360)
    def wheel_system(self, t, Y, k):
        self.dynamic_analysis = 1
        reference = self.reference()
        y1, y2 = Y  # Unpack Y = [y1, y2]z
        dy1_dt = y2
        c_factor = 2*np.pi*self.pinion
        if (y1/360*c_factor)>self.assumed_rack_stroke:
            y1 = self.assumed_rack_stroke/c_factor*360
        angle = self.KPA_rotation_angle_vs_rack(y1/360*c_factor)
        opp_angle = self.KPA_rotation_angle_vs_rack(-y1/360*c_factor)
        friction_r = self.linkage_friction_contribution_on_steering*self.mechanical_advantage_dynamic(angle)
        friction_l = self.linkage_friction_contribution_on_steering*self.mechanical_advantage_dynamic(opp_angle)
        factor = 1
        if(np.sign(y2)>=0 and t>0):
            factor = 0
            print(f"Optimal parameters: y1 = {y1}, t = {t}")
        print(f"Temp parameters: x1 = {y1}, t = {t}")
        dy2_dt = -factor*((self.kpm_circular_dynamic_left(angle)+self.kpm_circular_dynamic_right(angle))/2 - np.maximum(friction_l,friction_r))/ k * self.steering_wheel_kpa_ratio(angle)
        #  extra torque from spring calculatioins
        # if angle>-35:
        #       dy2_dt = -factor*(self.left_plus_right_returning_moment(angle) - 2*friction)/ 2/ k * self.steering_wheel_kpa_ratio(angle)
        # else:
        #       dy2_dt = -factor*(self.left_plus_right_returning_moment(angle)-0.3*6*(angle+35) - 2*friction)/ 2/ k * self.steering_wheel_kpa_ratio(angle)
        #  print   (dy2_dt, angle)
        return [dy1_dt, dy2_dt]
    def steering_system(self,t,Y,k):
        self.dynamic_analysis = 1
        reference = self.reference()
        y1, y2 = Y  # Unpack Y = [y1, y2]
        dy1_dt = y2
        c_factor = 2*np.pi*self.pinion
        angle = self.KPA_rotation_angle_vs_rack(y1/360*c_factor)
        friction = self.linkage_friction_contribution_on_steering
        dy2_dt = -(self.dynamic_steering_effort(angle) - friction)/ k
        return [dy1_dt, dy2_dt]

    # --- Plotting Functions ---
    def plotmycoordinates(self, func, title, legend, ylabel, xlabel):
        reference = self.reference()
        num = len(func)
        for j in range(num):
            X2 = np.array([])
            y2 = np.array([])
            t2 = self.KPA_rotation_angle_vs_rack(self.rack_stroke)
            max = self.KPA_rotation_angle_vs_rack(-self.rack_stroke)
            opop = int(max - t2)
            for i in range(0,opop*reference.conversionstep):
                t2 = t2 + reference.step
                X2 = np.append(X2,func[j](t2)[1])
                y2 = np.append(y2,func[j](t2)[0])
            X2 = X2.reshape(-1,1)
            plt.plot(X2,-y2)
            plt.scatter(func[j](0)[1], -func[j](0)[0])
            plt.scatter(func[j](self.KPA_rotation_angle_vs_rack(self.rack_stroke))[1], -func[j](self.KPA_rotation_angle_vs_rack(self.rack_stroke))[0])
            print('Initial ' + legend[j*3] + ' is ' + '('+str(round(func[j](0)[1],2)) + ', ' + str(round(func[j](0)[0],2)) +')')
            print('Extreme Values are (' + str(round(func[j](max)[1],2)) + 
                  ', '+str(round(func[j](max)[0],2)) + 
                  ') and (' + str(round((func[j](self.KPA_rotation_angle_vs_rack(self.rack_stroke)))[1],2)) + 
                  ', '+ str(round((func[j](self.KPA_rotation_angle_vs_rack(self.rack_stroke)))[0],2))+')')
        plt.legend(legend)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
    def plotmyfn(self, funcY, funcX, title, legend, ylabel, xlabel):
        reference = self.reference()
        num = len(funcY)
        for j in range(num):
            X2 = np.array([])
            y2 = np.array([])
            t2 = np.round(self.KPA_rotation_angle_vs_rack(self.rack_stroke),2)
            max = np.round(self.KPA_rotation_angle_vs_rack(-self.rack_stroke))
            opop = int(max - t2)
            for i in range(0,opop*reference.conversionstep):
                t2 = t2 + reference.step
                X2 = np.append(X2,funcX(t2))
                y2 = np.append(y2,funcY[j](t2))
            X2 = X2.reshape(-1,1)
            minim = round(np.min(y2),1)
            maxim = round(np.max(y2),1)
            plt.plot(X2,y2)
            plt.scatter( funcX(0),funcY[j](0))
            print('Initial ' + legend[j*2] + ' is ' + str(round(funcY[j](0),2)) )
            print('Extreme Values are ' + str(round(funcY[j](max),2)) + ' and ' + str(round((funcY[j](self.KPA_rotation_angle_vs_rack(self.rack_stroke))),2)))
            print('Range for ' + legend[j*2] + ' is ' + '['+ str(minim)+ ', '+str(maxim)+']')
            print('Average Absolute Value is ' + str(np.round(np.average(np.abs(y2)),2)))
        plt.legend(legend)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()

if __name__ == "__main__":
    print("Creating vehicle...")
    
    vehicle = Vehicle(
        r_A=np.array([100, 500, 300]),
        r_B=np.array([150, 450, 250]),
        r_C=np.array([200, 0, 250]),
        r_O=np.array([100, 500, 100]),
        r_K=np.array([100, 500, 50]),
        slr=315.0, dlr=315.0,
        initial_camber=-2.0, toe_in=0.0,
        tw=1200.0, wb=1600.0, GVW=250.0, b=800.0,
        CG_height=300.0,
        wheel_rate_f=20000.0, wheel_rate_r=18000.0,
        tire_stiffness_f=200000.0, tire_stiffness_r=200000.0,
        pinion=100.0, tirep=30.0, dila=20.0,
        assumed_rack_stroke=80.0,
        r_La=np.array([50, 200, 50]),
        r_Lb=np.array([50, 800, 50]),
        r_Ua=np.array([50, 200, 300]),
        r_Ub=np.array([50, 800, 300])
    )
    
    print("✓ Success!")
    print("Camber:", vehicle.camber(0))
    fl, fr, rl, rr = vehicle.staticsolve(0)
    print(f"Loads: FL={fl:.0f}N FR={fr:.0f}N")    
    # def wheel_system(self, t, Y, k):
    #     self.dynamic_analysis = 1
    #     reference = self.reference() # https://imgur.com/a/Bv2rWt2

    #     x1, x2 = Y  # Unpack Y = [y1, y2]z
    #     dx1_dt = x2
    #     c_factor = 2*np.pi*self.pinion
    #     Iw = self.I_w
    #     Is = self.I_ss
    #     rack = np.round((x1/360*c_factor),1)
    #     if np.abs(rack)>self.assumed_rack_stroke:
    #         rack = self.assumed_rack_stroke*np.sign(x1)
    #     print(rack)
    #     theta_R = self.KPA_rotation_angle_vs_rack(rack)
    #     opp_theta_R = self.KPA_rotation_angle_vs_rack(-rack)
    #     i = self.steering_wheel_kpa_ratio(theta_R)
    #     di_x = derivative(self.steering_wheel_kpa_ratio, theta_R, dx=1e-6)
    #     d2i_x = derivative(self.steering_wheel_kpa_ratio, theta_R, n=2, dx=1e-6)
    #     term1 = (-x1*i**2 * d2i_x - 2 * i * (i - x1 * di_x) * di_x) / i**4
    #     term2 = ((i - x1 * di_x) / i**2) # * dx2_dt
    #     mar = self.mechanical_advantage_dynamic(theta_R)
    #     mal = self.mechanical_advantage_dynamic(opp_theta_R)
    #     torque_lh = self.kpm_circular_dynamic_left(theta_R)
    #     friction = np.sign(-x2)*self.linkage_friction_contribution_on_steering
             
    #     factor = 1
    #     # if(np.sign(x2)>=0 and t>0):
    #     #     factor = 0
    #     #     print(f"Optimal parameters: x1 = {x1}, t = {t}")
    #     print(f"Temp parameters: x1 = {x1}, t = {t}")
    #     dx2_dt = -factor*(Iw*term1*x2**2/mar + torque_lh/mal - friction)/ (Is - Iw*term2/mar) 

    #     return [dx1_dt, dx2_dt]
    # def wheel_system(self, t, Y, k):
    #     self.dynamic_analysis = 1
    #     reference = self.reference() # https://imgur.com/a/Bv2rWt2

    #     x1, x2 = Y  # Unpack Y = [y1, y2]z
    #     dx1_dt = x2
    #     c_factor = 2*np.pi*self.pinion
    #     Iw = self.I_w
    #     Is = self.I_ss
    #     rack = np.round((x1/360*c_factor),1)
    #     if np.abs(rack)>self.assumed_rack_stroke:
    #         rack = self.assumed_rack_stroke*np.sign(x1)
    #     print(rack)
    #     theta_R = self.KPA_rotation_angle_vs_rack(rack)
    #     opp_theta_R = self.KPA_rotation_angle_vs_rack(-rack)
    #     i = self.steering_wheel_kpa_ratio(theta_R)
    #     di_x = derivative(self.steering_wheel_kpa_ratio, theta_R, dx=1e-6)
    #     d2i_x = derivative(self.steering_wheel_kpa_ratio, theta_R, n=2, dx=1e-6)
    #     term1 = (-x1*i**2 * d2i_x - 2 * i * (i - x1 * di_x) * di_x) / i**4
    #     term2 = ((i - x1 * di_x) / i**2) # * dx2_dt
    #     mal = self.mechanical_advantage_dynamic(opp_theta_R)
    #     torque_lh = self.kpm_circular_dynamic_left(theta_R)
    #     friction = np.sign(-x2)*self.linkage_friction_contribution_on_steering*mal
             
    #     factor = 1
    #     # if(np.sign(x2)>=0 and t>0):
    #     #     factor = 0
    #     #     print(f"Optimal parameters: x1 = {x1}, t = {t}")
    #     print(f"Temp parameters: x1 = {x1}, t = {t}")
    #     dx2_dt = factor*((torque_lh - friction)/Iw-term1*x2**2)/ (term2) 

    #     return [dx1_dt, dx2_dt]
