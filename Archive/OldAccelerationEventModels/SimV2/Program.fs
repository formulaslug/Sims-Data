open FSharp.Data.UnitSystems.SI.UnitSymbols
open System.IO
open System
open Plotly.NET
open voltageTools.VoltageLookup

//Constants
let pi = 3.14159265358979323
let airDensity = 1.230<kg/m^3>
let g = 9.81<m/s^2>

//Mechanical Constants
let carMass = 231.4<kg> //about 509 lb
let driverMass = 70.0<kg>
let intertialMass = 27.0<kg>
let mass = carMass + driverMass + intertialMass
let crossSectionalArea = 0.6<m^2>
let dragCof = 0.63<N*s^2/m/kg>
let maxTorque = 180.0<N*m>
let wheelRadius = 0.216<m>
let wheelCircumferance = 2.0*pi*wheelRadius
let gearRatio = 40.0 / 11.0
let tireLongitudinalFrictionCoef = 1.50 //Very rough estimation
let longDistFrontTire = 0.7477<m> //longitudinal distance from front axle to center of gravity
let centerOfMassHeight = 0.247718<m>
let wheelbase = 1.6547084<m>

//Electrical Constants
let tractiveIMax = 600.0<A>

let CellIMax = tractiveIMax / 20.0
//Sim Constants
let delta = 0.001<s>
let duration = 30.0<s>

type World = {
    Time : float<s>
    Position : float<m>
    Speed : float<m/s>
    Acceleration : float<m/s^2>
    Charge : float<A*H>
    LastCurrent : float<A>
} with
    member x.Drag = 0.5 * airDensity * dragCof * crossSectionalArea * x.Speed * x.Speed
    member x.WheelRPM = x.Speed / wheelCircumferance * 60.0  // Rev/min
    member x.WheelRotationsHz = x.Speed / wheelCircumferance * 2.0 * pi // Rad/s
    member x.RPM = x.WheelRPM * gearRatio
    member x.MotorRotationHz = x.WheelRotationsHz * gearRatio
    member x.MaxPower = tractiveIMax * x.Voltage
    member x.Torque = //Wheel Torque
            let perfectTractionTorque =
                if x.MaxPower/maxTorque < x.MotorRotationHz then
                    x.MaxPower / x.MotorRotationHz * gearRatio
                else
                    maxTorque * gearRatio
            if x.RPM > 7500.0<1/s> then
                x.Drag * wheelRadius
            else
                min perfectTractionTorque x.MaxTractionTorqueAtWheel
    member x.motorTorque = x.Torque / gearRatio
    member x.Voltage = 28.0 * (lookup x.Charge x.LastCurrent)
    member x.Power = x.motorTorque * x.MotorRotationHz
    member x.Current = 
        if (x.Power / x.Voltage) > tractiveIMax then
            tractiveIMax
        else
            x.Power / x.Voltage
    member x.Weight = (carMass + driverMass) * g
    member x.MaxTraction = ((tireLongitudinalFrictionCoef * x.Weight * longDistFrontTire)/wheelbase)/(1.0-(centerOfMassHeight * tireLongitudinalFrictionCoef / wheelbase))
    member x.MaxTractionTorqueAtWheel = 
        x.MaxTraction * wheelRadius
    member x.MotorForce = x.Torque/wheelRadius
    member x.NetForce = 
        x.MotorForce - x.Drag //Account for resistance and friction. Also need to account for efficiency

(*
Tire Notes
Contact patch. Moments and torques are aligned around the center of the contact patch
rather than the center of the tire

Lateral forces:
- Those generated perpendicular to the path of travel
- Forces that help with turning
- Generally friction with contact patch
    - Also with camber thrust due to more normal force on one side of the tire than the other
    - Related to camber angle

Self Aligning Torque: Mz
- Torque along vertical direction of the tire and generated under slip angle and lateral acceleration
- Resisting direction of slip angle. It returns wheel to 0 slip angle

Rolling Moment: My
- Resistance to rolling at the front of the print
- Proportional to normal force on tire and width of the contact patch

Overturning Torque: Mx
- Means contact patch is not used efficiently in x direction (should optimize for 0)
- Cancel out with camber

Load Sensitivity
- The more normal force you add, the less effective it is at generating lateral force
- Slightly dependent on tire pressure and temperature, but largely determine by tire
geometry and construction techniques

Cornering Coefficient (Cornering stiffness):
- Sensitive to slip angle

Relaxation Length:
- Distance it takes to reach a certain lateral force

Model Types:
- Pacejka "Magic Formula"
- MRA Nondimensional Tire Model
    - Described in chapter 14 of RCVD
*)



let rec step (outF:StreamWriter) (w:World) (results:World list) = 
    outF.WriteLine $"%.2f{w.Time},%.3f{w.Position},%.3f{w.Speed},%.3f{w.Acceleration},%.3f{w.Torque},%.3f{w.Drag},%.3f{w.RPM},%.1f{w.Power},%.3f{w.MaxPower},%.3f{w.Charge},%.3f{w.Voltage},%.1f{w.Current},%.2f{w.Torque/gearRatio}"
    if w.Time >= duration then
        results |> List.rev |> Array.ofList
    else
         //Fix for first part of torque curve
        let charge = w.Charge - w.Current*delta/(3600.0<s/H>)
        let position = w.Position + w.Speed*delta
        let speed = w.Speed + w.Acceleration*delta
        let acceleration = w.NetForce/mass
        let w' = {
            Time = w.Time + delta
            Position = position
            Speed = speed
            Acceleration = acceleration
            Charge = charge
            LastCurrent = w.Current
            }
        step outF w' (w'::results)
[<EntryPoint>]
let main argv =
    let version = "3.11"
    printfn $"Version {version}"
    let startTime = DateTime.Now
    let start:World = {
        Time = 0.0<s>
        Position = 0.0<m>
        Speed = 0.0<m/s>
        Acceleration = 0.0<m/s^2>
        Charge = 2.5<A*H>*20.0
        LastCurrent = 0.0<A>
    }

    let results = 
        use outF = new StreamWriter "C:/Projects/FormulaSlug/VehicleDynamics/SimV2/SimData/Data.csv"
        outF.WriteLine "Time, Position, Speed, Acceleration, Torque, Drag, RPM, Power, Available Power, Charge, Voltage, Current, Motor Torque"
        step outF start []
    let time = results |> Array.map (fun x -> x.Time)
    let position = results |> Array.map (fun x -> x.Position)
    let speed = results |> Array.map (fun x -> x.Speed)
    let acceleration = results |> Array.map (fun x -> x.Acceleration)
    let torque = results |> Array.map (fun x -> x.Torque)
    let drag = results |> Array.map (fun x -> x.Drag)
    let rpm = results |> Array.map (fun x -> x.RPM)
    let power = results |> Array.map (fun x -> x.Power)
    let maxPower = results |> Array.map (fun x -> x.MaxPower)
    let charge = results |> Array.map (fun x -> x.Charge)
    let voltage = results |> Array.map (fun x -> x.Voltage)
    let current = results |> Array.map (fun x -> x.Current)
    let motorTorque = results |> Array.map (fun x -> x.Torque/gearRatio)
    let netForce = results |> Array.map (fun x -> x.NetForce)
    let motorForce = results |> Array.map (fun x -> x.MotorForce)




    //Plots
    let positionPlot = 
        Chart.Line(xy = Array.zip time position)
        // |> Chart.withTitle "Position Vs. Time"
        |> Chart.withXAxisStyle ("Time (s)")
        |> Chart.withYAxisStyle ("Position (m)")
    let speedPlot = 
        Chart.Line(xy = Array.zip time speed)
        // |> Chart.withTitle "Speed Vs. Time"
        |> Chart.withXAxisStyle ("Time (s)")
        |> Chart.withYAxisStyle ("Speed (m/s)")
    let accelerationPlot = 
        Chart.Line(xy = Array.zip time acceleration)
        // |> Chart.withTitle "Acceleration Vs. Time"
        |> Chart.withXAxisStyle ("Time (s)")
        |> Chart.withYAxisStyle ("Acceleration (m^2/s)")
    let torquePlot = 
        Chart.Line(xy = Array.zip time torque)
        // |> Chart.withTitle "Torque Vs. Time"
        |> Chart.withXAxisStyle ("Time (s)")
        |> Chart.withYAxisStyle ("Torque (Nm)")
    let motorTorquePlot = 
        Chart.Line(xy = Array.zip time motorTorque)
        // |> Chart.withTitle "Voltage Vs. Time"
        |> Chart.withXAxisStyle ("Time (s)")
        |> Chart.withYAxisStyle ("Motor Torque (Nm)")
    let dragPlot = 
        Chart.Line(xy = Array.zip time drag)
        // |> Chart.withTitle "Drag Vs. Time"
        |> Chart.withXAxisStyle ("Time (s)")
        |> Chart.withYAxisStyle ("Drag (N)")
    let motorForcePlot = 
        Chart.Line(xy = Array.zip time motorForce)
        // |> Chart.withTitle "Drag Vs. Time"
        |> Chart.withXAxisStyle ("Time (s)")
        |> Chart.withYAxisStyle ("Motor Force (N)")
    let netForcePlot = 
        Chart.Line(xy = Array.zip time netForce)
        // |> Chart.withTitle "Voltage Vs. Time"
        |> Chart.withXAxisStyle ("Time (s)")
        |> Chart.withYAxisStyle ("Net Force (N)")
    let rpmPlot = 
        Chart.Line(xy = Array.zip time rpm)
        // |> Chart.withTitle "RPM Vs. Time"
        |> Chart.withXAxisStyle ("Time (s)")
        |> Chart.withYAxisStyle ("RPM (Cycles/Min)")
    let powerPlot = 
        Chart.Line(xy = Array.zip time power)
        // |> Chart.withTitle "Power Vs. Time"
        |> Chart.withXAxisStyle ("Time (s)")
        |> Chart.withYAxisStyle ("Power (W)")
    let availablePowerPlot = 
        Chart.Line(xy = Array.zip time maxPower)
        // |> Chart.withTitle "Available Power Vs. Time"
        |> Chart.withXAxisStyle ("Time (s)")
        |> Chart.withYAxisStyle ("Available Power (W)")
    let chargePlot = 
        Chart.Line(xy = Array.zip time charge)
        // |> Chart.withTitle "Battery Charge Vs. Time"
        |> Chart.withXAxisStyle ("Time (s)")
        |> Chart.withYAxisStyle ("Charge (Amp Hours)")
    let voltagePlot = 
        Chart.Line(xy = Array.zip time voltage)
        // |> Chart.withTitle "Voltage Vs. Time"
        |> Chart.withXAxisStyle ("Time (s)")
        |> Chart.withYAxisStyle ("Voltage (V)")
    let currentPlot = 
        Chart.Line(xy = Array.zip time current)
        // |> Chart.withTitle "Voltage Vs. Time"
        |> Chart.withXAxisStyle ("Time (s)")
        |> Chart.withYAxisStyle ("Current (A)")
    
    let singleStack =
        [
        positionPlot
        speedPlot
        accelerationPlot
        torquePlot
        motorTorquePlot
        dragPlot
        motorForcePlot
        netForcePlot
        rpmPlot
        powerPlot
        availablePowerPlot
        chargePlot
        voltagePlot
        currentPlot
        ]
        // |> Chart.SingleStack(Pattern = StyleParam.LayoutGridPattern.Coupled)
        // |> Chart.withTitle (title = "Hi i am the new SingleStackChart")
        // |> Chart.withXAxisStyle (TitleText = "im the shared xAxis")
        |> Chart.Grid(3,5)
        |> Chart.withLayoutGridStyle (YGap = 0.4)
        |> Chart.withLayoutGridStyle (XGap = 0.3)
        |> Chart.withSize (Width = 1300, Height = 750)
        |> Chart.withTitle "Car Characteristics"


    singleStack |> Chart.show
    printfn $"Elapsed Time = {(DateTime.Now - startTime).TotalMilliseconds}"
    printfn $"Steps Per Second = {((1.0)/delta * duration)/((DateTime.Now - startTime).TotalSeconds)}"
    let time3tenthsMeter =  position |> Array.findIndex (fun x -> (System.Math.Round((x/1.0<m>), 2)) * 1.0<m> = 0.3<m>)
    let time75and3tenthsMeter = position |> Array.findIndex (fun x -> (System.Math.Round((x/1.0<m>), 3)) * 1.0<m> = 75.3<m>)
    printfn $"Time for Acceleration: {float (time75and3tenthsMeter - time3tenthsMeter) / 1000.0} seconds"
    printfn $"Start Time (0.3m): {time3tenthsMeter}"
    printfn $"End Time (75.3m): {time75and3tenthsMeter}"
    0 //Exit code
    



