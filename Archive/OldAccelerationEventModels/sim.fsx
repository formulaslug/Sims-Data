open FSharp.Data.UnitSystems.SI.UnitSymbols
open System.IO
#r "nuget: Plotly.NET"
open Plotly.NET


type World = {
    Time : float<s>
    Position : float<m>
    Speed : float<m/s>
    Acceleration : float<m/s^2>
    Charge : float
    Voltage : float<V>
}

let delta = 0.01<s>
let duration = 120.0<s>
let rec step (outF:StreamWriter) (w:World) (results:World list) = 
    outF.WriteLine $"%.2f{w.Time},%.3f{w.Position},%.3f{w.Speed},%.3f{w.Acceleration},%.3f{w.Charge},%.3f{w.Voltage}"
    if w.Time >= duration then
        results |> List.rev |> Array.ofList
    else
        let position = w.Position + w.Speed*delta
        let speed = w.Speed + w.Acceleration*delta
        let acceleration = 1.0<m/s^2>
        let w' = {
            Time = w.Time + delta
            Position = position
            Speed = speed
            Acceleration = acceleration
            Charge = 1.0
            Voltage = 120.0<V>
            }
        step outF w' (w'::results)

let start:World = {
    Time = 0.0<s>
    Position = 0.0<m>
    Speed = 0.0<m/s>
    Acceleration = 1.0<m/s^2>
    Charge = 1
    Voltage = 120.0<V>
}


let results = 
    use outF = new StreamWriter "C:/Projects/FormulaSlug/VehicleDynamics/SimV2/SimData/Data.csv"
    outF.WriteLine "Time, Position, Speed, Acceleration, Charge, Voltage"
    step outF start []
let t = results |> Array.map (fun x -> x.Time)
let s = results |> Array.map (fun x -> x.Speed)
let a = results |> Array.map (fun x -> x.Acceleration)
let c = results |> Array.map (fun x -> x.Charge)
let v = results |> Array.map (fun x -> x.Voltage)
let line2 =
    // Drawing graph of a 'square' function
    Chart.Line(xy = Array.zip t s)
line2 |> Chart.show     

    



