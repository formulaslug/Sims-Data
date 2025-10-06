namespace voltageTools
open FSharp.Data.UnitSystems.SI.UnitSymbols
// open System.IO

exception CellChargeOutofBounds of string
exception CurrentDrawOutofBounds of string

//https://www.thunderheartreviews.com/2019/09/sony-vtc5-test-review.html
//https://www.e-cigarette-forum.com/threads/sony-vtc5-2500mah-18650-retest-results-a-great-20a-2600mah-battery.733578/
//General documentation and Temp Stuff: https://www.powerstream.com/p/us18650vtc5-vtc5.pdf

module VoltageLookup =

    let table_half discharged =
        match discharged with
        | 0.00<A*H> -> 4.15<V>
        | 0.10<A*H> -> 4.09<V>
        | 0.20<A*H> -> 4.06<V>
        | 0.30<A*H> -> 4.03<V>
        | 0.40<A*H> -> 3.98<V>
        | 0.50<A*H> -> 3.94<V>
        | 0.60<A*H> -> 3.90<V>
        | 0.70<A*H> -> 3.87<V>
        | 0.80<A*H> -> 3.84<V>
        | 0.90<A*H> -> 3.80<V>
        | 1.00<A*H> -> 3.76<V>
        | 1.10<A*H> -> 3.72<V>
        | 1.20<A*H> -> 3.69<V>
        | 1.30<A*H> -> 3.66<V>
        | 1.40<A*H> -> 3.63<V>
        | 1.50<A*H> -> 3.61<V>
        | 1.60<A*H> -> 3.58<V>
        | 1.70<A*H> -> 3.56<V>
        | 1.80<A*H> -> 3.54<V>
        | 1.90<A*H> -> 3.52<V>
        | 2.00<A*H> -> 3.50<V>
        | 2.10<A*H> -> 3.46<V>
        | 2.20<A*H> -> 3.41<V>
        | 2.30<A*H> -> 3.35<V>
        | 2.40<A*H> -> 3.31<V>
        | 2.50<A*H> -> 3.25<V>
        | 2.60<A*H> -> 2.99<V>
        | _ -> 
            printfn $"{discharged} A*H"
            raise (CellChargeOutofBounds("Cell Charge out of Bounds: 0.5 A"))
    
    let table_5A discharged =
        match discharged with
        | 0.00<A*H> -> 4.05<V>
        | 0.10<A*H> -> 3.94<V>
        | 0.20<A*H> -> 3.90<V>
        | 0.30<A*H> -> 3.86<V>
        | 0.40<A*H> -> 3.82<V>
        | 0.50<A*H> -> 3.77<V>
        | 0.60<A*H> -> 3.73<V>
        | 0.70<A*H> -> 3.69<V>
        | 0.80<A*H> -> 3.66<V>
        | 0.90<A*H> -> 3.62<V>
        | 1.00<A*H> -> 3.58<V>
        | 1.10<A*H> -> 3.54<V>
        | 1.20<A*H> -> 3.50<V>
        | 1.30<A*H> -> 3.47<V>
        | 1.40<A*H> -> 3.44<V>
        | 1.50<A*H> -> 3.41<V>
        | 1.60<A*H> -> 3.38<V>
        | 1.70<A*H> -> 3.35<V>
        | 1.80<A*H> -> 3.32<V>
        | 1.90<A*H> -> 3.28<V>
        | 2.00<A*H> -> 3.25<V>
        | 2.10<A*H> -> 3.20<V>
        | 2.20<A*H> -> 3.14<V>
        | 2.30<A*H> -> 3.05<V>
        | 2.40<A*H> -> 2.86<V>
        | _ -> 
            printfn $"{discharged} A*H"
            raise (CellChargeOutofBounds("Cell Charge out of Bounds: 5 A"))
    
    let table_10A discharged = 
        match discharged with
        | 0.00<A*H> -> 3.97<V>
        | 0.10<A*H> -> 3.81<V>
        | 0.20<A*H> -> 3.77<V>
        | 0.30<A*H> -> 3.73<V>
        | 0.40<A*H> -> 3.68<V>
        | 0.50<A*H> -> 3.64<V>
        | 0.60<A*H> -> 3.61<V>
        | 0.70<A*H> -> 3.57<V>
        | 0.80<A*H> -> 3.53<V>
        | 0.90<A*H> -> 3.50<V>
        | 1.00<A*H> -> 3.46<V>
        | 1.10<A*H> -> 3.42<V>
        | 1.20<A*H> -> 3.39<V>
        | 1.30<A*H> -> 3.35<V>
        | 1.40<A*H> -> 3.32<V>
        | 1.50<A*H> -> 3.29<V>
        | 1.60<A*H> -> 3.26<V>
        | 1.70<A*H> -> 3.24<V>
        | 1.80<A*H> -> 3.21<V>
        | 1.90<A*H> -> 3.18<V>
        | 2.00<A*H> -> 3.14<V>
        | 2.10<A*H> -> 3.11<V>
        | 2.20<A*H> -> 3.05<V>
        | 2.30<A*H> -> 3.00<V>
        | 2.40<A*H> -> 2.91<V>
        | 2.50<A*H> -> 2.80<V>
        | 2.60<A*H> -> 1.0<V>
        | _ -> 
            printfn $"{discharged} A*H"
            raise (CellChargeOutofBounds("Cell Charge out of Bounds: 10 A"))

    let table_15A discharged =
        match discharged with
        | 0.00<A*H> -> 3.83<V>
        | 0.10<A*H> -> 3.69<V>      
        | 0.20<A*H> -> 3.64<V>      
        | 0.30<A*H> -> 3.60<V>      
        | 0.40<A*H> -> 3.56<V>      
        | 0.50<A*H> -> 3.52<V>      
        | 0.60<A*H> -> 3.49<V>      
        | 0.70<A*H> -> 3.45<V>      
        | 0.80<A*H> -> 3.41<V>      
        | 0.90<A*H> -> 3.38<V>      
        | 1.00<A*H> -> 3.34<V>      
        | 1.10<A*H> -> 3.31<V>      
        | 1.20<A*H> -> 3.28<V>      
        | 1.30<A*H> -> 3.25<V>      
        | 1.40<A*H> -> 3.22<V>      
        | 1.50<A*H> -> 3.19<V>      
        | 1.60<A*H> -> 3.16<V>      
        | 1.70<A*H> -> 3.14<V>      
        | 1.80<A*H> -> 3.11<V>      
        | 1.90<A*H> -> 3.08<V>      
        | 2.00<A*H> -> 3.05<V>      
        | 2.10<A*H> -> 3.01<V>      
        | 2.20<A*H> -> 2.97<V>      
        | 2.30<A*H> -> 2.91<V>      
        | 2.40<A*H> -> 2.83<V>      
        | 2.50<A*H> -> 2.70<V>      
        | 2.60<A*H> -> 1.0<V>
        | _ -> 
            printfn $"{discharged} A*H"
            raise (CellChargeOutofBounds("Cell Charge out of Bounds: 15 A"))

    let table_20A discharged = 
        match discharged with
        | 0.00<A*H> -> 3.70<V>
        | 0.10<A*H> -> 3.59<V>
        | 0.20<A*H> -> 3.54<V>
        | 0.30<A*H> -> 3.50<V>
        | 0.40<A*H> -> 3.46<V>
        | 0.50<A*H> -> 3.43<V>
        | 0.60<A*H> -> 3.40<V>
        | 0.70<A*H> -> 3.36<V>
        | 0.80<A*H> -> 3.33<V>
        | 0.90<A*H> -> 3.30<V>
        | 1.00<A*H> -> 3.27<V>
        | 1.10<A*H> -> 3.25<V>
        | 1.20<A*H> -> 3.22<V>
        | 1.30<A*H> -> 3.20<V>
        | 1.40<A*H> -> 3.17<V>
        | 1.50<A*H> -> 3.14<V>
        | 1.60<A*H> -> 3.12<V>
        | 1.70<A*H> -> 3.10<V>
        | 1.80<A*H> -> 3.08<V>
        | 1.90<A*H> -> 3.05<V>
        | 2.00<A*H> -> 3.02<V>
        | 2.10<A*H> -> 2.99<V>
        | 2.20<A*H> -> 2.96<V>
        | 2.30<A*H> -> 2.92<V>
        | 2.40<A*H> -> 2.86<V>
        | 2.50<A*H> -> 2.74<V>
        | 2.60<A*H> -> 1.0<V>
        | _ -> 
            printfn $"{discharged} A*H"
            raise (CellChargeOutofBounds("Cell Charge out of Bounds: 20 A"))

    let table_25A discharged =
        match discharged with
        | 0.00<A*H> -> 3.70<V>
        | 0.10<A*H> -> 3.49<V>
        | 0.20<A*H> -> 3.43<V>
        | 0.30<A*H> -> 3.38<V>
        | 0.40<A*H> -> 3.34<V>
        | 0.50<A*H> -> 3.30<V>
        | 0.60<A*H> -> 3.26<V>
        | 0.70<A*H> -> 3.22<V>
        | 0.80<A*H> -> 3.19<V>
        | 0.90<A*H> -> 3.17<V>
        | 1.00<A*H> -> 3.15<V>
        | 1.10<A*H> -> 3.13<V>
        | 1.20<A*H> -> 3.11<V>
        | 1.30<A*H> -> 3.10<V>
        | 1.40<A*H> -> 3.08<V>
        | 1.50<A*H> -> 3.06<V>
        | 1.60<A*H> -> 3.04<V>
        | 1.70<A*H> -> 3.01<V>
        | 1.80<A*H> -> 3.00<V>
        | 1.90<A*H> -> 2.97<V>
        | 2.00<A*H> -> 2.95<V>
        | 2.10<A*H> -> 2.92<V>
        | 2.20<A*H> -> 2.88<V>
        | 2.30<A*H> -> 2.80<V>
        | 2.40<A*H> -> 2.55<V>
        | 2.50<A*H> -> 2.20<V>
        | 2.60<A*H> -> 1.0<V>
        | _ -> 
            printfn $"{discharged} A*H"
            raise (CellChargeOutofBounds("Cell Charge out of Bounds: 25 A"))

    let table_30A discharged = 
        match discharged with
        | 0.00<A*H> -> 3.56<V>
        | 0.10<A*H> -> 3.40<V>
        | 0.20<A*H> -> 3.34<V>
        | 0.30<A*H> -> 3.28<V>
        | 0.40<A*H> -> 3.23<V>
        | 0.50<A*H> -> 3.18<V>
        | 0.60<A*H> -> 3.14<V>
        | 0.70<A*H> -> 3.11<V>
        | 0.80<A*H> -> 3.09<V>
        | 0.90<A*H> -> 3.07<V>
        | 1.00<A*H> -> 3.05<V>
        | 1.10<A*H> -> 3.04<V>
        | 1.20<A*H> -> 3.04<V>
        | 1.30<A*H> -> 3.02<V>
        | 1.40<A*H> -> 3.01<V>
        | 1.50<A*H> -> 3.01<V>
        | 1.60<A*H> -> 2.99<V>
        | 1.70<A*H> -> 2.98<V>
        | 1.80<A*H> -> 2.96<V>
        | 1.90<A*H> -> 2.94<V>
        | 2.00<A*H> -> 2.92<V>
        | 2.10<A*H> -> 2.88<V>
        | 2.20<A*H> -> 2.86<V>
        | 2.30<A*H> -> 2.80<V>
        | 2.40<A*H> -> 2.70<V>
        | 2.50<A*H> -> 2.40<V>
        | 2.60<A*H> -> 1.0<V>
        | _ -> 
            printfn $"{discharged} A*H"
            raise (CellChargeOutofBounds("Cell Charge out of Bounds: 30 A"))

    let table_35A discharged =
        match discharged with
        | 0.00<A*H> -> 3.41<V>
        | 0.10<A*H> -> 3.25<V>
        | 0.20<A*H> -> 3.20<V>
        | 0.30<A*H> -> 3.13<V>
        | 0.40<A*H> -> 3.08<V>
        | 0.50<A*H> -> 3.03<V>
        | 0.60<A*H> -> 3.00<V>
        | 0.70<A*H> -> 2.97<V>
        | 0.80<A*H> -> 2.95<V>
        | 0.90<A*H> -> 2.94<V>
        | 1.00<A*H> -> 2.92<V>
        | 1.10<A*H> -> 2.92<V>
        | 1.20<A*H> -> 2.92<V>
        | 1.30<A*H> -> 2.91<V>
        | 1.40<A*H> -> 2.91<V>
        | 1.50<A*H> -> 2.91<V>
        | 1.60<A*H> -> 2.90<V>
        | 1.70<A*H> -> 2.88<V>
        | 1.80<A*H> -> 2.86<V>
        | 1.90<A*H> -> 2.85<V>
        | 2.00<A*H> -> 2.75<V>
        | 2.10<A*H> -> 2.50<V>
        | 2.20<A*H> -> 2.30<V>
        | 2.30<A*H> -> 2.00<V>
        | 2.40<A*H> -> 1.0<V>
        | 2.50<A*H> -> 1.0<V>
        | 2.60<A*H> -> 1.0<V>
        | _ -> raise (CellChargeOutofBounds("Cell Charge out of Bounds: 35 A"))

    let lookup (charge:float<A*H>) (currentDraw:float<A>) =
        let currentPerCell = currentDraw/20.0
        let chargePerCell = charge/20.0
        // printfn $"charge: {charge}, currentDraw: {currentDraw}, Current Per Cell: {currentPerCell}, Charge Per Cell: {chargePerCell}"
        let discharged = 2.6<A*H> - chargePerCell
        let lower = floor (currentPerCell/5.0<A>) * 5.0<A>
        let upper = (floor (currentPerCell/5.0<A>) + 1.0 ) * 5.0<A>
        let chargeRounded = (round (discharged*10.0/1.0<A*H>)) / 10.0 * 1.0<A*H>
        let voltageMatch current =
            match current with
            | 0.0<A> -> table_half chargeRounded
            | 5.0<A> -> table_5A chargeRounded
            | 10.0<A> -> table_10A chargeRounded
            | 15.0<A> -> table_15A chargeRounded
            | 20.0<A> -> table_20A chargeRounded
            | 25.0<A> -> table_25A chargeRounded
            | 30.0<A> -> table_30A chargeRounded
            | 35.0<A> -> table_35A chargeRounded
            | _ -> 
                printfn($"{current}")
                raise (CurrentDrawOutofBounds("Current Draw out of bounds (did not round to 0-35)"))
        let lowerVoltage = voltageMatch lower
        let upperVoltage = voltageMatch upper
        let lowerPercent = (currentPerCell - lower) / 5.0<A>
        let voltage = lowerPercent*upperVoltage + (1.0-lowerPercent)*lowerVoltage
        voltage