# Offline notes on trip Aug/Sep 2025

1. [DBC](#dbc)
1. [Drag Analysis](#drag)
    1. [RPM Increase on Left Turn](#speedincrease)
    1. [VDM GPS Lag](#vdmgpslag)
1. [Logging Issues](#li)
    1. [Acc Voltages](#accv)
    1. [VDM](#vdm)
    1. [Motor Controller](#mc)
1. [Logging Fixes](#lf)
    1. [08172025 Data](#08172025data)
    1. [apply_DBC_Scale_and_Offset.py fix](#soFix)
    1. [Time Column Reconstructions](#tcr)
1. [Building Fun Stuff](#creations)
    1. [Time Column](#tc)
    1. [Basic View](#bv)
    1. [GPS Lap Segmentation and Lap Time](#glsalt)
    1. [Integration with non-uniform time-series data](#0)
1. [Bit of Battery Research](#bobr)
    1. [Notes](#bobrnotes)
    1. [Battery Testing](#batterytester)
    1. [Sources](#bobrsources)
1. [Other Notes](#on)



<h2 id ="dbc"> DBC </h2>

- ```ACC_STATUS_GLV_VOLTAGE``` is missing its scale of ```0.1``` in the DBC.

<h2 id ="drag"> Drag Analysis </h2>

<h3 id="speedincrease"> RPM Increase On left turn </h3>

I noticed that the RPM stops decreasing/tends to increase on left turns. My hypothesis for this is that the back right wheel is connected directly to the motor and increases in speed relative to the back left wheel and rest of the car due to the differential resulting in a subtle motor RPM increase on moderate/sharp left turns. Right turns appear to have the opposite effect.

![alt text](refs/vdmgpsspeedincrease.png)

<h3 id="vdmgpslag"> VDM GPS Speed Lag </h3>

- Not an issue but VDM GPS speed looks to be shifted timewise relative to the motor RPM a little bit
![alt text](refs/gpsSpeedOffset.png)

<h2 id ="li"> Logging Issues </h2>

<h3 id ="accv"> ACC Voltages </h3>

- Accumulator board tends to send 0s for everything sometimes when the rest of the CAN bus is working. It tends to correlate with TS not being switched on and goes back to working when TS is turned on. It is inconsistent in most possible ways.
- Accumulator board total Voltage doesn't tend to match the sum voltage. Probably a logging issue but unclear if it's on the side of the telem board or acc board.
    - Seems isolated to [08172025_26autox1](../FS-3/08172025/08172025_26autox1.parquet)
    ![alt text](refs/cell_acc_voltage_discrepancy.png)
    - I was wrong it's also aparent in the first half of[08172025_20_Endurance1P1](../FS-3/08172025/08172025_20_Endurance1P1.parquet). This seems like it also has something to do with logging, not just the acc reporting?
    ![alt text](refs/cell_acc_voltage_discrepancy2.png)
- I am worried this also applies to ACC temps and then what will happen if the ACC overtemps?
- Worth noting how the MC voltage plays into this too:
![alt text](refs/cellAccMCV.png)

<h3 id ="vdm"> VDM </h3>

- CPP For logging does not account for the empty byte in the GPS_UTC CAN message and thus hours or minutes is lost and ```VDM_UTC_TIME_SECONDS``` is actually the minute counter
    - Couldn't actually find the hours or seconds anywhere when one of them should be available.
- VDM gps is somewhat intermittent on 08172025. I'm worried it's been damaged or the antenna was placed poorly this day. It is significantly less accurate than previous runs and drifts way across the track and even to like a nearby town occasionally. When filtering by ```VDM_GPS_VALID1```, some non-valid GPS values are returned as well.
![alt text](refs/gps1.png)
![alt text](refs/gps2.png)
![alt text](refs/gps3.png)


<h3 id ="mc"> Motor Controller </h3>

- Sometimes the motor controller reports a high current value (Although it appears like a flat line at ```6553.5```, it actually steps down a little bit almost logarithmically to ~```6551.0``` and wobbles +- ```1``` around there). [08172025_22_6LapsAndWeirdCurrData](../FS-3/08172025/08172025_22_6LapsAndWeirdCurrData.parquet)
    - This seems to be the only MC CAN message affected, all the other ones make sense still. It is preceded by error code ```26``` (assuming this is undervoltage) and happens right as the DC Bus V (voltage measured by the MC) falls below ~70V and settles at ~26V. It only does this the first time it undervolts. It comes back down to 0 and is fixed right as the driver tries to throttle again after the first undervolt recovers. Does not happen on the second undervolt. Starts again on the 4th undervolt.
    - MC repeatedly undervolted, stayed low for ~20 seconds, pause for ~10 seconds where car is precharged but driver doing nothing, then driver takes off and undervolts again. Repeated 4 times in this run before 
    - ```ACC_STATUS_PRECHARGE_DONE``` remains true (```1```) for the entire duration of these events.
    ![Section of 08172025_22_6LapsAndWeirdCurrData that demonstrates the above description](<refs/MCvoltDropimage.png>)

<h2 id ="lf"> Logging Fixes </h2>

<h3 id ="08172025data"> 08172025 Data </h3>


- Parquets were in various states of scaled/offset. Some were scaled twice, others weren't. This led me to find other issues with the data but in the end I just reconstructed them from the fsdaq files. They are renamed and saved properly in FS-3/08172025/..

<h3 id ="soFix"> apply_DBC_Scale_and_Offset.py fix </h3>


- scale and offset were applied as ```(x - offset) * scale``` when it should have been ```(x * scale) + offset```
    - 08102025 files were corrected to fit this. I noticed it when the ACC seg voltages (which are one of the few things that are offset) were ~1.9 instead of ~3.9
    - 08172025 files reflect this change as well

<h3 id ="tcr"> Time Column Reconstructions </h3>


- Time is missing (AEM Dash used to record it) so a millisecond counter is needed. Should already be implemented by Jack in the next version.
    - Until this is in operation, two methods were developed for doing time, ```simpleTimeCol``` and ```timeCol```.
        - ```simpleTimeCol``` uses the fact that the logging occurs at ~```5035/60``` times per second and just assumes every time step is that. 
        - ```timeCol``` interpolates between minutes to more accurately determine the time step for every data point (but also leads to variability in the time step). This method also depends on the VDM working.

<h2 id="creations"> Building Fun Stuff </h2>

<h3 id="tc"> Time Column </h3>

Interpolated between minutes to get a more accurate estimate of time! Using a fixed value didn't work super well as there was some drift. I did have to use the fixed value for the first and last segments because they weren't complete minutes and thus were not calculated easily unless you knew the amount of time that had passed which creates a paradox. Comparison:

![alt text](refs/timeColFix.png)

<h3 id="bv"> Basic View </h3>

Created a "Basic View" function that captures some of the key information coming out of a run to ensure validity and look for bugs. As a part of this, I also added ```read``` and ```readValid``` for reading a parquet more quickly and clearing it of non-valid GPS data easily.

Using just ```basicView(readValid("FS-3/08172025/08172025_27autox2&45C_35C_~28Cambient_100fans.parquet"))``` you can quickly determine basic information about the run:

![alt text](refs/basicView.png)

<h3 id="glsalt"> GPS Lap Segmentation and Lap Time </h3>

I needed a way of segmenting laps for lap times and because it is useful. To do this, I used a GPS Latitude/Longitude box + a few other factors:
1. Car is moving
1. Car was not in the box in the last 5 seconds
1. Car has a 10 sec delay on start from entering a new lap

It came out like this:

![alt text](refs/gpsLap1.png)
![alt text](refs/gpsLap2.png)

<h3 id="0"> Integration with non-uniform time-series data </h3>

I had an integration method which assumed 0.01 stepSize (100 Hz). This is no longer really true so I changed it to support any step size as an argument (Default still 0.01), and created a new one that takes a col and a timeCol which it uses to integrate. It is currently implemented as a riemann sum but should be swapped to scipy rk45 when possible.

Implemented this with RK45 as well. Didn't seem to yield better results and was about the same speed.

<h2 id="bobr"> Bit of Battery Reseach + MC issue</h2>

<h3 id="bobrnotes"> General Notes </h3>

I've been investigating our power consumption and working on making an ml based battery model that is more accurate than the one we built last year. For this, i've dug through lots of data on the VTC5A and related cells. The model will be based on charge usage in the last 20 sec (although I plan to improve this to be a more logarithmic based recovery as effects can last as long as 30+ seconds but have less effect over time. This is less relevant for our type of driving where current usage fluctuates constantly but still may be worth improving).

One important thing I've noticed going through all our car data is we face the same issue as before: we can't effectively use more than ~300 Amps (sometimes 400) without having motor controller issues. Thus, while the VTC5A cells may be the best choice if we have full use of our power train, in practice another cell with a higher energy density and lower current rating may be more useful as we can't use our 600-700A desired current draws.

To resolve the above, I plan to generate a bunch of data relating to the motor controller fault on jacks and ideally on a track, use that data in xgboost or something related to generate ```a random forest + explainer``` and/or a simpler ```decision tree``` that can inform us what conditions actually cause the motor controller fault. This will likely have to wait till we have more data.

<h3 id="batterytester"> Battery Tester </h3>
Something useful to us could also be using a battery tester. West mountain radio has a few options. 

For the $100 one, you can test constant current till cutoff voltage, change current draw during test with GUI, graph V vs Time, charge discharge cycles.

For the $1k one (really for the software), it can measure intenal resistance, constant power output, duty cycle test (and generally pulse testing). 
1. [Testers](https://www.westmountainradio.com/cba.php)
1. [$1k product](https://www.westmountainradio.com/product_info.php?products_id=sys500_watt)  
1. [West Mountain Radio Software Comparison](https://www.westmountainradio.com/pdf/cba-test-modes.pdf)

Should try and get it sponsored or find someone who has one (maybe ask Russell?).

<h3 id="bobrsources"> Sources </h3>

1. [Enepaq battery comparison](https://enepaq.com/wp-content/uploads/2025/02/VTC6-VTC5A-LG-HG2-LG-HE2-Sanyo-GA-Samsung-30Q-Samsung-25R-cells-discharge-characteristics-comparison-at-1C-and-5C-rates-.pdf)
1. [VTC5A discharge curve 20-35A](https://www.e-cigarette-forum.com/threads/sony-vtc5a-2500mah-18650-bench-test-results-a-fantastic-25a-battery.746719/)
1. [Murata VTC5A (discharge curves for 2.5-20A + more)](https://www.murata.com/-/media/webrenewal/products/batteries/cylindrical/datasheet/us18650vtc5a-product-datasheet.ashx?la=en-us&cvid=20250324010000000000)
1. [Murata VTC5D](https://www.murata.com/-/media/webrenewal/products/batteries/cylindrical/datasheet/us18650vtc5d-product-datasheet.ashx?la=en&cvid=20250324010000000000)

<h2 id ="on"> Other data parsing notes. Worth keeping in mind. </h2>

- Columns sometimes end up as Uint8 or int8 because we manually set them rather than having polars interpret them. This combined with the fact that polars supports doing ```df[<col>] * 100``` means that ```3*100``` wraps around to ```44```. Maybe we should promote uint8 and int8 to 16 or 32? Would make stuff sizeably bigger for just a little bit of convenience. But this is a really annoying thing to happen. I wonder if you can reprogram the standard action polars takes in dataframes. Interestingly, numpy auto promotes.

- You will get an error of ```float type does not support .to_bytes()``` when you try to do the byte correcton for the VDM data twice. This is because initially the data is interpreted as integers when it is saved into a parquet file so the int can be converted into bytes and then back to a float. When you try and do it again, the column will be floats which don't support being turned back into bytes.