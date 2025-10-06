# Ideas

## Nathaniel

### Reach out to West Mountain Radio for battery tester

For the $100 one, you can test constant current till cutoff voltage, change current draw during test with GUI, graph V vs Time, charge discharge cycles.

For the $1k one (really for the software), it can measure intenal resistance, constant power output, duty cycle test (and generally pulse testing). 
1. [Testers](https://www.westmountainradio.com/cba.php)
1. [$1k product](https://www.westmountainradio.com/product_info.php?products_id=sys500_watt)  
1. [West Mountain Radio Software Comparison](https://www.westmountainradio.com/pdf/cba-test-modes.pdf)

Should try and get it sponsored or find someone who has one (maybe ask Russell?).

### Lap Segment Breakdown

Segment a track into turns and straights that can be automatically be determined based on GPS data. For each segment, determine the amount of energy used, RMS/mean current, and distribution of power consumption throughout the segment. This could be displayed to drivers to let them test different things on a lap to determine the optimum for the car.

Ask people how to break up BlueMax into curves and straights so I can do it in the future.

### SOC Estimation

Do an estimation based on charge used in the last ```x``` time to determine a "current" that can be compared to the discharge curve for an accurate SOC. 

### VDM Data Quality Improvement

1. Make a proper mount for the IMU
1. Make an aluminum grounded faraday cage around the IMU

### Data Driven Battery Model

Fit a few cubic functions to charge used in the last 20 sec, current draw, # of discharges, and SOC\

###

Button on the car to mark beginning of events (testing runs, comp, etc.)