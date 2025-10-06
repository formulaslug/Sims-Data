# How to save data!

1. In the google drive, under ```<car>/TestingData/<date>```, copy the raw .fsdaq files so we preserve a copy of the raw data. To format the date, do ```MMDDYYYY``` so Aug 10, 2025 is ```08102025```. If you don't have internet, make a copy of all the files to your desktop to upload later.

1. Make sure you have the ```fs-data``` repository cloned, that you've pulled recently, and plug the microSD card into your device.

1. Create a folder in ```fs-data``` structured as such to prepare for data copying:

    ```<car>/<date>/```
    
    As an example, the data we collected on Aug 10, 2025 would be ```FS-3/08102025/``` 

1. Decode each individual ```.fsdaq``` file into its ```.parquet``` counterpart with the [fsdaq_decoder.py](../DataDecoding&CorrectionScripts/fsdaq_decoder.py).

## Decoding

In a terminal, navigate to the location of your ```fs-data``` repository ```.../fs-data```. Run the following command for each file:

```
python DataDecoding\&CorrectionScripts/fsdaq_decoder.py -i <path to .fsdaq> -o FS-3/<date>/<useful file name that includes the date>
```

This would typically look like:

```
python DataDecoding\&CorrectionScripts/fsdaq_decoder.py -i /d/fsdaq/1.fsdaq -o FS-3/08172025/08172025short1.parquet
```
 for a file where the car was just turned on and off again or
```
python DataDecoding\&CorrectionScripts/fsdaq_decoder.py -i /d/fsdaq/1.fsdaq -o FS-3/08172025/08172025endurance1P1.parquet
```
for the first part of the first endurance run of the day. Try to stay consistent with file naming schemes by looking at what people have previously done. In the case where the file is too small (will not create output file), just ignore it.

## Last Very Important Step

Upload your .fsdaq files to the drive if you didn't have internet earlier! If you still don't have internet, set youself a reminder somehow to do it later.

Git commit and git push!!!!