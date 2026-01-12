# Sims-Data

Vehicle Dynamics Simulations and Data Analysis for Formula Slug written from the ground up in python.



\# Blue Max Turn Speed Analysis



This script analyzes vehicle turn speed against duration at various designated turn points on the Blue Max Track to figure out any statistical correlations/ calculations that could be derived from them and provide the driver with better feedback.



Status:

I have successfully created a working script for a single blue max track session file that runs the necessary data and outputs a graph depicting the project scope. 

However, I am stuck while trying to aggregate this code onto all the Blue Max sessions through the combined file that Nathaniel made. I'm running into primarily 

lap designation errors where the code isn't depicting all the laps involved in the file as valid data points and ignores most of the laps' data. After trying to debug

I think it has something to do with the fact that I use static GPS points based on points on the track and when using the combined file which is not just a single continuous

session, these gps points skew resulting in invalid data. If this is true, I don't know how to address the problem but in the meantime I was thinking of individually analyzing 

each session (as my code does run for them that way) and aggregating the data at the end.

