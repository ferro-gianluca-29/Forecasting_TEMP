"""
1) Put this script and your data file in the same folder.
2) Make sure your data file is named 'my_data.csv' or adjust the script accordingly.
@author: Sushil Silwal, ssilwal@ucsd.edu 
Modified to work with the provided dataset.
"""
import datetime
import time
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

t = time.time()

#---------------- Data Input Section ---------------------------------
# Read in your dataset
data_in = pd.read_csv('./data/Dataset/Consumer_power.csv', sep=',', decimal='.', parse_dates=['Data'], dayfirst=True)

# Rename columns to match the expected names in the script
data_in.rename(columns={'Data': 'DateTime', 'Cons 1': 'RealPower'}, inplace=True)

#------------------------------------------------------------------------------
            
# Separating datetime and power data
TimeStamp = data_in['DateTime']
RealPower = data_in['RealPower']

# Compute the start and end times
StartTime = TimeStamp.min()
EndTime = TimeStamp.max()

tDelta = datetime.timedelta(minutes=15)
TimeAll = []
current = StartTime

# Generating all timestamps within the data range at 15-minute intervals
while current <= EndTime:
    TimeAll.append(current)
    current = current + tDelta

# Identifying missing timestamps
TimeMiss = list(set(TimeAll) - set(TimeStamp))
print('Number of missing datapoints: ' + str(len(TimeMiss)) + '\n')   
print('Equivalent missing days: ' + str(len(TimeMiss)/96) + '\n')

# Adding missing data points with zero power
df_miss = pd.DataFrame()
df_miss['DateTime'] = TimeMiss
df_miss['RealPower'] = 0
data_in = pd.concat([data_in, df_miss]).sort_values('DateTime', ascending=True).reset_index(drop=True)
data_in.sort_values('DateTime', ascending=True, inplace=True, ignore_index=True)

# Smoothing the data
data_smooth = data_in.copy()
data_smooth.RealPower = gaussian_filter1d(data_smooth.RealPower, 1)
pMax = max(data_smooth.RealPower)  # Maximum power generation

# Replace error data with suitable data
error_data = pd.DataFrame()

# First day correction
for i in range(1, 96):  # Skipping the first datapoint
    if data_in.RealPower[i] < -3 or data_in.RealPower[i] > 1.1 * pMax: 
        error_data = error_data._append(data_in.loc[i])  
        data_in.RealPower[i] = data_in.RealPower[i-1]             
    if data_in.RealPower[i] > 1 and data_in.DateTime[i].hour in [21, 22, 23, 0, 1, 2, 3, 4]:     
        error_data = error_data._append(data_in.loc[i])
        data_in.RealPower[i] = data_in.RealPower[i-1]        

# Other day error correction
for i in range(97, len(data_in.RealPower)):    
    if data_in.RealPower[i] < -3 or data_in.RealPower[i] > 1.1 * pMax:      
        error_data = error_data._append(data_in.loc[i])
        data_in.RealPower[i] = data_in.RealPower[i-96]        
    if data_in.RealPower[i] > 1 and data_in.DateTime[i].hour in [21, 22, 23, 0, 1, 2, 3, 4]: 
        error_data = error_data._append(data_in.loc[i])
        data_in.RealPower[i] = data_in.RealPower[i-96]       
    if data_in.DateTime[i].hour == 0 and data_in.DateTime[i].minute == 0:
        if max(data_in.RealPower[i-96:i]) <= 0.1 and i > 96*2:
            data_in.RealPower[i-96:i] = data_in.RealPower[i-96*2:i-96] 

# Saving the processed data
data_in.to_csv('./data/Dataset/Data_PV_Processed.csv', index=False, header=True)     

# Plotting the data
plt.plot(data_in.DateTime, data_in.RealPower)
if len(error_data) > 0:
    plt.plot(error_data.DateTime, error_data.RealPower, '*')

print('Number of error datapoints replaced: ' + str(len(error_data)) + '\n')      
print('Time taken to process data: ' + str(time.time() - t))
plt.show()
