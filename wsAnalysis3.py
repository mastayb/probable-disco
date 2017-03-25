#!/usr/bin/python3.6
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from pandas.tools.plotting import lag_plot
from scipy.stats.stats import pearsonr
import sys


def DataframeFromFile(filename):
    return pd.read_csv(filename, sep='\t')

def GetOneIndexedDiff(df, label):
    oneIndexedDiff = pd.Series(np.diff(df[label]), index=range(1,len(df)))
    return oneIndexedDiff 

def DeriveAverageVelocity(df, dimension):
    avgVel = pd.Series(np.divide(df['delta.'+dimension], df['delta.time']), index= range(1, len(df)))
    return avgVel

def CalcNorm(df, label, dimensions):
    v = df.loc[:,[label+d for d in dimensions]]
    return np.linalg.norm(v, axis=1)

def myPlot(df, title):
    plt.figure(); 
    plt.subplot(311); 
    df['speed_error'].plot(); plt.ylabel('Speed Error'); plt.xlabel('Timestamp') 
    plt.title(title)
    plt.subplot(312); 
    df['delta.time'].plot(); plt.ylabel('Time Delta'); plt.xlabel('Timestamp') 
    plt.subplot(313); 
    df['delta.time'].plot.kde(); plt.xlabel('Time Delta')

def myScatterPlot(df, title):
    plt.figure(); 
    plt.subplot(311); 
    df['speed_error'].plot(style='o'); plt.ylabel('Speed Error'); plt.xlabel('Timestamp')
    plt.title(title)
    plt.subplot(312); 
    df['delta.time'].plot(style='o'); plt.ylabel('Time Delta'); plt.xlabel('Timestamp')
    plt.subplot(313); 
    df['delta.time'].plot.kde(); plt.xlabel('Time Delta')

def myDualScatterPlot(df, title):
    fig, ax = plt.subplots()
    ax1 = ax
    ax1.set_xlabel('Timestamp')
    ax1_b = ax1.twinx()
    ax1.set_ylabel('Speed Error', color='b')
    ax1.tick_params('y', colors='b')
    ax1_b.set_ylabel('Time Delta', color='r')
    ax1_b.tick_params('y', colors='r')
    df['speed_error'].plot(ax=ax1, style='bv', alpha=0.5)
    df['delta.time'].plot(ax=ax1_b, style='r^', alpha=0.5)
    plt.title(title)
    plt.tight_layout()
    


if __name__ == "__main__":
    df = DataframeFromFile(sys.argv[1])

    df['delta.time'] = GetOneIndexedDiff(df, 'dis.timestamp')

    dims = ['x','y','z']
    for d in dims:
        df['delta.'+d] = GetOneIndexedDiff(df, 'dis.entity_location.'+d)
        df['average_velocity.'+d] = DeriveAverageVelocity(df, d)

    df['speed'] = CalcNorm(df, 'dis.entity_linear_velocity.', dims)
    df['average_speed'] = CalcNorm(df, 'average_velocity.', dims)

    df['speed_error'] = df['speed'] - df['average_speed']
    df['error_magnitude'] = df['speed_error'].map(np.fabs)

    df = df.set_index('dis.timestamp')

    myPlot(df, 'All Data')

    smallDT = df[df['delta.time'] < 0.2]
    myPlot(smallDT, 'Time Deltas < 0.2')

    verySmallDT = df[df['delta.time'] < 0.05]
    myDualScatterPlot(verySmallDT, 'Time Deltas < 0.05')

    unacceptableError = df[df['error_magnitude'] > 0.2]
    myDualScatterPlot(unacceptableError, 'Error Magnitude > 0.2')


    highError = df[df['error_magnitude'] > 0.5]
    myDualScatterPlot(highError, 'Error Magnitude > 0.5')

    plt.show()

    
