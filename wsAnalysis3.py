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

def ProcessData(df):
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
    return df

def SummaryPlot(df, title):
    plt.figure(); 
    plt.subplot(311); 
    df['speed_error'].plot(); plt.ylabel('Speed Error'); plt.xlabel('Timestamp') 
    plt.title(title)
    plt.subplot(312); 
    df['delta.time'].plot(); plt.ylabel('Time Delta'); plt.xlabel('Timestamp') 
    plt.subplot(313); 
    df['delta.time'].plot.kde(); plt.xlabel('Time Delta')

def DualScatterPlot(df, title):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Timestamp')
    ax2 = ax1.twinx()
    ax1.set_ylabel('Speed Error', color='b')
    ax1.tick_params('y', colors='b')
    ax2.set_ylabel('Time Delta', color='r')
    ax2.tick_params('y', colors='r')
    df['speed_error'].plot(ax=ax1, style='bv', alpha=0.5)
    df['delta.time'].plot(ax=ax2, style='r^', alpha=0.5)
    plt.title(title)
    plt.tight_layout()
    
def PlotData(df):
    SummaryPlot(df, 'All Data')

    smallDT = df[df['delta.time'] < 0.2]
    SummaryPlot(smallDT, 'Time Deltas < 0.2')

    verySmallDT = df[df['delta.time'] < 0.05]
    DualScatterPlot(verySmallDT, 'Time Deltas < 0.05')

    unacceptableError = df[df['error_magnitude'] > 0.2]
    DualScatterPlot(unacceptableError, 'Error Magnitude > 0.2')

    highError = df[df['error_magnitude'] > 0.5]
    DualScatterPlot(highError, 'Error Magnitude > 0.5')
    
    plt.show()

if __name__ == "__main__":
    df = DataframeFromFile(sys.argv[1])

    df = ProcessData(df)

    PlotData(df)


    
