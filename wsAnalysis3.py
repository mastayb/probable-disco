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

def NearestMultiple(x, y):
    """Return the multiple m = a*x where a is an integer s.t. d(m, y) <= d(b*x,y)
    for any integer b != a. Assumes x and y are positive"""
    if x == 0:
        return 0
    c = np.floor(y/x)
    l = c*x
    u = (c+1)*x
    if (y - l <= u - y):
        return l
    else:
        return u
    
def Find60HzDivergence(x):
    harmonic = NearestMultiple((1/60.0), x)
    harmonicNum = harmonic/(1/60.0)
    return np.fabs(harmonic - x) / harmonicNum
    return FindDistanceFromNearestHarmonic(x, (1/60.0))

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

    df['60hz_divergence'] = df['delta.time'].map(Find60HzDivergence)

    df = df.set_index('dis.timestamp')
    #df = df.set_index('_ws.col.absTime')
    return df

def SummaryPlot(df, title):
    plt.figure(); 
    plt.subplot(311); 
    df['speed_error'].plot(); plt.ylabel('Speed Error'); plt.xlabel('Timestamp') 
    plt.title(title)
    plt.subplot(312); 
    df['delta.time'].plot(); plt.ylabel('Time Delta'); plt.xlabel('Timestamp') 
    plt.subplot(313); 
    df['60hz_divergence'].plot(); plt.ylabel('60Hz Divergence'); plt.xlabel('Timestamp')

def VelocitySummaryPlot(df, title):
    plt.figure()
    ax = plt.subplot(511)
    df['speed_error'].plot(ax=ax); plt.ylabel('Speed Error');
    plt.title(title)
    for i,d in ((1,'x'),(2,'y'),(3,'z')):
        ax = plt.subplot(511+i)
        df[['average_velocity.'+d, 'dis.entity_linear_velocity.'+d]].plot(ax=ax, alpha=0.5, color=['r','b'])
        plt.ylabel('Velocity')
    ax = plt.subplot(515)
    df[['speed','average_speed']].plot(ax=ax, alpha=0.5, color=['r','b']); plt.ylabel('Speed');



def DualScatterPlot(df, x_label, y_label1, y_label2, title):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(x_label)
    ax2 = ax1.twinx()
    ax1.set_ylabel(y_label1, color='b')
    ax1.tick_params('y', colors='b')
    ax2.set_ylabel(y_label2, color='r')
    ax2.tick_params('y', colors='r')
    df[y_label1].plot(ax=ax1, style='bv', alpha=0.5)
    df[y_label2].plot(ax=ax2, style='r^', alpha=0.5)
    plt.title(title)
    plt.tight_layout()
    

def PlotData(df):
    SummaryPlot(df, 'All Data')

    VelocitySummaryPlot(df, 'Error Summary')

    dims = ['x','y','z']
    positionPlotLabels = [l+d for d in dims for l in ('dis.entity_location.', 'delta.')]
    positionPlotLabels.insert(0, 'speed_error'); positionPlotLabels.append('delta.time')
    df[positionPlotLabels].plot(subplots=True, title='Positional Data')

    #smallDT = df[df['delta.time'] < 0.2]
    #SummaryPlot(smallDT, 'Time Deltas < 0.2')

    #DualScatterPlot(df, 'Timestamp', 'error_magnitude', '60hz_divergence', "Error Mag vs 60Hz Divergence")
    
    plt.figure()
    ax = plt.subplot()
    df.sort(['60hz_divergence']) \
            .plot(x='60hz_divergence', y='error_magnitude', ax=ax, title='Error Mag vs 60Hz Divergence')
    ax.set_xlim(df['60hz_divergence'].min(), df['60hz_divergence'].max())
    

if __name__ == "__main__":
    df = DataframeFromFile(sys.argv[1])

    df = ProcessData(df)
    #df['speed_error'].plot(style='b.')
    PlotData(df)
    plt.show()



    
