#!/usr/bin/python3.6
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import sys

def DataframeFromFile(filename):
    return pd.read_csv(filename, sep='\t')

def DeriveTimeDelta(df):
    dt = pd.Series(np.diff(df['dis.timestamp']), index=range(1,len(df)))
    return dt

def DeriveLocationDeltas(df):
    dX = pd.Series(np.diff(df['dis.entity_location.x']), index= range(1,len(df)))
    dY = pd.Series(np.diff(df['dis.entity_location.y']), index= range(1,len(df)))
    dZ = pd.Series(np.diff(df['dis.entity_location.z']), index= range(1,len(df)))
    return dX, dY, dZ


if __name__ == "__main__":
    df = DataframeFromFile(sys.argv[1])
    df['deltaTime'] = DeriveTimeDelta(df)
    dX, dY, dZ = DeriveLocationDeltas(df)
    df['deltaX'] = dX
    df['deltaY'] = dY
    df['deltaZ'] = dZ

    print(df.describe())
