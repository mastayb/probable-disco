#!/usr/bin/python3.6
import csv
import numpy as np
import sys
from collections import namedtuple
from itertools import tee
import matplotlib.pyplot as plot

def ReadCsvToListOfDicts(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        return [row for row in reader]

EntityKinematics = namedtuple('EntityKinematics', ['Time','Position','Velocity'])

def KinmaticsFromWiresharkDict(d):
    p = map(float, [d['dis.entity_location.x'], d['dis.entity_location.y'], d['dis.entity_location.z']])
    v = map(float, [d['dis.entity_linear_velocity.x'], \
            d['dis.entity_linear_velocity.y'], \
            d['dis.entity_linear_velocity.z']])

    return EntityKinematics(Time=float(d['dis.timestamp']), \
            Position=np.array(list(p)), \
            Velocity=np.array(list(v)))

DerivedKinematics = namedtuple('DerivedKinematics', \
        ['Time','Position','Velocity', 'DeltaTime', 'AverageVelocity', 'Speed', 'AverageSpeed', 'SpeedError'])

def pairwise(iterable):
   "s -> (s0,s1), (s1,s2), (s2,s3), ..."
   a, b = tee(iterable)
   next(b, None)
   return zip(a,b)

def CalcAverageVelocity(entityKinematicsA, entityKinematicsB):
   a = entityKinematicsA
   b = entityKinematicsB
   return np.divide(b.Position - a.Position, b.Time - a.Time) 


def CalculateDerivedKinematics(entityKinematicsA, entityKinematicsB):
   a = entityKinematicsA
   b = entityKinematicsB
   dt = b.Time - a.Time 
   avgVel = CalcAverageVelocity(a,b)
   s = np.linalg.norm(b.Velocity)
   sAvg = np.linalg.norm(avgVel)
   sErr = s-sAvg

   return DerivedKinematics( \
           Time = b.Time, \
           Position = b.Position, \
           Velocity = b.Velocity, \
           DeltaTime = dt, \
           AverageVelocity = avgVel, \
           Speed = s, \
           AverageSpeed = sAvg, \
           SpeedError = sErr)


def PlotError(derivedKinematicsList):
    frame = derivedKinematicsList
    plot.plot([f.SpeedError for f in frame])
    plot.show()

def PlotErrorVsTimeDelta(derivedKinematicsList):
    frame = list(derivedKinematicsList)
    plot.figure(1)
    plot.subplot(211)
    x = np.array([f.Time for f in frame])
    y = np.array([f.SpeedError for f in frame])
    plot.plot( x, y)

    plot.subplot(212)
    plot.plot( x, np.array([f.DeltaTime for f in frame]))
    plot.show()


if __name__ == "__main__":
    rawData = ReadCsvToListOfDicts(sys.argv[1])
    kinematics = map(KinmaticsFromWiresharkDict, rawData)
    kinematicsAnalysis = map(lambda p: CalculateDerivedKinematics(*p), pairwise(kinematics))
    PlotErrorVsTimeDelta(kinematicsAnalysis)
