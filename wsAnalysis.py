from math import sqrt, fabs, atan, cos, sin
import math
import operator
import fileinput
from itertools import tee, izip
import collections
import csv

_sentinel = object()
def next(iterable, default=_sentinel):
   try:
      return iterable.next()
   except StopIteration:
      if default is _sentinel:
         raise
      return default


def pairwise(iterable):
   "s -> (s0,s1), (s1,s2), (s2,s3), ..."
   a, b = tee(iterable)
   next(b, None)
   return izip(a,b)

multisub = lambda a,b: map(operator.sub, a,b)
multidiv = lambda a,b: map(operator.div, a,b)

def isZeroVector(vector):
   for x in vector:
      if x != 0:
         return False
   return True

def ReadData(fileObject):
   fo = fileObject
   velocities = []
   positions = []
   times = []
   timeStamps = []

   while True:
      timeStamp = fo.readline().strip()
      if not timeStamp:
         break
      if ':' in timeStamp:
         min, ms = map(float, timeStamp.split(':'))
         t = min*60 + ms/1000
      else:
         t = float(timeStamp)/1000
     
      times.append(t)
      timeStamps.append(timeStamp)
      velocities.append([float(fo.readline().strip()) for _ in range(3)])
      positions.append([float(fo.readline().strip()) for _ in range(3)])

   return  times,timeStamps, velocities, positions


def CalcAvgVelocities(times, positions):
   posDeltas = [multisub(p1,p0) for p0,p1 in pairwise(positions)]
   timeDeltas = [t1-t0 for t0,t1 in pairwise(times)]
   vels = []
   for dt,dp in zip(timeDeltas,posDeltas):
      if dt :
         vels.append(map(lambda x:x/dt, dp))
      else:
         vels.append([0]*len(positions[0]))

   vels.insert(0, [0,0,0])
   return vels 


def Mean(v):
   return sum(v)/max(len(v), 1)


def calcRMSError(errors):
   rmsErr = map(sqrt, map(lambda y: reduce(lambda a,b: a+b, map(lambda x: x*x,y)), errors))
   return rmsErr


def calcVectorMag(vec):
   return sqrt(sum(map(lambda x: x*x, vec)))


def printListSummaryStats(l, name):
   nameOffset = (40 - len(name))/2
   print "="*nameOffset + name+"="*nameOffset
   print "avg: %20.6f" % Mean(l)
   print "min: %20.6f" % min(l)
   print "max: %20.6f" % max(l) 


multimult = lambda a,b: map(operator.mul, a,b)


def calcPositionChange(times, velocities):
   timeDeltas = [t1-t0 for t0,t1 in pairwise(times)]
   return [map(lambda x: x*dt, v) for v,dt in zip(velocities, timeDeltas)]


def maxIndex(x):
   maxIdx, maxValue = max(enumerate(x), key=operator.itemgetter(1))
   return maxIdx


def ecef_to_lla(x,y,z):
   #wgs84 parameters
   a = 6378137
   e = 8.1819190842622e-2

   asq = math.pow(a,2)
   esq = math.pow(e,2)

   b = math.sqrt( asq * (1-esq) )
   bsq = math.pow(b, 2)
   ep = math.sqrt( (asq - bsq) / bsq )
   p = math.sqrt( math.pow(x,2) + math.pow(y,2))
   th = math.atan2(a*z, b*p)

   lon = math.atan2(y,x)
   lat = math.atan2( (z + math.pow(ep,2)*b*math.pow(math.sin(th),3) ), (p - esq*a*math.pow(math.cos(th),3)) )
   N = a/( math.sqrt(1 - esq*math.pow(math.sin(lat),2)) )
   alt = p / math.cos(lat) - N

   #lon = lon % (2*math.pi)

   return lat, lon, alt
   
def test_ecef_to_lla():
   print "Inputs:   -576793.17, -5376363.47, 3372298.51"
   print "Expected: 32.12345, -96.12345, 500.0"
   lat, lon, alt = ecef_to_lla(-576793.17, -5376363.47, 3372298.51)
   lat = lat*180.0/math.pi
   lon = lon*180.0/math.pi
   print "Actuals:", lat, lon, alt
   print "-----Test 2---------"
   print "Inputs:   2297292.91, 1016894.94, -5843939.62" 
   print "Expected: -66.87654, 23.87654, 1000.0" 
   lat,lon,alt = ecef_to_lla(2297292.91, 1016894.94, -5843939.62) 
   lat = lat*180.0/math.pi 
   lon = lon*180.0/math.pi 
   print "Actuals:", lat, lon, alt 

def rad_to_deg(r):
   return r*180.0/math.pi

def ecefvec_to_llavec(v):
   x,y,z = v
   lat, lon, alt = ecef_to_lla(x,y,z)
   return [lat, lon, alt]

def lla_to_lat_long(v):
   lat, lon, alt = v
   lat = rad_to_deg(lat)
   lon = rad_to_deg(lon)
   return [lat, lon, alt]

def printFormattedNum(label, num):
   print '%-15s  %14.3f' % (label+':', num)

def printFormattedString(label, s):
   print '%-15s  %14s' % (label+':', s)

def printFormatted3Vec(label, vec):
   print '%-15s (%14.3f, %14.3f, %14.3f)' % (label+':', vec[0], vec[1], vec[2])

def highErrorCondition(frame):
   return fabs(frame[len(frame)-1]) > 3.0

def printTopErrorPoints(times, timeStamps, tDeltas, pDeltas, positions, velocities, avgVels, errors):

   frames = zip(times, timeStamps, tDeltas, pDeltas, positions, velocities, avgVels, errors)
   tIdx=0
   tsIdx=1
   dtIdx=2
   dpIdx=3
   pIdx=4
   vIdx=5
   avIdx=6
   eIdx=7

   errorFrames = sorted(frames, key=lambda x: fabs(x[eIdx]), reverse=True)

   numHighErrorFrames = sum( map(highErrorCondition, errorFrames))
   printFormattedNum('Errors > 3', numHighErrorFrames)
   for f in errorFrames[:numHighErrorFrames]: 
      posIdx = list(positions).index(f[pIdx])
      timeIdx = list(times).index(f[tIdx])

      print "-" *80
      printFormattedNum('Error', f[eIdx])
      printFormattedNum('Delta Time', f[dtIdx])
      printFormatted3Vec('AvgVel', f[avIdx])
      printFormatted3Vec('Velocity', f[vIdx])

      if fabs(f[eIdx]) > 10:
         printFormattedNum('Packet No', timeIdx)
         printFormattedNum('Time', f[tIdx])
         printFormattedString('Timestamp', f[tsIdx])
         printFormatted3Vec('Position', f[pIdx])
         printFormatted3Vec('Pos LLA', lla_to_lat_long( ecefvec_to_llavec( f[pIdx])))
         if posIdx > 0:
            printFormatted3Vec('Prev pos', positions[posIdx-1])
            printFormatted3Vec('Prev pos LLA', lla_to_lat_long( ecefvec_to_llavec( positions[posIdx-1])))

         if posIdx + 1 < len(positions):
            printFormatted3Vec('Next pos', positions[posIdx+1])
            printFormatted3Vec('Next pos LLA', lla_to_lat_long( ecefvec_to_llavec( positions[posIdx+1])))

   print "-" *80 

def outPutSummary(data):
   times, timeStamps, tDeltas, pDeltas, positions, velocities, avgVels, errors = zip(*data)

   print "#" * 80 
   print
   printListSummaryStats(tDeltas, "Time deltas")
   printListSummaryStats(map(calcVectorMag, pDeltas), "Entity position change magnitudes")
   printListSummaryStats(map(calcVectorMag, velocities), "Entity linear velocity magnitudes" )
   printListSummaryStats(map(calcVectorMag, avgVels ), "Calculated average velocity magnitudes" )
   printListSummaryStats(errors, "Magnitude errors")
   print
   printTopErrorPoints(times, timeStamps, tDeltas, pDeltas, positions, velocities, avgVels, errors)
   print
  


def processData():
   times, timeStamps, velocities, positions = ReadData(fileinput.input())

   tDeltas = [t1-t0 for t0,t1 in pairwise(times) if t0 != t1]
   posDeltas = [multisub(p1,p0) for p0,p1 in pairwise(positions)]
   avgVels = CalcAvgVelocities(times, positions)
   #first frame is not relevant
   avgVels.pop(0)
   times.pop(0)
   timeStamps.pop(0)
   velocities.pop(0)
   positions.pop(0)

   errors = [v2-v1 for v1, v2 in zip(map(calcVectorMag, velocities), map(calcVectorMag, avgVels))]
   
   return  zip(times, timeStamps, tDeltas, posDeltas, positions, velocities, avgVels, errors)

def outPutCsv(data):
   f =open('wsData.csv', 'w') 
   writer = csv.writer(f)
   collumns = ['Time', 'Timestamp', 'Time Delta', \
         'Position Delta X', 'Position Delta Y', 'Position Delta Z',\
         'Position X', 'Position Y', 'Position Z', \
         'Velocity X','Velocity Y','Velocity Z', \
         'Average Vel X','Average Vel Y', 'Average Vel Z',\
         'Velocity Magnitude', 'Average Velocity Magnitude', 'Error']

   writer.writerow(collumns) 

   times, timeStamps, tDeltas, posDeltas, positions, velocities, avgVels, errors = zip(*data)
   dpX, dpY, dpZ = zip(*posDeltas)
   posX, posY, posZ = zip(*positions)
   vX, vY, vZ = zip(*velocities)
   avX, avY, avZ = zip(*avgVels)
   velMags = map(calcVectorMag, velocities)
   avgVelMags= map(calcVectorMag, avgVels)

   rows = zip(times, timeStamps, tDeltas, dpX, dpY, dpZ, posX, posY, posZ, vX, vY,vZ, avX, avY, avZ, velMags, avgVelMags, errors) 
   writer.writerows(rows)
   f.close()




def outPutJumps(data):
   times, timeStamps, tDeltas, pDeltas, positions, velocities, avgVels, errors = zip(*data)

   print "Index, Time, Latitude, Longitude, Altitude"
   num = 0
   for t,p,e in zip(times, positions,errors):
      if fabs(e) > 100:
         num+=1
         tIndex = list(times).index(t)
         pIndex = list(positions).index(p)
   
         tRange = [i for i in range(tIndex-5, tIndex+6) if i >= 0 and i < len(times)]
         pRange = [i for i in range(pIndex-5, pIndex+6) if i >= 0 and i < len(positions)]
         
         jumpPositions = [positions[i] for i in pRange]
         jumpPositions = map(lla_to_lat_long, map(ecefvec_to_llavec, jumpPositions))

         jumpTimes = [times[i] for i in tRange]

         for jp, jt in zip(jumpPositions, jumpTimes): 
            lat, lon, alt = jp
            print str(num) + "," + str(jt) + "," + str(lat) + "," + str(lon) + "," + str(alt) 

         

      


if __name__ == "__main__":
   data = processData()
#   outPutJumps(data)
   outPutSummary(data) 
   outPutCsv(data)

