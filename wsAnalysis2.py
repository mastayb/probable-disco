import csv
import numpy as np
import sys

def ReadWiresharkCSVBasic(filename):
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        return [row for row in reader]


if __name__ == "__main__":
    data = ReadWiresharkCSVBasic(sys.argv[1])
    for row in data:
        print(row)
