#!/usr/bin/bash


find Mar22Caps/ -name '*.csv' -exec bash -c './wsAnalysis2.py "$1" &' - {}  \;
