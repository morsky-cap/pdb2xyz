#!/bin/bash

INFILE=$1
C0=$2

duello scan -1 ${INFILE}.xyz -2 ${INFILE}.xyz --rmin 50 --rmax 152 --dr 2.0 --resolution 0.6 --top ${INFILE}.yaml --molarity $C0 --cutoff 1000.0 --pmf ${INFILE}_c0${C0}_pmf.dat
