#!/bin/bash

INFILE=$1
C0=$2

duello scan -1 ${INFILE}.xyz -2 ${INFILE}.xyz --rmin 35 --rmax 101 --dr 1.0 --resolution 0.5 --top ${INFILE}.yaml --molarity $C0 --cutoff 1000.0 --pmf ${INFILE}_c0${C0}_pmf.dat
