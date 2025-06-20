# PDB2XYZ

Tool to convert atomistic protein structures to
a coarse grained representation where residues
are described by one or two beads only.
Meant to be used with Duello and/or Faunus with
e.g. the Calvados force field.

## Features

- Convert PDB to XYZ
- Optional off-center sites for ionizable side-chains
- N and C terminal handling
- SS-bond handling
- Create atom file definitions file for Duello / Faunus
- Partial charge approximation
- Create Calvados3 atom list for Duello / Faunus

## Usage

```sh
usage: pdb2xyz [-h] -i INFILE -o OUTFILE [-a ATOMFILE] [--pH PH] [--alpha ALPHA] [--sidechains]

Convert PDB files to XYZ format

options:
  -h, --help            show this help message and exit
  -i INFILE, --infile INFILE
                        Input PDB file path
  -o OUTFILE, --outfile OUTFILE
                        Output XYZ file path
  -a ATOMFILE, --atomfile ATOMFILE
                        Template file path
  --pH PH               pH value (float)
  --alpha ALPHA         Excess polarizability (float)
  --sidechains          Off-center ionizable sidechains (flag)
```
