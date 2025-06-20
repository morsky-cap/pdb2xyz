# Convert PDB → Coarse Grained XYZ files

`pdb2xyz` is a small tool to convert atomistic protein structures to coarse grained representations where residues
are reduced to one or two interactions siters.
Meant to construct models for use with the Calvados force field in the Duello and Faunus software.

## Features

- Convert PDB to XYZ
- Optional off-center sites for ionizable side-chains
- N and C terminal handling
- SS-bond handling
- Partial charge approximation according to pH
- Create Calvados3 atom list for Duello / Faunus

## Install

```sh
pip install pdb2xyz
```

## Usage

```sh
usage: pdb2xyz [-h] -i INFILE -o OUTFILE [-a ATOMFILE] [--pH PH] [--alpha ALPHA] [--sidechains]

Convert PDB files to coarse grained XYZ format with one or two beads per residue

options:
  -h, --help            show this help message and exit
  -i INFILE, --infile INFILE
                        Input PDB/GRO/XYZ file path
  -o OUTFILE, --outfile OUTFILE
                        Output XYZ file path
  -a ATOMFILE, --atomfile ATOMFILE
                        Output atomfile path (default: atoms.yaml)
  --pH PH               pH value (default: 7.0)
  --alpha ALPHA         Excess polarizability (default: 0.0)
  --sidechains          Off-center ionizable sidechains (default: disabled)
```
