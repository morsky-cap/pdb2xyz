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
- Create Calvados3 topology for Duello

## Install

```sh
pip install pdb2xyz
```

## Usage

```sh
usage: pdb2xyz [-h] -i INFILE -o OUTFILE [-t TOP] [--pH PH] [--alpha ALPHA] [--sidechains]

Convert PDB files to XYZ format

options:
  -h, --help            show this help message and exit
  -i INFILE, --infile INFILE
                        Input PDB file path
  -o OUTFILE, --outfile OUTFILE
                        Output XYZ file path
  -t TOP, --top TOP     Output topology path (default: topology.yaml)
  --pH PH               pH value (default: 7.0)
  --alpha ALPHA         Excess polarizability (default: 0.0)
  --sidechains          Off-center ionizable sidechains (default: disabled)
```
