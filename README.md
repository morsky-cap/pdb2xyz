# Convert PDB → Coarse Grained XYZ files

`pdb2xyz` is a small tool to convert atomistic protein structures to coarse grained representations where residues
are reduced to one or two interactions siters.
Meant to construct models for use with the Calvados force field in the Duello and Faunus software.

## Features

- Convert PDB to XYZ
- Optional off-center sites for ionizable side-chains
- N and C terminal handling
- SS-bond handling
- Partial charge approximation according to pH using
[average residue pKa values](https://doi.org/10.1093/database/baz024)
- Create Calvados3 topology for [Duello](https://github.com/mlund/duello)

## Install

```sh
pip install pdb2xyz
```

## Usage

It is recommended that you fix your atomistic PDB file before converting
using e.g. [pdbfixer](https://github.com/openmm/pdbfixer?tab=readme-ov-file).

Default settings use bulk pKa values; you can use [propKa](https://github.com/jensengroup/propka)-predicted values instead.

It is possible to use PQR files instead of PDB files and use precomputed values of charge; these can be obtained using e.g. [pdb2pqr](https://pdb2pqr.readthedocs.io/en/latest/) or [Faunus](https://faunus.readthedocs.io/en/latest/).

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
  --chains [CHAINS ...]
                        List of chain IDs to include (default: all chains)
```
