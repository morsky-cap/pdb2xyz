#!/usr/bin/env python3

# Copyright 2025 Mikael Lund
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import jinja2
import MDAnalysis as mda
import logging
import itertools
import numpy as np

def parse_args():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description="Convert PDB files to XYZ format")
    parser.add_argument(
        "-i", "--infile", type=str, required=True, help="Input PDB file path"
    )
    parser.add_argument(
        "-o", "--outfile", type=str, required=True, help="Output XYZ file path"
    )
    parser.add_argument(
        "-t",
        "--top",
        type=str,
        required=False,
        help="Output topology path (default: topology.yaml)",
        default="topology.yaml",
    )

    parser.add_argument(
        "--pH", type=float, required=False, help="pH value (default: 7.0)", default=7.0
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        help="Excess polarizability (default: 0.0)",
        default=0.0,
    )
    parser.add_argument(
        "--sidechains",
        action="store_true",
        help="Off-center ionizable sidechains (default: disabled)",
        default=False,
    )
    parser.add_argument(
        "--pqr",
        action="store_true",
        help="Use PQR input format (default: disabled)",
        default=False,
    )
    parser.add_argument(
        "-pka", "--propka", type=str, required=False, help="Input PROPKA file path (default: None)", default=None
    )
    # take list of chain IDs to include (list of strings)
    parser.add_argument(
        "--chains",
        type=str,
        nargs="*",
        required=False,
        help="List of chain IDs to include (default: all chains)",
        default=None,
    )
    return parser.parse_args()


def render_template(context: dict):
    template_str = calvados_template()
    return jinja2.Template(template_str).render(context)


def ssbonds(traj):
    """return set of cysteine indices participating in SS-bonds"""
    bonds = traj.bonds
    ss_bonds = []
    for bond in bonds:
        atom1, atom2 = bond.atoms
        if (
            atom1.name == "SG"
            and atom1.resname == "CYS"
            and atom2.name == "SG"
            and atom2.resname == "CYS"
        ):
            ss_bonds.append((atom1.resid, atom2.resid))
    return set(res for pair in ss_bonds for res in pair)


def convert_pdb(pdb_file: str, output_xyz_file: str, pH: float=7.0, use_sidechains: bool=False, pqr: bool=False, propka: str=None, chains=None):
    """Convert PDB to coarse grained XYZ file; one bead per amino acid"""
    assert not (pqr and propka), "Cannot use both PQR and PROPKA options"

    # load structure with MDAnalysis and move COM to origin
    traj = mda.Universe(pdb_file)
    traj.atoms.translate(-traj.atoms.center_of_mass())

    # keep only protein atoms and (optionally) selected chains; omit hydrogen atoms
    if chains: traj = traj.select_atoms('protein and not name H* and segid %s' % ' '.join(chains))
    else: traj = traj.select_atoms('protein and not name H*')

    # we need to determine bonded CYS only if we don't have PQR or PROPKA input
    if not (pqr or propka):
        cys_with_ssbond = ssbonds(traj)

    # load partial charges via bulk pKa
    if not (pqr or propka): pcr = bulk_charges(pH)
    # load partial charges via propKa
    if propka: pcr = propka_charges(propka,pH)

    # labels and (optionally) positions for charged amino acids
    charge_map = add_charges(use_sidechains)

    residues = []
    charges = {}
    for res in traj.residues:

        cm = res.atoms.center_of_mass() # residue mass center
        mw = res.mass # residue weight

        name = res.resname

        ### Non-electrostatic part of the interaction

        # rename CYS -> CSS participating in SS-bonds
        if not (pqr or propka):
            if res.resname == "CYS" and res.resid in cys_with_ssbond:
                name = "CSS"
                logging.info(f"Renaming SS-bonded CYS{res.resid} to {name}")

        # Add coarse grained bead for non-electrostatic part of the interaction
        residues.append(dict(name=name, cm=cm))

        ### Electrostatic part of the interaction

        # charges via PQR
        if pqr:
            chr_ = 0.0
            for atom in res.atoms: chr_ += np.float32(atom.charge) # the integrated function in MDAnalysis has rounding errors
        # charges via propKa
        elif propka:
            chr_ = pcr.get((name,str(res.resid),res.segid),0.0)
        # default: charges via bulk pKa
        else:
            chr_ = pcr.get(name,0.0)

        # consider only charges above some cutoff
        if abs(chr_) >= 1e-3:
            bead_name, atom_name = charge_map.get(name,(None,None))

            # PQR: we might have a non-ionizable amino acid with a terminal charge
            if pqr and not bead_name:
                bn = 'TRC%i%s' % (res.resid,res.segid)
                residues.append(dict(name=bn, cm=cm))
            
            elif (pqr or propka): bn = '%s%i%s' % (bead_name,res.resid,res.segid)
            else: bn = '%s' % bead_name

            # charge beads at the same positions as amino acid beads
            if not atom_name: residues.append(dict(name=bn, cm=cm))            
            # charge beads positioned at amino acid sidechains
            else: residues.append(dict(name=bn, cm=traj.select_atoms('resid %i and name %s' % (res.resid,atom_name)).positions[0]))

            charges[bn] = chr_

        # charge via pKa: we need to add terminal charges separately
        if not pqr:
            if res.ix == 0:
                bn="N+"
                ntr = traj.select_atoms('atom %s %s N' % (res.segid, res.resid))
                residues.append(dict(name=bn, cm=ntr.center_of_mass()))
                if not propka:
                    chr_ = pcr.get(bn,0.0)
                    if abs(chr_) >= 1e-3: charges[bn] = chr_
                else:
                    chr_ = pcr.get((bn,str(res.resid),res.segid),0.0)
                    if abs(chr_) >= 1e-3: charges[bn] = chr_
            if 'OXT' in res.atoms.names:
                bn="C-"
                oxt = traj.select_atoms('atom %s %s OXT' % (res.segid, res.resid))
                residues.append(dict(name=bn, cm=oxt.center_of_mass()))
                if not propka:
                    chr_ = pcr.get(bn,0.0)
                    if abs(chr_) >= 1e-3: charges[bn] = chr_
                else:
                    chr_ = pcr.get((bn,str(res.resid),res.segid),0.0)
                    if abs(chr_) >= 1e-3: charges[bn] = chr_

    ### Output: write XYZ and return dictionary of charges

    with open(output_xyz_file, "w") as f:
        f.write(f"{len(residues)}\n")
        f.write(
            f"Converted with Duello pdb2xyz.py with {pdb_file} (https://github.com/mlund/pdb2xyz)\n"
        )
        for i in residues:
            f.write(f"{i['name']} {i['cm'][0]:.3f} {i['cm'][1]:.3f} {i['cm'][2]:.3f}\n")
        logging.info(
            f"Converted {pdb_file} -> {output_xyz_file} with {len(residues)} residues."
        )

    return charges


def propka_charges(propka_file,pH):
    """Obtain partial charges from PROPKA output file"""
    # Read relevant section of PROPKA output file and store in result
    with open(propka_file) as fp:
        result = list(itertools.takewhile(lambda x: '---' not in x, 
            itertools.dropwhile(lambda x: 'Group' not in x, fp)))

    result=np.array([line.split()[:4] for line in result[1:]])

    negative=['C-','ASP','GLU','TYR','CYS']
    positive=['N+','ARG','LYS','HIS']

    # Dictionary with (AA,resid,segid) as key and charge as value
    pcr={}

    for i,entry in enumerate(result):

        AA=str(entry[0])
        resid=str(entry[1])
        segid=str(entry[2])
        pKa=float(entry[-1])

        if AA in negative: pcr[(AA,resid,segid)] = - 10**(pH-pKa) / (1 + 10**(pH-pKa))
        elif AA in positive: pcr[(AA,resid,segid)] = 1.0 - 10**(pH-pKa) / (1 + 10**(pH-pKa))
        else: continue

    return pcr


def bulk_charges(pH):
    """Obtain standard charges for amino acids at given pH using bulk pKa values"""

    negative=['C-','ASP','GLU','TYR','CYS']
    positive=['N+','ARG','LYS','HIS']

    # Average pKa values from https://doi.org/10.1093/database/baz024
    pKa={
        'C-':3.16,
        'ASP':3.43,
        'GLU':4.14,
        'TYR':10.1,
        'CYS':8.3,
        'N+':7.64,
        'ARG':12.5,
        'LYS':10.68,
        'HIS':6.45
    }

    # Dictionary with AA as key and charge as value
    pcr={}

    for AA in negative: pcr[AA] = - 10**(pH-pKa[AA]) / (1 + 10**(pH-pKa[AA]))
    for AA in positive: pcr[AA] = 1.0 - 10**(pH-pKa[AA]) / (1 + 10**(pH-pKa[AA]))

    return pcr


def add_charges(use_sidechains=False):
    """Add charge bead for ionizable amino acids"""
    if not use_sidechains:
        charge_map = {
            "ASP": ("Dch", None),
            "GLU": ("Ech", None),
            "TYR": ("Tch", None),
            "ARG": ("Rch", None),
            "LYS": ("Kch", None),
            "HIS": ("Hch", None),
            "CYS": ("Cch", None),
        }
    # Map residue to sidechain bead names and charged atoms
    else:
        charge_map = {
            "ASP": ("Dsc", "OD1"),
            "GLU": ("Esc", "OE1"),
            "TYR": ("Tsc", "OH"),
            "ARG": ("Rsc", "CZ"),
            "LYS": ("Ksc", "NZ"),
            "HIS": ("Hsc", "NE2"),
            "CYS": ("Csc", "SG"),
        }
    
    return charge_map


def write_topology(output_path: str, context: dict, cdict: dict):
    """Render and write the topology template."""
    template = calvados_template()
    rendered = jinja2.Template(template).render(context,cdict=cdict)
    with open(output_path, "w") as file:
        file.write(rendered)
        logging.info(f"Topology written to {output_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    charges=convert_pdb(args.infile, args.outfile, args.sidechains, args.pqr, args.propka, args.chains)

    context = {
        "pH": args.pH,
        "alpha": args.alpha,
        "sidechains": args.sidechains,
    }
    write_topology(args.top, context, cdict=charges)
    

# Average pKa values from https://doi.org/10.1093/database/baz024
def calvados_template():
    return """
{%- set f = 1.0 - sidechains -%}
comment: "Calvados 3 coarse grained amino acid model for use with Duello / Faunus"
pH: {{ pH }}
sidechains: {{ sidechains }}
version: 0.1.0
atoms:
{% for name, charge in cdict.items() -%}
{%- if not loop.last -%}
{{"  - "}}{charge: {{ "%.2f" % charge }}, hydrophobicity: !Lambda 0, mass: 0, name: {{ "%s" % name }}, σ: 2.0, ε: 0.8368}{{"\n"}}
{%- else -%}
{{"  - "}}{charge: {{ "%.2f" % charge }}, hydrophobicity: !Lambda 0, mass: 0, name: {{ "%s" % name }}, σ: 2.0, ε: 0.8368}
{%- endif -%}
{%- endfor %}
  - {charge: 0.0, hydrophobicity: !Lambda 0.7407902764839954, mass: 156.19, name: ARG, σ: 6.56, ε: 0.8368, custom: {alpha: {{ f * alpha }}}}
  - {charge: 0.0, hydrophobicity: !Lambda 0.092587557536158,  mass: 115.09, name: ASP, σ: 5.58, ε: 0.8368, custom: {alpha: {{ f * alpha }}}}
  - {charge: 0.0, hydrophobicity: !Lambda 0.000249590539426,  mass: 129.11, name: GLU, σ: 5.92, ε: 0.8368, custom: {alpha: {{ f * alpha }}}}
  - {charge: 0.0, hydrophobicity: !Lambda 0.1380602542039267, mass: 128.17, name: LYS, σ: 6.36, ε: 0.8368, custom: {alpha: {{ f * alpha }}}}
  - {charge: 0.0, hydrophobicity: !Lambda 0.4087176216525476, mass: 137.14, name: HIS, σ: 6.08, ε: 0.8368, custom: {alpha: {{ f * alpha }}}}
  - {charge: 0.0, hydrophobicity: !Lambda 0.3706962163690402, mass: 114.1,  name: ASN, σ: 5.68, ε: 0.8368}
  - {charge: 0.0, hydrophobicity: !Lambda 0.3143449791669133, mass: 128.13, name: GLN, σ: 6.02, ε: 0.8368}
  - {charge: 0.0, hydrophobicity: !Lambda 0.4473142572693176, mass: 87.08,  name: SER, σ: 5.18, ε: 0.8368}
  - {charge: 0.0, hydrophobicity: !Lambda 0.7538308115197386, mass: 57.05,  name: GLY, σ: 4.5,  ε: 0.8368}
  - {charge: 0.0, hydrophobicity: !Lambda 0.2672387936544146, mass: 101.11, name: THR, σ: 5.62, ε: 0.8368}
  - {charge: 0.0, hydrophobicity: !Lambda 0.3377244362031627, mass: 71.07,  name: ALA, σ: 5.04, ε: 0.8368}
  - {charge: 0.0, hydrophobicity: !Lambda 0.5170874160398543, mass: 131.2,  name: MET, σ: 6.18, ε: 0.8368}
  - {charge: 0.0, hydrophobicity: !Lambda 0.950628687301107,  mass: 163.18, name: TYR, σ: 6.46, ε: 0.8368}
  - {charge: 0.0, hydrophobicity: !Lambda 0.2936174211771383, mass: 99.13,  name: VAL, σ: 5.86, ε: 0.8368}
  - {charge: 0.0, hydrophobicity: !Lambda 1.033450123574512,  mass: 186.22, name: TRP, σ: 6.78, ε: 0.8368}
  - {charge: 0.0, hydrophobicity: !Lambda 0.5548615312993875, mass: 113.16, name: LEU, σ: 6.18, ε: 0.8368}
  - {charge: 0.0, hydrophobicity: !Lambda 0.5130398874425708, mass: 113.16, name: ILE, σ: 6.18, ε: 0.8368}
  - {charge: 0.0, hydrophobicity: !Lambda 0.3469777523519372, mass: 97.12,  name: PRO, σ: 5.56, ε: 0.8368}
  - {charge: 0.0, hydrophobicity: !Lambda 0.8906449355499866, mass: 147.18, name: PHE, σ: 6.36, ε: 0.8368}
  - {charge: 0.0, hydrophobicity: !Lambda 0.5922529084601322, mass: 103.14, name: CYS, σ: 5.48, ε: 0.8368, custom: {alpha: {{ f * alpha }}}}
  - {charge: 0.0, hydrophobicity: !Lambda 0.5922529084601322, mass: 103.14, name: CSS, σ: 5.48, ε: 0.8368}

system:
  energy:
    nonbonded:
      # Note that a Coulomb term is automatically added, so don't specify one here!
      default:
        - !AshbaughHatch {mixing: arithmetic, cutoff: 20.0}
"""


if __name__ == "__main__":
    main()
