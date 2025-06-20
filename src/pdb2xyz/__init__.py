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
import mdtraj as md
import logging


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
        "-a",
        "--atomfile",
        type=str,
        required=False,
        help="Output atomfile path (default: atoms.yaml)",
        default="atoms.yaml",
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
    return parser.parse_args()


def render_template(context: dict):
    template_str = calvados_template()
    return jinja2.Template(template_str).render(context)


def ssbonds(traj):
    """return set of cysteine indices participating in SS-bonds"""
    bonds = traj.topology.bonds
    ss_bonds = []
    for bond in bonds:
        atom1, atom2 = bond
        if (
            atom1.name == "SG"
            and atom1.residue.name == "CYS"
            and atom2.name == "SG"
            and atom2.residue.name == "CYS"
        ):
            ss_bonds.append((atom1.residue.index, atom2.residue.index))
    return set(res for pair in ss_bonds for res in pair)


def convert_pdb(pdb_file: str, output_xyz_file: str, use_sidechains: bool):
    """Convert PDB to coarse grained XYZ file; one bead per amino acid"""
    traj = md.load_pdb(pdb_file, frame=0)
    cys_with_ssbond = ssbonds(traj)
    residues = []
    for res in traj.topology.residues:
        if not res.is_protein:
            continue

        cm = [0.0, 0.0, 0.0]  # residue mass center
        mw = 0.0  # residue weight
        for a in res.atoms:
            # Add N-terminal
            if res.index == 0 and a.index == 0 and a.name == "N":
                residues.append(dict(name="NTR", cm=traj.xyz[0][a.index] * 10))
                logging.info("Adding N-terminal bead")

            # Add C-terminal
            if a.name == "OXT":
                residues.append(dict(name="CTR", cm=traj.xyz[0][a.index] * 10))
                logging.info("Adding C-terminal bead")

            # Add coarse grained bead
            cm = cm + a.element.mass * traj.xyz[0][a.index]
            mw = mw + a.element.mass

        # rename CYS -> CSS participating in SS-bonds
        if res.name == "CYS" and res.index in cys_with_ssbond:
            name = "CSS"
            logging.info(f"Renaming SS-bonded CYS{res.index} to {name}")
        else:
            name = res.name

        residues.append(dict(name=name, cm=cm / mw * 10))

        if use_sidechains and name != "CSS":
            side_chain = add_sidechain(traj, res)
            if side_chain is not None:
                residues.append(side_chain)

    with open(output_xyz_file, "w") as f:
        f.write(f"{len(residues)}\n")
        f.write(f"Converted with Duello pdb2xyz.py using input file {pdb_file}\n")
        for i in residues:
            f.write(f"{i['name']} {i['cm'][0]:.3f} {i['cm'][1]:.3f} {i['cm'][2]:.3f}\n")
        logging.info(
            f"Converted {pdb_file} -> {output_xyz_file} with {len(residues)} residues."
        )


def add_sidechain(traj, res):
    """Add sidechain bead for ionizable amino acids"""
    for atom in res.atoms:
        # Map residue and atom names to sidechain bead names
        sidechain_map = {
            ("ASP", "OD1"): "Dsc",
            ("GLU", "OE1"): "Esc",
            ("ARG", "CZ"): "Rsc",
            ("LYS", "NZ"): "Ksc",
            ("HIS", "NE2"): "Hsc",
            ("CYS", "SG"): "Csc",
        }
        bead_name = sidechain_map.get((res.name, atom.name))
        if bead_name:
            return dict(name=bead_name, cm=traj.xyz[0][atom.index] * 10)
    return None


def write_atom_file(output_path: str, context: dict):
    """Render and write the atom file template."""
    template = calvados_template()
    rendered = jinja2.Template(template).render(context)
    with open(output_path, "w") as atom_file:
        atom_file.write(rendered)
        logging.info(f"Atom file written to {output_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    convert_pdb(args.infile, args.outfile, args.sidechains)

    context = {
        "pH": args.pH,
        "alpha": args.alpha,
        "sidechains": args.sidechains,
    }
    write_atom_file(args.atomfile, context)


def calvados_template():
    return """
{%- set f = 1.0 - sidechains -%}
{%- set zCTR = - 10**(pH-2.0) / (1 + 10**(pH-2.0)) -%}
{%- set zASP = - 10**(pH-3.9) / (1 + 10**(pH-3.9)) -%}
{%- set zGLU = - 10**(pH-4.2) / (1 + 10**(pH-4.2)) -%}
{%- set zCYS = 10**(pH-8.3) / (1 + 10**(pH-8.3)) -%}
{%- set zHIS = 1 - 10**(pH-6.0) / (1 + 10**(pH-6.0)) -%}
{%- set zNTR = 1 - 10**(pH-9.0) / (1 + 10**(pH-9.0)) -%}
{%- set zLYS = 1 - 10**(pH-10.4) / (1 + 10**(pH-10.4)) -%}
{%- set zARG = 1 - 10**(pH-12.5) / (1 + 10**(pH-12.5)) -%}
comment: "Calvados 3 coarse grained amino acid model for use with Duello / Faunus"
pH: {{ pH }}
sidechains: {{ sidechains }}
version: 0.1.0
atoms:
  - {charge: {{ "%.2f" % zCTR }}, hydrophobicity: !Lambda 0, mass: 0, name: CTR, σ: 3.2, ε: 0.8368, custom: {alpha: {{ alpha }}}}
  - {charge: {{ "%.2f" % zNTR }}, hydrophobicity: !Lambda 0, mass: 0, name: NTR, σ: 3.2, ε: 0.8368, custom: {alpha: {{ alpha }}}}
{%- if sidechains %}
  - {charge: {{ "%.2f" % zGLU }}, hydrophobicity: !Lambda 0, mass: 0, name: Esc, σ: 3.2, ε: 0.8368, custom: {alpha: {{ alpha }}}}
  - {charge: {{ "%.2f" % zASP }}, hydrophobicity: !Lambda 0, mass: 0, name: Dsc, σ: 3.2, ε: 0.8368, custom: {alpha: {{ alpha }}}}
  - {charge: {{ "%.2f" % zHIS }}, hydrophobicity: !Lambda 0, mass: 0, name: Hsc, σ: 3.2, ε: 0.8368, custom: {alpha: {{ alpha }}}}
  - {charge: {{ "%.2f" % zARG }}, hydrophobicity: !Lambda 0, mass: 0, name: Rsc, σ: 3.2, ε: 0.8368, custom: {alpha: {{ alpha }}}}
  - {charge: {{ "%.2f" % zLYS }}, hydrophobicity: !Lambda 0, mass: 0, name: Ksc, σ: 3.2, ε: 0.8368, custom: {alpha: {{ alpha }}}}
  - {charge: {{ "%.2f" % zCYS }}, hydrophobicity: !Lambda 0, mass: 0, name: Csc, σ: 3.2, ε: 0.8368, custom: {alpha: {{ alpha }}}}
{%- endif %}
  - {charge: {{ "%.2f" % (zARG * f) }}, hydrophobicity: !Lambda 0.7407902764839954, mass: 156.19, name: ARG, σ: 6.56, ε: 0.8368, custom: {alpha: {{ f * alpha }}}}
  - {charge: {{ "%.2f" % (zASP * f) }}, hydrophobicity: !Lambda 0.092587557536158,  mass: 115.09, name: ASP, σ: 5.58, ε: 0.8368, custom: {alpha: {{ f * alpha }}}}
  - {charge: {{ "%.2f" % (zGLU * f) }}, hydrophobicity: !Lambda 0.000249590539426,  mass: 129.11, name: GLU, σ: 5.92, ε: 0.8368, custom: {alpha: {{ f * alpha }}}}
  - {charge: {{ "%.2f" % (zLYS * f) }}, hydrophobicity: !Lambda 0.1380602542039267, mass: 128.17, name: LYS, σ: 6.36, ε: 0.8368, custom: {alpha: {{ f * alpha }}}}
  - {charge: {{ "%.2f" % (zHIS * f) }}, hydrophobicity: !Lambda 0.4087176216525476, mass: 137.14, name: HIS, σ: 6.08, ε: 0.8368, custom: {alpha: {{ f * alpha }}}}
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
  - {charge: {{ "%.2f" % (zCYS * f) }}, hydrophobicity: !Lambda 0.5922529084601322, mass: 103.14, name: CYS, σ: 5.48, ε: 0.8368, custom: {alpha: {{ f * alpha }}}}
  - {charge: 0.0, hydrophobicity: !Lambda 0.5922529084601322, mass: 103.14, name: CSS, σ: 5.48, ε: 0.8368}
"""


if __name__ == "__main__":
    main()
