# -*- coding: utf-8 -*-\
"""
Convert an atlas from FsAverge space to subject space

This program is part of the SimNIBS package.
Please check on www.simnibs.org how to cite our work in publications.

Copyright (C) 2018  Guilherme B Saturnino
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

"""

import argparse
import sys
import os

import nibabel as nib


from simnibs.cli.utils import args_general
from simnibs.utils.file_finder import SubjectFiles
from simnibs.utils.simnibs_logger import logger
from simnibs.utils.transformations import atlas2subject

FILENAME_ATLAS_OUT = "{hemi}.{subid}_{atlas}.annot"


def parse_arguments(argv):
    atlases = """Choose one of the following atlases:

a2009s
------
Destrieux atlas (FreeSurfer v4.5, aparc.a2009s)
Cite: Destrieux, C. Fischl, B. Dale, A., Halgren, E. A sulcal
depth-based anatomical parcellation of the cerebral cortex.
Human Brain Mapping (HBM) Congress 2009, Poster #541

DK40
----
Desikan-Killiany atlas (FreeSurfer, aparc.a2005s)
Cite: Desikan RS, S�gonne F, Fischl B, Quinn BT, Dickerson BC,
Blacker D, Buckner RL, Dale AM, Maguire RP, Hyman BT, Albert MS,
Killiany RJ. An automated labeling system for subdividing the
human cerebral cortex on MRI scans into gyral based regions of
interest. Neuroimage. 2006 Jul 1;31(3):968-80.

HCP_MMP1
--------
Human Connectome Project (HCP) Multi-Modal Parcellation
Cite: Glasser MF, Coalson TS, Robinson EC, et al. A multi-modal
parcellation of human cerebral cortex. Nature. 2016;536(7615):171-178.
"""

    parser = argparse.ArgumentParser(
        prog="subject_atlas",
        description=(
            "Transform an atlas from fsaverage space to subject space. "
            f"Atlases are written in the following format {FILENAME_ATLAS_OUT}"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    args_general.subid.add_to(parser)
    parser.add_argument(
        "-a",
        "--atlas",
        dest="atlas",
        required=True,
        choices=["a2009s", "DK40", "HCP_MMP1"],
        help=atlases,
    )
    args_general.out_dir.add_to(parser)
    args_general.version.add_to(parser)

    return parser.parse_args(argv)


def main():
    args = parse_arguments(sys.argv[1:])
    m2m_dir = os.path.abspath(os.path.realpath(os.path.expanduser(args.subid)))
    if not os.path.isdir(m2m_dir):
        raise IOError("Could not find directory: {0}".format(args.m2mpath))
    subject_files = SubjectFiles(subpath=m2m_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    labels, ctab, names = atlas2subject(
        m2m_dir, args.atlas, return_ctab=True, return_names=True
    )

    for hemi in labels:
        fn_out = os.path.join(
            args.out_dir,
            FILENAME_ATLAS_OUT.format(
                hemi=hemi, subid=subject_files.subid, atlas=args.atlas
            ),
        )
        logger.info("Writing: " + fn_out)
        nib.freesurfer.write_annot(
            fn_out, labels[hemi], ctab[hemi], names[hemi], fill_ctab=True
        )


if __name__ == "__main__":
    main()
