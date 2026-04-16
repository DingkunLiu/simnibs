# -*- coding: utf-8 -*-
"""
Command line tool for opening the GMSH GUI.
Gets automatically linked into the binary folder during installation.
"""
import sys
import gmsh

gmsh.initialize(sys.argv, run=True)
gmsh.finalize()
