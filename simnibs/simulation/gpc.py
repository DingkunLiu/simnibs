# -*- coding: utf-8 -*-\
'''
    Generalized Polynomial Chaos things for SimNIBS
    This program is part of the SimNIBS package.
    Please check on www.simnibs.org how to cite our work in publications.
    Copyright (C) 2017, 2018 Konstantin Weise, Guilherme B Saturnino, Sicheng An

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''

from __future__ import print_function
import os

import h5py
import numpy as np
import copy
from collections import OrderedDict
from simnibs.simulation.tms_coil.tms_coil import TmsCoil

from simnibs.utils.mesh_element_properties import ElementTags

import pygpc
from pygpc.SGPC import Reg
from pygpc.AbstractModel import AbstractModel
import inspect

from ..mesh_tools import mesh_io
from .sim_struct import SimuList
from . import fem
from simnibs.utils.simnibs_logger import logger

FIELD_NAME = {"v": "v", "E": "E", "e": "magnE", "J": "J", "j": "magnJ"}


def write_data_hdf5(data, data_name, hdf5_fn, path="data/", compression="gzip"):
    """Saves a field in an hdf5 file

    Parameters:
    ---------------
    data: np.array
        data to be saved
    data_name: str
        name of data
    path: str (optional)
        path inside hdf5 file (default: data/)
    """
    with h5py.File(hdf5_fn, "a") as f:
        f.create_dataset(path + data_name, data=data, compression=compression)

def read_data_hdf5(data_name, hdf5_fn, path='data/'):
    ''' Saves a field in an hdf5 file

    Parameters:
    ---------------
    data_name: str
        name of data
    hdf5_fn: str
        name of hdf5 file
    path: str (optional)
        path inside hdf5 file (default: data/)
    '''
    with h5py.File(hdf5_fn, "r") as f:
        return np.squeeze(f[path + data_name])


class gPC_regression(Reg):
    ''' Class defining the parameters of a gPC regression

    Inherits from pygpc's "Reg" class

    Parameters:
    ------------
    problem: pygpc.problem
        problem object in pygpc, including gPC problem to investigate
    regularization_factors: ndarray [n_factors]
        fators used for Tikhonov regularization for coefficients calculation
    multi_indices: ndarray [n_basis x dim]
        multi-indices of polynomial basis functions
    coords_norm: ndarray of float [n_grid x dim]
        array of sampled points in normalized space
    sim_type: {'TMS', 'TCS'}
        Type of simulation
    data_file: str, optional
        Path to file with raw data
    '''
    def __init__(self, problem, regularization_factors, multi_indices, coords_norm, sim_type, data_file=None,n_cpu=None):

        if sim_type in ["TMS", "TCS"]:
            self._sim_type = sim_type
        else:
            raise ValueError("invalid sim_type")

        self.random_vars = [int(key.split('_')[1]) for key in problem.parameters_random.keys()]
        # initialize Reg gPC object
        options = dict()
        options["error_type"] = "kcv"
        options["solver"] = "Tikhonov"
        options["backend"] = "python"
        options["n_cpu_comp"] = n_cpu
        options["n_cpu_basis"] = 1
        super().__init__(problem, 0, 0 * np.zeros(problem.dim), 1,
                         problem.dim, options, problem.dim, None)
        # initialize grid class
        self.coords_norm = coords_norm
        self.grid = pygpc.Random(parameters_random=problem.parameters_random, coords_norm= coords_norm)

        # enrich polynomial basis
        self.multi_indices = multi_indices
        self.basis.add_basis_poly_by_order(multi_indices=multi_indices, problem=problem)
        # Update gPC matrix
        self.init_gpc_matrix()
        # set up data (HDF5) file
        self.data_file = data_file

        self.regularization_factors = regularization_factors

    @property
    def sim_type(self):
        """Simulation type, TMS or TCS"""
        return self._sim_type

    @property
    def mesh_file(self):
        """Name of mesh file"""
        return os.path.splitext(self.data_file)[0] + ".msh"

    def postprocessing(self, postprocessing_type, order_sobol_max=1, normalize_sobol=False):
        ''' Postprocessing for TMS/tDCS gPC operation
        Makes the operation for the postprodesssing and saves in in the data_file

        Parameters
        -----------------------
        postprocessing_type: str
            A combination of 'v'(potential), 'E'(electric field vector),
            'e'(electric field magnitude), 'J'(current density vector) and
            'j'(current density magnitude) eg: 'eEJ'

        order_sobol_max: int (Optional)
            Maximum number of components for Sobol coefficients. Default: 1
            
        normalize_sobol: bool (Optional)
            whether to normalize the Sobol indices relative to the variance. Default: False
            
        Returns
        -----------------------
        Writes mean, std, expansion coefficiets, sobol coefficiets, sensitivity and
        global sensitivity to file
        '''

        if self.data_file is None:
            raise ValueError("Please set a data_file before running the postprocessing")
        if not os.path.exists(self.data_file):
            raise ValueError("Could not find the data_file: {0}".format(self.data_file))
        # make the vector into a list
        postprocessing_type = [p for p in postprocessing_type]
        for p in postprocessing_type:
            if p not in ["v", "E", "e", "J", "j"]:
                raise ValueError("Unrecognized postprocessing type: {0}".format(p))

        # load the cropped mesh
        msh = mesh_io.Msh.read_hdf5(self.data_file, "mesh_roi/")

        lst = SimuList()
        lst._get_conductivity_from_hdf5(self.data_file)
        lst.mesh = msh

        potentials = read_data_hdf5(
            "v_samples", self.data_file, "mesh_roi/data_matrices/"
        )

        nr_simu = potentials.shape[0]

        """
        if 'v' in postprocessing_type:
            self._postprocessing_core(potentials, 'mesh_roi/nodedata', False)
'mesh_roi/' + dtype + '/',
                FIELD_NAME[p],
                order_sobol_max
            postprocessing_type.remove('v')
        """

        # This will NOT work when changing positions
        if self.sim_type == "TMS":
            dAdt = read_data_hdf5("dAdt", self.data_file, "mesh_roi/elmdata/")
            dAdt = mesh_io.ElementData(dAdt, mesh=msh)
        else:
            dAdt = None

        # See which fields have already been calculated
        fields_dict = dict.fromkeys(postprocessing_type)
        to_calc = []
        for postprocess in postprocessing_type:
            try:
                fields_dict[postprocess] = read_data_hdf5(
                    FIELD_NAME[postprocess] + "_samples",
                    self.data_file,
                    "mesh_roi/data_matrices/",
                )
            except:
                if postprocess in ["e", "j"]:
                    fields_dict[postprocess] = np.nan * np.ones(
                        (nr_simu, msh.elm.nr), dtype=float
                    )
                elif postprocess in ["E", "J"]:
                    fields_dict[postprocess] = np.nan * np.ones(
                        (nr_simu, msh.elm.nr, 3), dtype=float
                    )
                elif postprocess in ["v"]:
                    fields_dict[postprocess] = np.nan * np.ones(
                        (nr_simu, msh.nodes.nr), dtype=float
                    )
                else:
                    raise ValueError(
                        "Unrecognized postprocessing option: " + postprocess
                    )
                to_calc.append(postprocess)

        # Calculate the ones which have not been calculated
        # This step is slow! I've opted for greater flexibility and reliability rather
        # than just speed. it uses calc_fields
        if len(to_calc) > 0:
            logger.info("Calculating fields: {0}".format("".join(to_calc)))
            for i, phi, gr in zip(range(nr_simu), potentials, self.grid.coords):
                # set the conductivities right
                for rv, g in zip(self.random_vars, gr):
                    if isinstance(rv, int):
                        lst.cond[rv - 1].value = g
                elmdata = lst.cond2elmdata(logger_level=10)
                # sets the potential
                pot = mesh_io.NodeData(phi, mesh=msh)
                # calclate the remaining fields
                # We can use 'E' to calculate the remaining fields, if it has been
                # defined
                if "E" in fields_dict.keys() and not np.all(
                    np.isnan(fields_dict["E"][i])
                ):
                    m = fem.calc_fields(
                        pot, to_calc, cond=elmdata, dadt=dAdt, E=fields_dict["E"][i]
                    )
                # Else we also need to calculate E
                else:
                    m = fem.calc_fields(pot, to_calc, cond=elmdata, dadt=dAdt)

                for p in to_calc:
                    fields_dict[p][i] = m.field[FIELD_NAME[p]].value

        for p, f in fields_dict.items():
            logger.info("Expanding field: {0}".format(FIELD_NAME[p]))
            if p == "v":
                dtype = "nodedata"
            else:
                dtype = "elmdata"
            self._postprocessing_core(
                f, "mesh_roi/" + dtype + "/", FIELD_NAME[p], order_sobol_max, normalize_sobol
            )

    def _postprocessing_core(self, data, path, name, order_sobol_max, algorithm="standard", n_samples=1e5, normalize_sobol=False):
        ''' Convinience function to calculate postprocessing output '''
        with h5py.File(self.data_file, 'a') as f:
            try:
                f.create_dataset(
                    "mesh_roi/data_matrices/" + name + "_samples",
                    data=data,
                    compression="gzip",
                )
            except (RuntimeError, OSError, ValueError):
                pass

        data_dims = data.shape[1:]
        if data.ndim == 3:
            data = data.reshape(data.shape[0], -1)

        coeffs, error = self.expand(data)
        logger.info("Estimated error: {0:1e} for field {1}".format(error, name))

        # Mean and std
        mean = self.get_mean(coeffs=coeffs)
        std = self.get_std(coeffs=coeffs)
        # Sobol
        sobol, sobol_idx, _ = self.get_sobol_indices(coeffs, normalize_sobol=normalize_sobol)
            
        # Filter sobol
        sobol_order = np.array([len(idx) for idx in sobol_idx])
        sobol = sobol[sobol_order <= order_sobol_max]
        sobol_idx = [s_idx for s_idx in sobol_idx if len(s_idx) <= order_sobol_max]
        # Sensitivity
        sens = self.get_global_sens(coeffs)

        # Put everything in Data structures
        mean = mesh_io.Data(mean.reshape(*data_dims), name=name + "_mean")
        mean.write_hdf5(self.data_file, path=path)
        std = mesh_io.Data(std.reshape(*data_dims), name=name + "_std")
        std.write_hdf5(self.data_file, path=path)
        for s, s_idx in zip(sobol, sobol_idx):
            s_name = (
                name
                + "_sobol_"
                + "_".join([str(self.random_vars[s_i]) for s_i in s_idx])
            )
            sobol_dat = mesh_io.Data(s.reshape(*data_dims), name=s_name)
            sobol_dat.write_hdf5(self.data_file, path=path)

        for i, s in enumerate(sens):
            s_name = name + "_sensitivity_" + str(self.random_vars[i])
            sens_dat = mesh_io.Data(s.reshape(*data_dims), name=s_name)
            sens_dat.write_hdf5(self.data_file, path=path)

    def save_hdf5(self, data_file=None):
        """ Saves the gPC information in an hdf5 file
        problem, options, multi_indices, coords_norm, sim_type are saved in
        "gpc_object" in the data file

        Parameters:
        -----------------
        data_file (optional): str
            name of data file. default: self.data_file

        """
        if data_file is None:
            data_file = self.data_file
        if data_file is None:
            raise ValueError('Please specify a data file')
        pdftype = [param.pdf_type for param in self.problem.parameters_random.values()]
        pdfshape = [param.pdf_shape for param in self.problem.parameters_random.values()]
        limits = [param.pdf_limits for param in self.problem.parameters_random.values()]

        for i, pt in enumerate(pdftype):
            if pt in ['norm']:
                limits[i] = [-1e10, 1e10]

        with h5py.File(data_file, 'a') as f:
            f.attrs['type'] = self.sim_type
            if 'gpc_object' in f.keys():
                del f['gpc_object']
            # save random parameters
            f.create_dataset('gpc_object/random_vars',
                             data=np.array(self.random_vars, dtype=np.bytes_))
            f.create_dataset('gpc_object/pdftype',
                             data=np.array(pdftype, dtype='S10'))
            f.create_dataset('gpc_object/pdfshape', data=np.array(pdfshape))
            f.create_dataset('gpc_object/limits', data=np.array(limits))
            f.create_dataset('gpc_object/poly_idx',
                             data=np.array(self.multi_indices))
            f.create_dataset('gpc_object/grid/coords_norm',
                             data=np.array(self.grid.coords_norm))
            f.create_dataset('gpc_object/regularization_factors',
                             data=np.array(self.regularization_factors))


    @classmethod
    def read_hdf5(cls, fn_hdf5, cpus=None):
        """Reads gPC information from hdf5 file
        Information must have the same format as in gPC_regression.save_hdf5

        Parameteres:
        -----------------
        fn_hdf5: str
            Name of hdf5 file

        Returns:
        -----------------
        gPC_regression: gPC_regression
            regression object
        """
        with h5py.File(fn_hdf5, "r") as f:
            sim_type = f.attrs["type"]
            random_vars = f["gpc_object/random_vars"][()].tolist()
            processed = []
            for rv in random_vars:
                try:
                    processed.append(int(rv.decode()))
                except (AttributeError, ValueError):
                    processed.append(rv.decode)
            random_vars = processed
            pdftype = f["gpc_object/pdftype"][()].tolist()
            pdftype = [s.decode() for s in pdftype]
            pdfshape = f['gpc_object/pdfshape'][()].tolist()
            limits = f['gpc_object/limits'][()].tolist()
            pdf_paras = [ps if pt == 'norm' else np.concatenate((ps, lm)) for ps, lm, pt in
                         zip(pdfshape, limits, pdftype)]
            poly_idx = f['gpc_object/poly_idx'][()]
            coords_norm = f['gpc_object/grid/coords_norm'][()]
            regularization_factors = f['gpc_object/regularization_factors'][()]
            model = pygpc.testfunctions.Dummy()
            parameters = prep_parameters(random_vars=random_vars, pdf_types=pdftype, pdf_paras=pdf_paras)
            problem = pygpc.Problem(model, parameters)

        return cls(problem, regularization_factors, poly_idx, coords_norm,
                   sim_type,data_file=fn_hdf5, n_cpu=cpus)

    def visualize(self):
        """Creates a mesh file for visualization

        Returns:
        --------
        writes a mesh file in the same folder as the hdf5 file
        msh: simnibs.msh.mesh_io
            mesh with gpc fields
        """
        msh = mesh_io.Msh.read_hdf5(self.data_file, path="mesh_roi/")
        mesh_io.write_msh(msh, self.mesh_file)
        return msh

    def expand_quantity(self, func=None, field="E"):
        """Expand an arbitrary quantity

        Parameters
        --------......
        func (optional): function
            Function which takes up a single argument and returns a single number or a
            vector to be expanded by gpc.
            The arguments corresponds the the field samples in the format
            [N_simulations x N_roi x 3] for vector fields
            [N_simulations x N_roi] for scalar fields

        field (Optinal): 'v', 'e', 'E', 'J' or 'j'
            field to be passed as an argument to the function, must have been previously
            calculated suing the postprocessing method. Default: E

        Returns
        --------
        coeffs: ndarray
            gPC polynomial coefficients. You can then call the attributes mean, std,
            sobol, and globalsens to calculate the mean, standard deviation, sobol coefficients
            and sensitivity of you quantity of interest
        """
        try:
            field = read_data_hdf5(
                FIELD_NAME[field] + "_samples",
                self.data_file,
                "mesh_roi/data_matrices/",
            )
        except KeyError:
            field = read_data_hdf5(
                field + "_samples", self.data_file, "mesh_roi/data_matrices/"
            )
        except RuntimeError:
            raise IOError(f'Could not read field "{field}" in file "{self.data_file}"')
        if func:
            f = func(field)
        else:
            f = field.reshape(field.shape[0], -1)
        coeffs, cv = self.expand(f)
        logger.info("CV value: {0:1e}".format(cv))
        return coeffs

    def roi_mesh(self):
        """Returns the mesh where the expansion is defined"""
        return mesh_io.Msh.read_hdf5(self.data_file, "mesh_roi/")

    def expand(self, res):
        """ Compute the gPC coefficients using the Tikhonov regularization regression method.
        Parameters:
        -------------
        res: ndarray [n_grid x n_out]
            Results from simulations with N_out output quantities
        Returns:
        -------------
        coeffs: ndarray of float [n_coeffs x n_out]
            gPC coefficients
        eps: float
            The 10-fold cross-validation error for the gPC model
        """
        if res.ndim == 1:
            res = res[:, None]
        # regularization_factors = self.options["settings"].get('alpha')
        regularization_factors = self.regularization_factors
        coeffs = None
        min_error = float('inf')

        for reg_factor in regularization_factors:
            # determine gpc coefficients
            coeffs_temp = self.solve(results=res,
                                     solver='Tikhonov',
                                     settings={'alpha': reg_factor},
                                     verbose=False)
            # validate gPC approximation
            errors_temp = self.validate(coeffs=coeffs_temp,
                                        results=res,
                                        settings={'alpha': reg_factor},
                                        verbose=False)
            if errors_temp < min_error:
                min_error = errors_temp
                eps = errors_temp
                coeffs = coeffs_temp
                selected_reg = reg_factor
        return coeffs, eps

def run_tms_gpc(poslist, fn_simu, cpus=1, tissues=[ElementTags.GM], eps=1e-2,
                max_iter=1000, min_iter=2, data_poly_ratio=2, regularization_factors=[0]):
    '''Run one TMS gPC for each position in the current TMSLIST

    Parameters
    --------------
    poslist: simnibs.sim_struct.TMSLIST
        TMSLIST structure defining the simulation
    fn_simu: str
        Output name
    cpus: int (optional)
        Number of CPUs to use
    tissues: list (optional)
        List of tissue tags where to evalute the electric field. Default: [2]
    eps: float (optional)
        Tolerance for gPC expansions. Default:1e-2
    max_iter: int (optinal)
        Maximum number of adaptive gPC expansion. Defaut:1000
    min_iter: int (optinal)
        Minimum number of adaptive gPC expansion interations. Defaut:2
    data_poly_ratio: int
        Ratio of number of new simulation per new polynomial. Default:2

    Returns
    --------
    fns: list
        List of mesh file names
    '''
    poslist._prepare()
    fn_simu = os.path.abspath((os.path.expanduser(fn_simu)))

    logger.info('Running a gPC expansion with tolerance: {0:1e}'.format(eps))
    path, basename = os.path.split(fn_simu)
    if not os.path.isdir(path) and path != "":
        os.mkdir(path)
    path, basename = os.path.split(fn_simu)
    if not os.path.isdir(path) and path != '':
        os.mkdir(path)
    parameters, random_vars = prep_gpc(poslist)

    fns = []
    for i, p in enumerate(poslist.pos):
        fn_hdf5 = fn_simu + "_{0:0=4d}_gpc.hdf5".format(i + 1)
        if os.path.isfile(fn_hdf5):
            raise IOError("Output file " + fn_hdf5 + " already exists")
        matsimnibs = p.calc_matsimnibs(poslist.mesh)
        sampler = TMSgPCSampler(
            poslist.mesh,
            poslist,
            fn_hdf5,
            poslist.fnamecoil,
            matsimnibs,
            p.didt,
            roi=tissues,
        )
        sampler.create_hdf5()
        # construct gPC model
        algorithm = setup_gpc_algorithm(sampler=sampler,
                                        parameters=parameters,
                                        data_poly_ratio=data_poly_ratio,
                                        max_iter=max_iter,
                                        eps=eps,
                                        n_cpus=cpus,
                                        min_iter=min_iter,
                                        regularization_factors=np.array(regularization_factors))
        gpc_session, _, _ = algorithm.run()
        gpc_reg = gPC_regression(problem=gpc_session.problem, regularization_factors=regularization_factors,
                                 multi_indices=gpc_session.basis.multi_indices,
                                 coords_norm=gpc_session.grid.coords_norm,
                                 sim_type='TMS', data_file=fn_hdf5, n_cpu=cpus)
        gpc_reg.save_hdf5()
        gpc_reg.postprocessing(poslist.postprocess)
        gpc_reg.visualize()
        print_gpc_summary(gpc_session)
        fns.append(gpc_reg.mesh_file)

    return fns

def run_tcs_gpc(poslist, fn_simu, cpus=1, tissues=[2], eps=1e-2,
                max_iter=1000, min_iter=2, data_poly_ratio=2,
                regularization_factors=[1e-5]):
    ''' Runs a tDCS gPC expansion

    Parameters
    -------------
    poslist: simnibs.sim_struct.TDCSLIST
        TDCSLIST structure defining the simulaton
    fn_simuL str
        Output name
    cpus: int (optional)
        Number of CPUs to use
    tissues: list (optional)
        List of tissue tags where to evaluate the electric field. Default: [2]
    eps: float (optional)
        Tolerance fro gPC expansions. Default:1e-2
    max_iter: int (optinal)
        Maximum number of adaptive gPC expansion iterations. Defaut:1000
    min_iter: int (optinal)
        Minimum number of adaptive gPC expansion interations. Defaut:2
    data_poly_ratio(optional): int
        Ratio of number of new simulation per new polynomial. Default:2

    Returns
    --------
    fns: list
        List of mesh file names
    '''

    poslist._prepare()
    fn_simu = os.path.abspath(os.path.expanduser(fn_simu))
    logger.info("Running a gPC expansion with tolerance: {0:1e}".format(eps))
    fn_hdf5 = fn_simu + "_gpc.hdf5"
    if os.path.isfile(fn_hdf5):
        raise IOError('Output file ' + fn_hdf5 + ' already exists')
    parameters, random_vars = prep_gpc(poslist)
    path, basename = os.path.split(fn_simu)
    if not os.path.isdir(path) and path != "":
        os.mkdir(path)
    # place electrodes
    fn_no_extension, extension = os.path.splitext(fn_simu)
    m, electrode_surfaces = poslist._place_electrodes()
    mesh_io.write_msh(m, fn_simu + "_electrodes.msh")
    sampler = TDCSgPCSampler(
        m,
        poslist,
        fn_simu + "_gpc.hdf5",
        electrode_surfaces,
        poslist.currents,
        roi=tissues,
    )
    sampler.create_hdf5()
    # construct gPC model
    algorithm = setup_gpc_algorithm(sampler=sampler,
                                    parameters=parameters,
                                    data_poly_ratio=data_poly_ratio,
                                    max_iter=max_iter,
                                    eps=eps,
                                    n_cpus=cpus,
                                    min_iter=min_iter,
                                    regularization_factors=np.array(regularization_factors)) 
    gpc_session, _ , _ = algorithm.run()
    gpc_reg = gPC_regression(problem=gpc_session.problem, regularization_factors=regularization_factors,
                   multi_indices=gpc_session.basis.multi_indices,coords_norm=gpc_session.grid.coords_norm,
                   sim_type='TCS', data_file=fn_hdf5, n_cpu=cpus)
    # postprocessing
    gpc_reg.save_hdf5()
    gpc_reg.postprocessing(poslist.postprocess)
    gpc_reg.visualize()
    # gpc summary
    print_gpc_summary(gpc_session)

    return [gpc_reg.mesh_file]


class gPCSampler(object):
    """Object used by pygpc to sample

    Attributes
    -----------
    mesh: simnibs.msh.mesh_io.Msh
        Mesh structure
    roi: list of integers
        List of tags defining the ROI
    mesh_roi: simnibs.msh.mesh_io.Msh
        Mesh of the ROI only
    fn_hdf5: string
        Name of hdf5 file with simulation output
    poslist: simnibs.simulation.sim_struct.SimuList
        Structure where the conductivity is defined
    identifiers: list
        List of random variable identifiers
    qoi_function: OrderedDict
        dictionaty with functions for each QOI.
        The first QOI will be passed to the gPC algorithm.


    Parameters
    ----------
    mesh: simnibs.msh.mesh_io.Msh
        Mesh structure
    poslist: simnibs.simulation.sim_struct.SimuList
        Structure where the conductivity is defined
    fn_hdf5: string
        Name of hdf5 file with simulation output
    roi: list of integers (Optional)
        List of tags defining the ROI. Default: [2]
    """

    def __init__(self, m, poslist, fn_hdf5, roi=[2]):
        self.mesh = m
        self.roi = roi
        if self.roi is not None:
            self.mesh_roi = m.crop_mesh(roi)
        else:
            self.mesh_roi = m
            self.roi = np.unique(m.elm.tag1).tolist()
        self.fn_hdf5 = fn_hdf5
        self.poslist = poslist
        self.parameters, self.identifiers =  prep_gpc(poslist)
        self.qoi_function = OrderedDict([('E', self._calc_E)])

    def create_hdf5(self):
        """Creates an HDF5 file to store the data"""
        # if the hdf5 file does not exist, create it
        file_exists = os.path.exists(self.fn_hdf5)
        if file_exists:
            raise IOError(
                "Cannot create hdf5 file: {0} " "it already exists!".format(
                    self.fn_hdf5
                )
            )

        self.mesh.write_hdf5(self.fn_hdf5, "mesh/")
        self.mesh_roi.write_hdf5(self.fn_hdf5, "mesh_roi/")
        self.poslist._write_conductivity_to_hdf5(self.fn_hdf5)
        with h5py.File(self.fn_hdf5, "a") as f:
            f.create_dataset("roi", data=np.atleast_1d(np.array(self.roi, dtype=int)))

    @classmethod
    def load_hdf5(cls, fn_hdf5):
        """Loads structure from hdf5 file"""
        mesh = mesh_io.Msh.read_hdf5(fn_hdf5, "mesh/")
        poslist = SimuList()
        poslist._get_conductivity_from_hdf5(fn_hdf5)
        with h5py.File(fn_hdf5, "r") as f:
            roi = f["roi"][()].tolist()
        return cls(mesh, poslist, fn_hdf5, roi)

    def record_data_matrix(self, data, name, group):
        ''' Appends or create data to the HDF5 file

        Parameters:
        -------------
        data: np.ndarray
            Data to be appended. Will be appended along the first dimension
        name: str
            Name of data seet
        group: str
            Group where to place data set
        '''
        data = np.array(data).squeeze()
        data = np.atleast_1d(data)
        with h5py.File(self.fn_hdf5, "a") as f:
            try:
                g = f.create_group(group)
            except:
                g = f[group]
            if name not in g.keys():
                g.create_dataset(
                    name,
                    shape=(0,) + data.shape,
                    maxshape=(None,) + data.shape,
                    dtype=data.dtype,
                    chunks=(1,) + data.shape,
                )

            dset = g[name]
            dset.resize((dset.shape[0] + 1,) + data.shape)
            dset[-1, ...] = data

    def run_simulation(self, random_vars):
        raise NotImplementedError("This method is to be implemented in a subclass!")

    def _calc_E(self, v, random_vars, dAdt=None):
        grad = v.gradient()
        grad.assign_triangle_values()
        E = -grad.value * 1e3
        if dAdt is not None:
            E -= dAdt.value
        return E

    def _update_poslist(self, parameters):
        poslist = copy.deepcopy(self.poslist)
        for i, iden in enumerate(self.identifiers):
            if type(iden) == int:
                poslist.cond[iden-1].value = parameters[f"cond_{iden}"]
        return poslist

    def run_N_random_simulations(self, N):
        grid = pygpc.Random(parameters_random=self.parameters, n_grid=N)
        for i, x in enumerate(grid.coords):
            logger.info("Running simulation {0} out of {1}".format(i + 1, N))
            self.run_simulation(x)

class TDCSgPCSampler(gPCSampler):
    """Object used by pygpc to sample a tDCS problem

    Attributes
    -----------
    mesh: simnibs.msh.mesh_io.Msh
        Mesh structure
    roi: list of integers
        List of tags defining the ROI
    mesh_roi: simnibs.msh.mesh_io.Msh
        Mesh of the ROI only
    fn_hdf5: string
        Name of hdf5 file with simulation output
    poslist: simnibs.simulation.sim_struct.SimuList
        Structure where the conductivity is defined
    gpc_vars: list
        List with variables for inputing to pygpc
    identifiers: list
        List of random variable identifiers
    qoi_function: OrderedDict
        dictionaty with functions for each QOI.
        The first QOI will be passed to the gPC algorithm, flattened.
        The QOI function should take only one argument (the potential v)
    el_tags: list of integers
        List of integers with surface tags of electrodes
    el_currents: list of floats
        List with current values for each electrode (in A)

    Parameters
    ----------
    mesh: simnibs.msh.mesh_io.Msh
        Mesh structure
    poslist: simnibs.simulation.sim_struct.SimuList
        Structure where the conductivity is defined
    fn_hdf5: string
        Name of hdf5 file with simulation output
    el_tags: list of integers
        List of integers with surface tags of electrodes
    el_currents: list of floats
        List with current values for each electrode (in A)
    roi: list of integers (Optional)
        List of tags defining the ROI. Default: [2]

    """

    def __init__(self, mesh, poslist, fn_hdf5, el_tags, el_currents, roi=[2]):
        super(TDCSgPCSampler, self).__init__(mesh, poslist, fn_hdf5, roi=roi)
        self.el_tags = el_tags
        self.el_currents = el_currents

    def create_hdf5(self):
        super(TDCSgPCSampler, self).create_hdf5()
        with h5py.File(self.fn_hdf5, "a") as f:
            f.create_dataset("el_tags", data=np.array(self.el_tags, dtype=int))
            f.create_dataset(
                "el_currents", data=np.array(self.el_currents, dtype=float)
            )

    @classmethod
    def load_hdf5(cls, fn_hdf5):
        s = gPCSampler.load_hdf5(fn_hdf5)
        with h5py.File(fn_hdf5, "r") as f:
            el_tags = f["el_tags"][()].tolist()
            el_currents = f["el_currents"][()].tolist()
        return cls(s.mesh, s.poslist, s.fn_hdf5, el_tags, el_currents, roi=s.roi)

    def run_simulation(self, parameters):
        poslist = self._update_poslist(parameters)
        random_vars = parameters
        cond = poslist.cond2elmdata(self.mesh)
        v = fem.tdcs(self.mesh, cond, self.el_currents, self.el_tags, units="mm")

        self.mesh.nodedata = [v]
        cropped = self.mesh.crop_mesh(self.roi)
        v_c = cropped.nodedata[0]
        self.mesh.nodedata = []

        qois = []
        for qoi_name, qoi_f in self.qoi_function.items():
            qois.append(qoi_f(v_c, random_vars))

        random_vars = np.array(list(parameters.values()))
        self.record_data_matrix(random_vars, 'random_var_samples', '/')
        self.record_data_matrix(v.value, 'v_samples', 'mesh/data_matrices')
        self.record_data_matrix(v_c.value, 'v_samples', 'mesh_roi/data_matrices')
        for qoi_name, qoi_v in zip(self.qoi_function.keys(), qois):
            self.record_data_matrix(
                qoi_v, qoi_name + "_samples", "mesh_roi/data_matrices"
            )

        del cropped
        del cond
        del v
        del v_c

        return np.atleast_1d(qois[0]).reshape(-1)


class TMSgPCSampler(gPCSampler):
    """Object used by pygpc to sample a TMS problem

    Attributes
    -----------
    mesh: simnibs.msh.mesh_io.Msh
        Mesh structure
    roi: list of integers
        List of tags defining the ROI
    mesh_roi: simnibs.msh.mesh_io.Msh
        Mesh of the ROI only
    fn_hdf5: string
        Name of hdf5 file with simulation output
    poslist: simnibs.simulation.sim_struct.SimuList
        Structure where the conductivity is defined
    gpc_vars: list
        List with variables for inputing to pygpc
    identifiers: list
        List of random variable identifiers
    qoi_function: OrderedDict
        dictionaty with functions for each QOI.
        The first QOI will be passed to the gPC algorithm.
        The QOI function should take only 2 arguments: (the potential v and dAdt)
    matsimnibs: np.ndarray
        Matrix defining coil position
    dIdt: float
        Current intensity in coil

    Parameters
    ----------
    mesh: simnibs.msh.mesh_io.Msh
        Mesh structure
    poslist: simnibs.simulation.sim_struct.SimuList
        Structure where the conductivity is defined
    fn_hdf5: string
        Name of hdf5 file with simulation output
    pos: simnibs.simulation.sim_struct.POS
       Coil position definition
    roi: list of integers (Optional)
        List of tags defining the ROI. Default: [2]

    """

    def __init__(self, mesh, poslist, fn_hdf5, fnamecoil, matsimnibs, didt, roi=[2]):
        super(TMSgPCSampler, self).__init__(mesh, poslist, fn_hdf5, roi=roi)
        self.matsimnibs = np.ascontiguousarray(matsimnibs)
        self.didt = didt
        self.fnamecoil = fnamecoil
        self.constant_dAdt = True

    def create_hdf5(self):
        super(TMSgPCSampler, self).create_hdf5()
        with h5py.File(self.fn_hdf5, "a") as f:
            f.create_dataset("matsimnibs", data=self.matsimnibs)
            f.create_dataset("didt", data=np.array(self.didt, dtype=float))
            f.create_dataset(
                "fnamecoil", data=np.array(self.fnamecoil, dtype=np.bytes_)
            )

    @classmethod
    def load_hdf5(cls, fn_hdf5):
        s = gPCSampler.load_hdf5(fn_hdf5)
        with h5py.File(fn_hdf5, "r") as f:
            matsimnibs = f["matsimnibs"][()]
            fnamecoil = f["fnamecoil"][()].decode()
            didt = f["didt"][()]

        return cls(s.mesh, s.poslist, s.fn_hdf5, fnamecoil, matsimnibs, didt, roi=s.roi)

    def run_simulation(self, parameters):
        poslist = self._update_poslist(parameters)
        cond = poslist.cond2elmdata(self.mesh)
        if self.constant_dAdt:
            try:
                dAdt = self.dAdt
                dAdt_roi = self.dAdt_roi
            except AttributeError:
                tms_coil = TmsCoil.from_file(self.fnamecoil)
                didt = np.atleast_1d(self.didt)
                if len(didt) == 1:
                    for (
                        stimulator
                    ) in tms_coil.get_elements_grouped_by_stimulators().keys():
                        stimulator.di_dt = didt
                else:
                    for stimulator, stimulator_didt in zip(tms_coil.get_elements_grouped_by_stimulators().keys(), didt):
                        stimulator.di_dt = stimulator_didt
                dAdt = tms_coil.get_da_dt(self.mesh, self.matsimnibs)
                if isinstance(dAdt, mesh_io.NodeData):
                    dAdt = dAdt.node_data2elm_data()
                dAdt.field_name = "dAdt"
                dAdt.write_hdf5(self.fn_hdf5, "mesh/elmdata/")
                self.dAdt = dAdt
                self.mesh.elmdata = [dAdt]
                cropped = self.mesh.crop_mesh(self.roi)
                dAdt_roi = cropped.elmdata[0]
                dAdt_roi.write_hdf5(self.fn_hdf5, "mesh_roi/elmdata/")
                self.mesh.elmdata = []
                self.dAdt_roi = dAdt_roi
        else:
            raise NotImplementedError

        v = fem.tms_dadt(self.mesh, cond, dAdt)
        self.mesh.nodedata = [v]
        cropped = self.mesh.crop_mesh(self.roi)
        v_c = cropped.nodedata[0]
        self.mesh.nodedata = []

        qois = []
        for qoi_name, qoi_f in self.qoi_function.items():
            qois.append(qoi_f(v_c, dAdt_roi))

        random_vars = np.array(list(parameters.values()))
        self.record_data_matrix(random_vars, 'random_var_samples', '/')
        self.record_data_matrix(v.value, 'v_samples', 'mesh/data_matrices')
        self.record_data_matrix(v_c.value, 'v_samples',
                                'mesh_roi/data_matrices')
        for qoi_name, qoi_v in zip(self.qoi_function.keys(), qois):
            self.record_data_matrix(
                qoi_v, qoi_name + "_samples", "mesh_roi/data_matrices"
            )

        del cropped
        del cond
        del v

        return np.atleast_1d(qois[0]).reshape(-1)


class NIBS_Model(AbstractModel):
    def __init__(self, fname_matlab=None, matlab_model=False):
        super(type(self), self).__init__(matlab_model=matlab_model)
        self.fname = inspect.getfile(inspect.currentframe())

    def validate(self):
        pass

    def simulate(self, process_id=None, matlab_engine=None):

        para_temp = OrderedDict()
        n_grid = len(next(iter(self.p.values())))
        qoi_list = []

        for i in range(n_grid):
            for key, value in self.p.items():
                para_temp[key] = value[i]

            current_qoi = self.sampler.run_simulation(para_temp)
            qoi_list.append(current_qoi)

        qoi = np.array(qoi_list)

        return qoi


def setup_gpc_algorithm(sampler,parameters,data_poly_ratio=2, max_iter=1000, eps= 1E-2,
                        regularization_factors=np.logspace(-5, 3, 9),n_cpus=1, min_iter=2, order_end=20, interaction_order=3, error_type="kcv"):
    """ Setup the algorithm to build up a gPC model for the sampler. """
    # Convert the sampler to a pygpc model.


    model = NIBS_Model()
    model.sampler = sampler
    # define the gPC problem
    problem = pygpc.Problem(model, parameters)
    # define the algorithm options
    options = dict()
    options["order_start"] = 0
    options["order_end"] = order_end
    options["solver"] = "Tikhonov"
    options["settings"] = {"alpha": regularization_factors}
    options["interaction_order"] = interaction_order
    options["n_cpu"] = n_cpus
    options["fn_results"] = None
    options["matrix_ratio"] = data_poly_ratio
    options["grid"] = pygpc.Random
    options["grid_options"] = {"seed": 1}
    options["error_type"] = error_type
    options["eps"] = eps
    options["max_iter"] = max_iter
    options["n_cpu_comp"] = n_cpus
    if n_cpus == 1:
        options["n_cpu_basis"] = 1
    else:
        options["n_cpu_basis"] = 0
    options["min_iter"] = min_iter
    options["print_function"] = logger.info
    algorithm = pygpc.RegAdaptiveOldSet(problem=problem, options=options)
    return algorithm

def prep_gpc(simlist):
    "Extract random parameters from the simlist to prepare for pygpc"
    cond = simlist.cond
    random_vars = []
    pdf_types = []
    pdf_paras = []

    for i, c in enumerate(cond):
        if c.distribution_type is not None:
            random_vars.append(i + 1)
            pdf_types.append(c.distribution_type)
            pdf_paras.append(c.distribution_parameters)

    if len(random_vars) == 0:
        raise ValueError('No random variables found for simulation')

    parameters = prep_parameters(random_vars, pdf_types, pdf_paras)

    return parameters, random_vars

def prep_parameters(random_vars, pdf_types, pdf_paras):
    '''Define random parameters for pygpc based on variables indices, distribution types, and distribution parameters'''
    parameters = OrderedDict()

    for rv, pdf, pars in zip(random_vars, pdf_types, pdf_paras):
        if pdf == 'uniform':
            if len(pars) != 2:
                raise ValueError('uniform random variables must have 2 parameters')
            parameters[f"cond_{rv}"] = pygpc.Beta(pdf_shape=[1, 1],
                                                  pdf_limits=[pars[0], pars[1]])
        elif pdf == 'beta':
            if len(pars) != 4:
                raise ValueError('Beta random variables must have 4 parameters')
            parameters[f"cond_{rv}"] = pygpc.Beta(pdf_shape=[pars[0], pars[1]],
                                                  pdf_limits=[pars[2], pars[3]])
        elif pdf == 'normal' or 'norm':
            if len(pars) != 2:
                raise ValueError('Normal random variables must have 2 parameters')
            parameters[f"cond_{rv}"] = pygpc.Norm(pdf_shape=[pars[0], pars[1]])
        else:
            raise ValueError('Invalid distribution_type: {0}'.format(pdf))

    return parameters

def print_gpc_summary(gpc_session):
    logger.info("gPC information summary:")
    logger.info("========================")
    logger.info(f"Number of Simulations: {gpc_session.grid.coords.shape[0]}")
    logger.info(f"Number of Polynomials: {gpc_session.basis.multi_indices.shape[0]}")
    if gpc_session.options["error_type"] == 'kcv':
        logger.info("Validation Method: K-fold Cross-Validation")
    elif gpc_session.options["error_type"] == 'kcv':
        logger.info("Validation Method: Leave-one-out Cross-Validation")
    logger.info(f"Final error: {min(gpc_session.error)}")
