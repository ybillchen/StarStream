# Licensed under BSD-3-Clause License - see LICENSE

from functools import partial

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import astropy.coordinates as coord
import astropy.units as u

from . import spray as spray
from . import utils as utils

__all__ = [
    "KernelPDF", 
    "GaiaStreamPDF", 
    "GaiaStreamFromProgPDF",
    "GaiaBackgroundPDF"
]

def gaussian(u):
    return np.exp(-0.5*u**2)

def kde(Xdata, Xeval, h, proj_axes=[], error=None, kernel=gaussian):
    """
    A simple, fast multi-dimensional KDE function
    """
    assert kernel == gaussian # currently only support gaussian
    non_proj_axes = np.ones(Xdata.shape[1], dtype=bool)
    non_proj_axes[proj_axes] = False
    if error is None:
        heff = h[:,np.newaxis,non_proj_axes]
    else:
        heff = np.sqrt(
            h[:,np.newaxis,non_proj_axes]**2 + \
            error[np.newaxis,:,non_proj_axes]**2
        )
    dist = np.sqrt(np.sum(
        (
            (
                Xeval[np.newaxis,:,non_proj_axes] - \
                Xdata[:,np.newaxis,non_proj_axes]
            ) / heff
        )**2, 
        axis=2
    ))
    k = np.sum(
        kernel(dist)/np.prod(2.50663*heff,axis=2), axis=0
    )
    return k / len(Xdata)

class KernelPDF(object):
    """
    Base class for KernelPDF
    """
    def __init__(self, data, grids, hs, groups):
        """
        data: array-like (N, M): N data points with M dimensions
        grids: list (M): arrays of evaluation grids
        hs: array-like (N, M): bandwidths for Gaussian KDE
        groups: list: 
            1) None as member means direct KDE in this dimension
            2) array as member means interpolation in this dimension
        """
        self.N, self.dimension = data.shape
        assert self.dimension == len(grids)
        self.data = data
        self.grids = grids
        self.hs = hs
        self.groups = groups

        self.get_pdf()

    def get_pdf(self):
        self.pdfs = []
        for group in self.groups:
            group_data = self.data[:,group]
            group_h = self.hs[:,group]
            group_grid = [self.grids[i] for i in group]
            if group_grid[0] is None:
                # direct KDE
                self.pdfs.append(partial(kde, Xdata=group_data, h=group_h))
            else:
                # interpolation
                group_mesh = np.meshgrid(*group_grid, indexing="ij")
                group_mesh_flatten = np.column_stack([
                    m.flatten() for m in group_mesh
                ])
                
                group_prob = kde(group_data, group_mesh_flatten, group_h)
                group_prob = group_prob.reshape(group_mesh[0].shape)
                self.pdfs.append(RegularGridInterpolator(
                    group_grid,
                    group_prob,
                    bounds_error=False,
                    fill_value=0
                ))

    def eval_pdf(self, data_eval, err_eval=None):
        """
        data_eval: array-like (N, M): N data points with M dimensions
        err_eval: array-like (N, M): N data points with M dimensions
            err_eval not used in direct KDE mode
        """
        prob = np.ones(len(data_eval))
        for group, pdf in zip(self.groups, self.pdfs):
            group_grid = [self.grids[i] for i in group]
            if group_grid[0] is None:
                if err_eval is None:
                    err = np.zeros_like(data_eval[:,group])
                else:
                    err = err_eval[:,group]
                prob *= pdf(Xeval=data_eval[:,group], error=err)
            else:
                prob *= pdf(data_eval[:,group])
        return prob

class GaiaStreamPDF(KernelPDF):
    """
    Stream PDF for Gaia, as in Chen+25
    """
    def __init__(self, data, f_h=0.1):
        """
        data: array-like (N, 6): N data points of
            (phi1, phi2, muphi1, muphi2, color, magnitude)
        f_h: KDE bandwidth ratio, default: 0.1
        """

        grids = [None]*6

        hs = np.c_[
            np.full(len(data), fill_value=f_h*np.std(data[:,0])),
            np.full(len(data), fill_value=f_h*np.std(data[:,1])),
            np.full(len(data), fill_value=f_h*np.std(data[:,2])),
            np.full(len(data), fill_value=f_h*np.std(data[:,3])),
            np.full(len(data), fill_value=0.02),
            np.full(len(data), fill_value=0.1)
        ]

        groups = [[0, 1, 2, 3, 4, 5]]

        super().__init__(data, grids=grids, hs=hs, groups=groups)

class GaiaStreamFromProgPDF(GaiaStreamPDF):
    """
    Stream PDF from prognitor for Gaia, as in Chen+25
    """
    def __init__(self, pot, mass_gc, coord_gc, 
        iso_mini, iso_color, iso_mag, mag_cut=20.0, frame=None,
        t_tot=1.0, n_steps=2000, pot_gc=None, 
        nhalf_release=None, f_h=0.1, seed=0):
        """
        pot: Galactic potential, agama.Potential
        mass_gc: GC mass in Msun
        coord_gc: GC coordinate, astropy.coord.Galactocentric
        iso_mini: initial mass column of isochrone
        iso_color: color column of isochrone
        iso_mag: magnitude column of isochrone
        mag_cut: magnitude faint-end cut, default: 20.0
        frame: astropy.coord.SkyCoord, default: None -> phi1phi2
        t_tot: total integration time, default: 1.0 (Gyr)
        n_steps: number of tracer releases, default: 2000
        pot_gc: GC potential, agama.Potential, 
            default: None -> Plummer with Rs = 4 pc
        nhalf_release: half of tracer release rate, array (n_steps), 
            default: None -> uniform
        f_h: KDE bandwidth ratio, default: 0.1
        seed: random seed, default: 0
        """

        if frame is None:
            skycoord_gc = coord_gc.transform_to(coord.ICRS())
            frame = utils.calc_prog_frame(skycoord_gc)

        # generate simulated stream

        coord_tracer, t_tracer = spray.spray(
            pot=pot, mass_gc=mass_gc, coord_gc=coord_gc, t_tot=t_tot, 
            n_steps=n_steps, pot_gc=pot_gc, 
            nhalf_release=nhalf_release, seed=seed)

        phi1phi2_tracer = coord_tracer.transform_to(frame())

        # Assign color and mag to stream tracers

        mag_tracer, color_tracer = utils.assign_color_mag(
            phi1phi2_tracer.distance.to_value("kpc"), 
            mag_cut, iso_mini, iso_color, iso_mag, seed=seed
        )

        mask = mag_tracer < mag_cut
        mag_tracer = mag_tracer[mask]
        color_tracer = color_tracer[mask]
        phi1phi2_tracer = phi1phi2_tracer[mask]

        # gather data into arrays

        data_tracer = np.c_[
            phi1phi2_tracer.phi1.wrap_at(180*u.deg).to_value("deg"),
            phi1phi2_tracer.phi2.to_value("deg"),
            phi1phi2_tracer.pm_phi1_cosphi2.to_value("mas/yr"),
            phi1phi2_tracer.pm_phi2.to_value("mas/yr"),
            color_tracer,
            mag_tracer
        ]

        super().__init__(data_tracer, f_h=f_h)

class GaiaBackgroundPDF(KernelPDF):
    """
    Background PDF for Gaia, as in Chen+25
    """
    def __init__(self, data, field_range=(-10.0,10.0), mag_cut=20.0, 
        dx=0.5, dmu=1.0, dcolor=0.1, dmag=0.1,
        n_subsample=10000, seed=0):
        """
        data: array-like (N, 6): N data points of
            (phi1, phi2, muphi1, muphi2, color, magnitude)
        field_range: array-like (2): lower/upper bounds of field,
            default: (-10,0, 10.0) (deg)
        mag_cut: magnitude faint-end cut, default: 20.0
        dx: KDE bandwidth for spatial scales, default: 0.5 (deg)
        dmu: KDE bandwidth for proper motions, default: 1.0 (mas/yr)
        dcolor: KDE bandwidth for color, default: 0.1
        dmag: KDE bandwidth for magnitude, default: 0.1
        n_subsample: number of subsamples
        seed: random seed
        """

        dx = 0.5 # in deg
        dmu = 1.0 # in mas/yr
        dcolor = 0.1
        dmag = 0.1
        grids = [
            np.arange(field_range[0],field_range[1]+0.1,dx), 
            np.arange(field_range[0],field_range[1]+0.1,dx), 
            np.arange(-100.0,100.0+0.1,dmu),
            np.arange(-100.0,100.0+0.1,dmu), 
            np.arange(-6.0,10.0+0.01,dcolor),
            np.arange(0.0,mag_cut+0.01,dmag)
        ]

        # select subsample
        if n_subsample < len(data):
            rng = np.random.default_rng(seed)
            subsample = rng.choice(
                a=len(data), size=n_subsample, replace=False
            )
        else:
            n_subsample = len(data)
            subsample = np.arange(n_subsample, dtype=int)

        hs = np.c_[
            np.full(n_subsample, fill_value=dx),
            np.full(n_subsample, fill_value=dx),
            np.full(n_subsample, fill_value=dmu),
            np.full(n_subsample, fill_value=dmu),
            np.full(n_subsample, fill_value=dcolor),
            np.full(n_subsample, fill_value=dmag)
        ]

        groups = [[0,1], [2], [3], [4,5]]

        super().__init__(
            data[subsample], grids=grids, hs=hs, groups=groups
        )
