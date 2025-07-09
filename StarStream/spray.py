# Licensed under BSD-3-Clause License - see LICENSE

import numpy as np
import astropy.units as u
import astropy.coordinates as coord

import agama

from . import agama_stream as ags

__all__ = [
    "spray"
]

def spray(pot, mass_gc, coord_gc, t_tot=1.0, n_steps=2000, pot_gc=None, nhalf_release=None, seed=0):
    """
    pot: Galactic potential, agama.Potential
    mass_gc: GC mass in Msun
    coord_gc: GC coordinate, astropy.coord.Galactocentric
    t_tot: total integration time, default: 1.0 (Gyr)
    n_steps: number of tracer releases, default: 2000
    pot_gc: GC potential, agama.Potential, default: None -> Plummer with Rs = 4 pc
    nhalf_release: half of tracer release rate, array (n_steps), default: None -> uniform
    seed: random seed, default: 0
    """

    t_tot_agama = t_tot / 0.978 # in agama units
    
    if pot_gc is None:
        pot_gc = agama.Potential(
            type="Plummer", mass=mass_gc, scaleRadius=4e-3
        )
    xv_gc = np.r_[
        coord_gc.x.to_value("kpc"),
        coord_gc.y.to_value("kpc"),
        coord_gc.z.to_value("kpc"),
        coord_gc.v_x.to_value("km/s"),
        coord_gc.v_y.to_value("km/s"),
        coord_gc.v_z.to_value("km/s")
    ]
    time_sat, orbit_sat = ags.integrate_prog(
        -t_tot_agama, n_steps, pot, xv_gc)
    
    if nhalf_release is None:
        # default: release particles uniformly in time
        nhalf_release = np.ones(n_steps, dtype=int)
    assert len(nhalf_release) == n_steps
    
    time_sat, orbit_sat, xv_stream, ic_stream, time_stream = \
        ags.create_stream(
            ags.create_ic_chen25, np.random.default_rng(seed), 
            -t_tot_agama, 2*n_steps, pot, xv_gc, mass_gc, 
            pot_sat=pot_gc, nhalf_release=nhalf_release
        )

    mask_valid = ~np.isnan(ic_stream[:,0])
    ic_stream, time_stream_agama = \
        ic_stream[mask_valid], time_stream[mask_valid]
    
    stream_coord = coord.Galactocentric(
        x=xv_stream[:,0]*u.kpc, 
        y=xv_stream[:,1]*u.kpc, 
        z=xv_stream[:,2]*u.kpc, 
        v_x=xv_stream[:,3]*u.km/u.s, 
        v_y=xv_stream[:,4]*u.km/u.s, 
        v_z=xv_stream[:,5]*u.km/u.s
    )

    time_stream = time_stream_agama * 0.978 # in Gyr

    return stream_coord, time_stream
