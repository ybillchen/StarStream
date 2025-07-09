import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.coordinates as coord
import agama

from StarStream import (
    GaiaStreamFromProgPDF, GaiaBackgroundPDF,
    calc_prog_frame, optimize_mle
)

test_params = [
    (
        True,
        coord.Galactocentric(
            x=0.0*u.kpc,
            y=10.0*u.kpc,
            z=10.0*u.kpc,
            v_x=-100.0*u.km/u.s,
            v_y=0.0*u.km/u.s,
            v_z=-150.0*u.km/u.s
        )
    ),
    (
        False,
        coord.Galactocentric(
            x=0.0*u.kpc,
            y=10.0*u.kpc,
            z=10.0*u.kpc,
            v_x=-150.0*u.km/u.s,
            v_y=0.0*u.km/u.s,
            v_z=100.0*u.km/u.s
        )
    )
]

@pytest.mark.parametrize("match,coord_gc", test_params)
def test_dataset_mini(match, coord_gc):
    """
    Test detection on a mini mock dataset
    """

    seed = 24
    mass_gc = 1e5 # in Msun

    # Spatial and magnitude ranges to find streams
    field_range = (-10.0, 10.0) # spatial range in deg
    mag_cut = 20.0 # magnitude cut
    mag_range = (15.0, mag_cut) # magnitude range

    # Obtain the phi1phi2 frame
    skycoord_gc = coord_gc.transform_to(coord.ICRS())
    frame = calc_prog_frame(skycoord_gc)

    # Load mock dataset
    path = os.path.join(os.path.dirname(__file__), "data")
    dataset = np.loadtxt(os.path.join(path, "mock_dataset_mini.txt"))
    data_all = dataset[:,0:6]
    err_all = dataset[:,6:12]
    label_all = dataset[:,12].astype(bool)
    ftrue = np.sum(label_all) / len(label_all)

    pot = agama.Potential(os.path.join(path, "MWPotential2014.ini"))

    iso_mini, iso_color, iso_mag = np.loadtxt(
        os.path.join(path, "mock_iso.txt"), unpack=True
    )

    # construct PDFs using the same method in Chen+25
    pdf_stream = GaiaStreamFromProgPDF(
        pot, mass_gc, coord_gc, iso_mini, iso_color, iso_mag, 
        mag_cut=mag_cut, frame=frame, t_tot=1.0, n_steps=500, 
        seed=seed
    )
    pdf_bg = GaiaBackgroundPDF(
        data_all, field_range=field_range, mag_cut=mag_cut, seed=seed
    )

    # estimate stream fraction using MLE
    frac_list = 10.0**np.arange(-2.0, -0.1+0.001, 0.02)
    fbest, prob, frac_list, dlnL_list = optimize_mle(
        data_all, err_all, pdf_stream, pdf_bg, frac_list=frac_list
    )

    if match:
        assert fbest > 0.5*ftrue and fbest < 2.0*ftrue
    else:
        assert fbest < 0.1*ftrue