# Licensed under BSD-3-Clause License - see LICENSE

import numpy as np
import astropy.coordinates as coord

__all__ = [
    "sig_pm_from_G_gdr3",
    "sig_color_from_G_gdr3",
    "make_frame_from_mat",
    "calc_prog_frame",
    "kroupa_cdf",
    "assign_mini",
    "assign_color_mag"
]

def sig_pm_from_G_gdr3(G):
    # Gaia DR3 proper motion error as a function of G
    z = 10**(0.4*(G-15))
    z[G<13] = 10**(0.4*(-2))
    return 1 * np.sqrt(40 + 800*z + 30*z**2) / 1000 # in mas/yr

def sig_color_from_G_gdr3(G):
    # Gaia DR3 color error as a function of G
    G_list = [4,14,20]
    log_sig_color_list = [-3,-3,-1]
    return 10**(np.interp(G, G_list, log_sig_color_list))

def make_frame_from_mat(mat):
    class NewFrame(coord.BaseCoordinateFrame):
        default_representation = coord.SphericalRepresentation
        default_differential = coord.SphericalCosLatDifferential

        frame_specific_representation_info = {
            coord.SphericalRepresentation: [
                coord.RepresentationMapping("lon", "phi1"),
                coord.RepresentationMapping("lat", "phi2"),
                coord.RepresentationMapping("distance", "distance"),
            ]
        }

    @coord.frame_transform_graph.transform(
        coord.StaticMatrixTransform, coord.ICRS, NewFrame
    )
    def icrs_to_NewFrame():
        return mat
    
    @coord.frame_transform_graph.transform(
        coord.StaticMatrixTransform, NewFrame, coord.ICRS
    )
    def NewFrame_to_icrs():
        return np.transpose(mat)
    
    return NewFrame

def calc_prog_frame(skycoord):
    """
    skycoord: astropy.coord.SkyCoord
    """

    ra = skycoord.ra.to_value("rad")
    dec = skycoord.dec.to_value("rad")
    mu_ra_cosdec = skycoord.pm_ra_cosdec.to_value("mas/yr")
    mu_dec = skycoord.pm_dec.to_value("mas/yr")

    star_vec = np.array([
        np.cos(dec)*np.cos(ra), 
        np.cos(dec)*np.sin(ra), 
        np.sin(dec)
    ])
    star_vec /= np.linalg.norm(star_vec)

    e_ra = np.array([-np.sin(ra),  np.cos(ra), 0.0])
    e_dec= np.array([
        -np.sin(dec)*np.cos(ra), 
        -np.sin(dec)*np.sin(ra), 
        np.cos(dec)
    ])

    pm_vec = mu_ra_cosdec * e_ra + mu_dec * e_dec
    norm_pm = np.linalg.norm(pm_vec)
    if norm_pm > 0:
        pm_unit = pm_vec / norm_pm
    else:
        # fallback: pick any tangent direction orthogonal to star_vec
        pm_unit = e_ra - np.dot(e_ra, star_vec)*star_vec
        pm_unit /= np.linalg.norm(pm_unit)

    y_axis = pm_unit - np.dot(pm_unit, star_vec)*star_vec
    y_axis /= np.linalg.norm(y_axis)

    z_axis = np.cross(star_vec, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    mat = np.vstack([star_vec, y_axis, z_axis])

    return make_frame_from_mat(mat)

def kroupa_cdf(m):
    """
    Kroupa CDF
    """
    m = np.asanyarray(m)
    m_min, m_brk, m_max = 0.1, 0.5, 1.0
    a1, a2 = 1.3, 2.3

    # continuity factor between the two power‑law segments
    A2_over_A1 = m_brk**(a2 - a1)

    # primitive (indefinite integral) of each segment, omitting the overall A1
    def I1(x):
        return (x**(1-a1) - m_min**(1-a1)) / (1-a1)

    def I2(x):
        return A2_over_A1 * (x**(1-a2) - m_brk**(1-a2)) / (1-a2)

    # total integral for normalisation
    I1_full = I1(m_brk)
    I2_full = I2(m_max)
    norm    = I1_full + I2_full

    # piecewise CDF
    F = np.where(
        m < m_brk,
        I1(m) / norm,
        (I1_full + I2(m)) / norm
    )

    return F.item() if np.isscalar(m) else F

def assign_mini(mmin, mmax, n_star, mf_cdf=kroupa_cdf, seed=0):
    """
    Assign initial mass to individual stars
    mmin, mmax: stellar mass ranges in Msun
    n_star: number of stars
    mf_cdf: CDF of mass function, default: kroupa_cdf
    seed: random seed
    """ 
    rng = np.random.default_rng(seed)
    mgrid = np.linspace(mmin,mmax,100)
    mini = np.interp(
        rng.uniform(low=mf_cdf(mmin), high=mf_cdf(mmax), size=n_star),
        kroupa_cdf(mgrid), mgrid
    )
    return mini

def assign_color_mag(
    distance, mag_cut, iso_mini, iso_color, iso_mag, 
    mf_cdf=kroupa_cdf, seed=0
):
    """
    distance: distance array of stars in kpc
    mag_cut: magnitude faint-end cut
    iso_mini: initial mass column of isochrone
    iso_color: color column of isochrone
    iso_mag: magnitude column of isochrone
    mf_cdf: CDF of mass function, default: kroupa_cdf
    seed: random seed
    """
    modulus = 5*(np.log10(distance) + 2)
    absmag_high = mag_cut - np.min(modulus) # the highest possible mag

    mmax = iso_mini.max()
    mmin = iso_mini[iso_mag < absmag_high].min()

    mini = assign_mini(
        mmin=mmin, mmax=mmax, n_star=len(distance), seed=seed
    )

    mag_stream = np.interp(mini, iso_mini, iso_mag) + modulus
    color_stream = np.interp(mini, iso_mini, iso_color)

    return mag_stream, color_stream
