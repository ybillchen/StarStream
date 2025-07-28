# Licensed under BSD-3-Clause License - see LICENSE

"""
Agama implementation of Chen+25 particle spray algorithm
"""

import numpy as np
import agama
agama.setUnits(length=1, velocity=1, mass=1)

def get_rot_mat(x, y, z, vx, vy, vz):
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    r = (x*x + y*y + z*z)**0.5
    L = (Lx*Lx + Ly*Ly + Lz*Lz)**0.5
    # rotation matrices transforming from the host to the satellite frame for each point on the trajectory
    R = np.zeros((len(x), 3, 3))
    R[:,0,0] = x/r
    R[:,0,1] = y/r
    R[:,0,2] = z/r
    R[:,2,0] = Lx/L
    R[:,2,1] = Ly/L
    R[:,2,2] = Lz/L
    R[:,1,0] = R[:,0,2] * R[:,2,1] - R[:,0,1] * R[:,2,2]
    R[:,1,1] = R[:,0,0] * R[:,2,2] - R[:,0,2] * R[:,2,0]
    R[:,1,2] = R[:,0,1] * R[:,2,0] - R[:,0,0] * R[:,2,1]
    return R, L, r

def get_d2Phi_dr2(pot_host, x, y, z):
    # compute  the second derivative of potential by spherical radius
    r = (x*x + y*y + z*z)**0.5
    der = pot_host.forceDeriv(np.column_stack([x,y,z]))[1]
    d2Phi_dr2 = -(x**2  * der[:,0] + y**2  * der[:,1] + z**2  * der[:,2] +
                  2*x*y * der[:,3] + 2*y*z * der[:,4] + 2*z*x * der[:,5]) / r**2
    return d2Phi_dr2

def create_ic_chen25(rng, pot_host, orb_sat, mass_sat):
    N = len(orb_sat)
    x, y, z, vx, vy, vz = orb_sat.T
    R, L, r = get_rot_mat(x, y, z, vx, vy, vz)
    d2Phi_dr2 = get_d2Phi_dr2(pot_host, x, y, z)
    
    # compute the tidal radius at this radius for each point on the trajectory
    Omega = L / r**2
    r_tidal = (agama.G * mass_sat / (Omega**2 - d2Phi_dr2))**(1./3)
    
    # assign positions and velocities (in the satellite reference frame) of particles
    r_tidal = np.repeat(r_tidal, 2)

    mean = np.array([1.6, -30, 0, 1, 20, 0])

    cov = np.array([
        [0.1225,   0,   0, 0, -4.9,   0],
        [     0, 529,   0, 0,    0,   0],
        [     0,   0, 144, 0,    0,   0],
        [     0,   0,   0, 0,    0,   0],
        [  -4.9,   0,   0, 0,  400,   0],
        [     0,   0,   0, 0,    0, 484],
    ])
    
    posvel = rng.multivariate_normal(mean, cov, size=2*N)

    Dr = posvel[:, 0] * r_tidal
    v_esc = np.sqrt(2 * agama.G * mass_sat / Dr)
    Dv = posvel[:, 3] * v_esc

    # convert degrees to radians
    phi = posvel[:, 1] * np.pi / 180
    theta = posvel[:, 2] * np.pi / 180
    alpha = posvel[:, 4] * np.pi / 180
    beta = posvel[:, 5] * np.pi / 180

    dx = Dr * np.cos(theta) * np.cos(phi)
    dy = Dr * np.cos(theta) * np.sin(phi)
    dz = Dr * np.sin(theta)
    dvx = Dv * np.cos(beta) * np.cos(alpha)
    dvy = Dv * np.cos(beta) * np.sin(alpha)
    dvz = Dv * np.sin(beta)

    dq = np.column_stack([dx, dy, dz])
    dp = np.column_stack([dvx, dvy, dvz])
    
    ic_stream = np.tile(orb_sat, 2).reshape(2*N, 6)
    
    # trailing arm
    ic_stream[::2,0:3] += np.einsum("ni,nij->nj", dq[::2], R)
    ic_stream[::2,3:6] += np.einsum("ni,nij->nj", dp[::2], R)
    
    # leading arm
    ic_stream[1::2,0:3] += np.einsum("ni,nij->nj", -dq[1::2], R)
    ic_stream[1::2,3:6] += np.einsum("ni,nij->nj", -dp[1::2], R)

    return ic_stream

def integrate_prog(time_total, trajsize, pot_host, posvel_sat):
    # integrate the orbit of the progenitor from its present-day posvel (at time t=0)
    # back in time for an interval time_total, storing the trajectory at num_steps points
    time_sat, orbit_sat = agama.orbit(potential=pot_host, ic=posvel_sat,
        time=time_total, trajsize=trajsize)
    if time_total < 0:
        # reverse the arrays to make them increasing in time
        time_sat  = time_sat [::-1]
        orbit_sat = orbit_sat[::-1]
    return time_sat, orbit_sat

def create_stream(create_ic_method, rng, time_total, num_particles, pot_host, posvel_sat, mass_sat, 
    pot_sat=None, nhalf_release=None, **kwargs):

    if nhalf_release is None:
        trajsize = num_particles//2
    else:
        trajsize = len(nhalf_release)
    assert len(nhalf_release) == trajsize
    time_sat, orbit_sat = integrate_prog(time_total, trajsize, pot_host, posvel_sat)

    # at each point on the trajectory, create a pair of seed initial conditions
    # for particles released at Lagrange points
    if nhalf_release is None:
        release_points = orbit_sat
        time_seed = np.repeat(time_sat, 2)
    else:
        release_points = np.repeat(orbit_sat, nhalf_release, axis=0)
        time_seed = np.repeat(time_sat, 2*nhalf_release)
    ic_stream = create_ic_method(rng, pot_host, release_points, mass_sat, **kwargs)
    
    if pot_sat is None:
        pot_tot = pot_host
    else:
        # include the progenitor"s potential
        traj = np.column_stack([time_sat, orbit_sat])
        pot_traj = agama.Potential(potential=pot_sat, center=traj)
        pot_tot = agama.Potential(pot_host, pot_traj)
        
    dur = -time_seed if time_total<0 else time_total-time_seed
    ic_stream = ic_stream[dur>0]
    time_seed = time_seed[dur>0]
    xv_stream = np.vstack(agama.orbit(potential=pot_tot,
        ic=ic_stream, time=dur[dur>0], timestart=time_seed, trajsize=1)[:,1])
    return time_sat, orbit_sat, xv_stream, ic_stream, time_seed