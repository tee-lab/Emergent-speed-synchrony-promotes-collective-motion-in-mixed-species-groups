"""
Implementation of a model with a positive feedback loop in swimming speeds and a mixed species equivalent,
implemented using a threshold behaviour in the preferred swimming speed,

Arshed Nabeel, May 2025
(c) TEE-Lab, IISc
"""

from attr import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from scipy.spatial import Delaunay

# from numba import njit


@dataclass
class SingleSpeciesModel():
    """ A model where the speed and orientation are coupled separately with different coupling
    constants. """
        
    # k : int             # Number of neighbours to interact
    v0 : float          # Preferred speed (can be set to 1 to de-dimensionalize velocity)
    mu_0 : float        # Speed relaxation (can be set to 1 to de-dimensionalize time)
    D : float           # Temparature (noise strength)

    r0 : float           # Repulsion zone radius
    mu_d : float        # Strength of attr/rep interaction

    mu_v : float         # Coupling constant for speed matching with group
    mu_al : float         # Coupling constant for alignment with group

    def simulate(self, N : int, T : int, dt : float,
                 r_init : NDArray[np.float64],
                 v_init : NDArray[np.float64], phi_init : NDArray[np.float64]):
        
        r = np.empty((T, N, 2))
        v = np.empty((T, N))     # Speed (scalar)
        # e = np.empty((T, N, 2))  # Orientation (unit vector)
        phi = np.empty((T, N))   # Orientation (angle, scalar)

        r[0, ...] = r_init
        v[0, ...] = v_init
        phi[0, ...] = phi_init
        # e[0, :, 0] = np.cos(phi[0, :])
        # e[0, :, 1] = np.sin(phi[0, :])
        
        for t in range(T - 1):
            f_v = np.empty((N, ))
            f_t = np.empty((N, ))
            f_d = np.empty((N, 2))

            e_t = np.vstack((np.cos(phi[t]), np.sin(phi[t]))) # (2, N)
            e_perp_t = np.vstack((-np.sin(phi[t]), np.cos(phi[t])))  # (2, N)
            v_t = v[t] * e_t  # (2, N)

            triangulation = Delaunay(r[t])
            ptr, idx = triangulation.vertex_neighbor_vertices
            for i in range(N):
                neighbours = idx[ptr[i]:ptr[i+1]]  # Voronoi neighbours of fish i
                # Compute local group velocity based on a random neighbourhood.
                # neighbours = np.random.choice([j for j in range(N) if j != i], size=self.k, replace=False)  
                r_ji = r[t, neighbours, :] - r[t, i, :]  # (6, 2)
                r_mod_ji = np.linalg.norm(r_ji, axis=1)       # (6)
                r_hat_ji = r_ji / r_mod_ji[:, None]
                # print(r_ji.shape, r_mod_ji.shape, r_hat_ji.shape)
                v_local = np.linalg.norm(np.mean(v_t[:, neighbours], axis=1))

                f_v[i] = v[t, i] - v_local
                f_t[i] = np.mean(np.sin(phi[t, i] - phi[t, neighbours]))
                f_d[i] = - np.nanmean((r_mod_ji[:, None] - self.r0) * r_hat_ji, axis=0)

            
            f_d_para = np.einsum('ij,ji->j', e_t, f_d)
            f_d_perp = np.einsum('ij,ji->j', e_perp_t, f_d)

            v[t + 1] = (v[t] + dt * (- self.mu_0 * (v[t] - self.v0)  # relaxation to v0
                                    # + self.D / v[t]                  # some temparature-dependent speeding-up?
                                    - self.mu_v * f_v
                                    - self.mu_d * f_d_para)             # Speed matching to v_local
                        + np.sqrt(dt * 2 * self.D) * np.random.normal(size=(N, )))  # Fluctuations
            
            phi[t + 1] = (phi[t] 
                          + dt * (-self.mu_al * f_t - self.mu_d * f_d_perp)
                          + np.sqrt(dt * 2 * self.D) * np.random.normal(size=(N, )) / v[t])
            phi[t + 1] %= (2 * np.pi)

            r[t + 1] = r[t] + v_t.T * dt

        return r, v, phi


@dataclass
class MixedSpeciesModel():
    """ 
    A model where the speed and orientation are coupled separately with different coupling
    constants. 
    The parameters k, v0, D, mu_0, mu_v, mu_al are tuples, giving the respective values for 
    species A and species B repectively. 
    """
        
    # k : Tuple[int, int]             # Number of neighbours to interact
    v0 : Tuple[float, float]          # Preferred speed (can be set to 1 to de-dimensionalize velocity)
    mu_0 : Tuple[float, float]        # Speed relaxation (can be set to 1 to de-dimensionalize time)
    D : Tuple[float, float]           # Temparature (noise strength)

    r0 : Tuple[float, float]           # Repulsion zone radius
    mu_d : np.ndarray                  # Strength of attr/rep interaction (2 x 2 ndarray)

    mu_v : Tuple[float, float]        # Coupling constant for speed matching with group
    mu_al : Tuple[float, float]       # Coupling constant for alignment with group


    def simulate(self, N : Tuple[int, int], T : int, dt : float,
                 r_init : NDArray[np.float64],
                 v_init : NDArray[np.float64], phi_init : NDArray[np.float64]):
        """
        N is a tuple (N_A, N_B), denoting the number of elements of each species.
        v_init, phi_init should be of size (T, NA + NB) respectively.
        """

        NA, NB = N
        r = np.empty((T, NA + NB, 2))
        v = np.empty((T, NA + NB))
        phi = np.empty((T, NA + NB))

        r[0, ...] = r_init
        v[0, ...] = v_init
        phi[0, ...] = phi_init

        for t in range(T - 1):
            f_v = np.empty((NA + NB, ))
            f_t = np.empty((NA + NB, ))
            f_d = np.empty((NA + NB, 2))

            e_t = np.vstack((np.cos(phi[t]), np.sin(phi[t]))) # (2, N)
            e_perp_t = np.vstack((-np.sin(phi[t]), np.cos(phi[t])))  # (2, N)
            v_t = v[t] * e_t  # (2, N)

            triangulation = Delaunay(r[t])
            ptr, idx = triangulation.vertex_neighbor_vertices
            for i in range(NA + NB):
                neighbours = idx[ptr[i]:ptr[i+1]]  # Voronoi neighbours of fish i
                # print(type(neighbours))
                species_a = (neighbours < NA)
                species_b = ~species_a
                # neighbours_a = [j for j in neighbours if neighbours < NA]  # Neighbours of species A
                # neighbours_b = [j for j in neighbours if neighbours > NA]  # Neighbours of species B
                r_ji = r[t, neighbours, :] - r[t, i, :]  # (6, 2)
                r_mod_ji = np.linalg.norm(r_ji, axis=1)       # (6)
                r_hat_ji = r_ji / r_mod_ji[:, None]
                v_local = np.linalg.norm(np.mean(v_t[:, neighbours], axis=1))

                f_v[i] = v[t, i] - v_local
                f_t[i] = np.mean(np.sin(phi[t, i] - phi[t, neighbours]))

                r0 = self.r0[0] if i < NA else self.r0[1]
                mu_d = self.mu_d[0] if i < NA else self.mu_d[1]

                f_d[i] = (mu_d[0] * np.sum((r_mod_ji[species_a, None] - r0) * r_hat_ji[species_a], axis=0)
                          + mu_d[1] * np.sum((r_mod_ji[species_b, None] - r0) * r_hat_ji[species_b], axis=0))

            f_d_para = np.einsum('ij,ji->j', e_t, f_d)
            f_d_perp = np.einsum('ij,ji->j', e_perp_t, f_d)
                
            # print(- self.mu_0[0] * (v[t, :NA] - self.v0[0]))
            # print(v[t, :NA])
            # print((v[t, :NA] + dt * (- self.mu_0[0] * (v[t, :NA] - self.v0[0]))
            # Update velocity and orientation of Species A
            v[t + 1, :NA] = (v[t, :NA] + dt * (- self.mu_0[0] * (v[t, :NA] - self.v0[0])  # relaxation to v0
                                    # + self.D[0] / v[t, :NA]                  # some temparature-dependent speeding-up?
                                    - self.mu_v[0] * f_v[:NA]
                                    + f_d_para[:NA])             # Speed matching to v_local
                        + np.sqrt(dt * 2 * self.D[0]) * np.random.normal(size=(NA, )))  # Fluctuations
            
            phi[t + 1, :NA] = (phi[t, :NA]
                               + dt * (-self.mu_al[0] * f_t[:NA] + f_d_perp[:NA] / v[t, :NA])
                               + np.sqrt(dt * 2 * self.D[0]) * np.random.normal(size=(NA, )) / v[t, :NA])

            # Update velocity and orientation of Species B
            v[t + 1, NA:] = (v[t, NA:] + dt * (- self.mu_0[1] * (v[t, NA:] - self.v0[1])  # relaxation to v0
                                    # + self.D[1] / v[t, NA:]                  # some temparature-dependent speeding-up?
                                    - self.mu_v[1] * f_v[NA:]
                                    + f_d_para[NA:])             # Speed matching to v_local
                        + np.sqrt(dt * 2 * self.D[1]) * np.random.normal(size=(NB, )))  # Fluctuations
            
            phi[t + 1, NA:] = (phi[t, NA:]
                               + dt *(- self.mu_al[1] * f_t[NA:] + f_d_perp[NA:] / v[t, NA:])
                               + np.sqrt(dt * 2 * self.D[1]) * np.random.normal(size=(NB, )) / v[t, NA:])
            
            phi[t + 1, ...] %= (2 * np.pi)
            r[t + 1] = r[t] + v_t.T * dt

        return r, v, phi


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from animate import animate_trajectories

    # model = SingleSpeciesModel(
    #     # k=6,
    #     v0=10,
    #     mu_0=1,
    #     mu_v=5.5,
    #     mu_al=4.5,
    #     r0=1,
    #     mu_d=1,
    #     D=2,
    # )
    

    # N = 16
    # eta = np.pi
    # r, v, phi = model.simulate(N=N, T=10000, dt=0.01,
    #                               v_init=np.full(N, 10),
    #                               r_init=np.random.uniform(0, 5*N, size=(N, 2)),
    #                               phi_init=np.linspace(-eta, +eta, N))
    # print(r)

    model = MixedSpeciesModel(
        # k=6,
        v0=(8, 8),
        mu_0=(1, 1),
        mu_v=(2, 5),
        mu_al=(4, 4),
        r0=(.5, .5),
        mu_d=np.array(((.5, .1), (.1, .5))),
        D=(10, 10),
    )
    

    NA, NB = 4, 12
    eta = np.pi
    r, v, phi = model.simulate(N=(NA, NB), T=10000, dt=0.01,
                                  v_init=np.full(NA + NB, 10),
                                  r_init=np.random.uniform(0, 2, size=(NA + NB, 2)),
                                  phi_init=np.linspace(-eta, +eta, NA + NB))

    plt.plot(r[:, :, 0], r[:, :, 1])
    # m = np.mean(e, axis=1)
    # s = np.mean(v, axis=1)
    # modm = np.linalg.norm(m, axis=1)
    # # print(v.shape, modm.shape, )
    # plt.hist2d(modm, s, bins=100)
    plt.show()
    animate_trajectories(r[1000:, ...] % 50, N=(NA, NB),
                         trail_length=30, L=50, n_frames=None, filename='mixsp-sample.mp4')

    