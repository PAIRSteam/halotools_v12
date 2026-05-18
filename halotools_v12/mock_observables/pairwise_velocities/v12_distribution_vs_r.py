r"""
Module containing `v12_distribution_vs_r`: collect the full distribution of
pairwise radial velocities v12 in each radial bin, enabling histogram analysis.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import multiprocessing

from .engines import v12_distribution_vs_r_engine

from ..radial_profiles.radial_profiles_helpers import get_normalized_rbins

from functools import partial

from ..pair_counters.mesh_helpers import _set_approximate_cell_sizes, _cell1_parallelization_indices
from ..pair_counters.mesh_helpers import _enclose_in_box
from ..pair_counters.rectangular_mesh import RectangularDoubleMesh
from ..mock_observables_helpers import (enforce_sample_has_correct_shape,
    get_period, get_num_threads)

__all__ = ('v12_distribution_vs_r', )
__author__ = ('Antonela Taverna', )


def v12_distribution_vs_r(sample1, velocities1,
        rbins_absolute=None, rbins_normalized=None, normalize_rbins_by=None,
        sample2=None, velocities2=None, period=None,
        num_threads=1, approx_cell1_size=None, approx_cell2_size=None):
    r"""
    Collect the full distribution of pairwise radial velocities :math:`v_{12}(r)`
    in each radial bin.

    Parameters
    ----------
    sample1 : array_like
        Numpy array of shape (npts1, 3) containing 3-D positions.

    velocities1 : array_like
        Numpy array of shape (npts1, 3) containing 3-D velocities.

    rbins_absolute : array_like, optional
        Array of shape (num_rbins+1,) defining the bin edges in absolute length units.
        Either ``rbins_absolute`` or both ``rbins_normalized`` and
        ``normalize_rbins_by`` must be provided.

    rbins_normalized : array_like, optional
        Array of shape (num_rbins+1,) defining bin edges as :math:`r / R_{\rm norm}`.

    normalize_rbins_by : array_like, optional
        Array of shape (npts1,) with the normalization length for each point in sample1.

    sample2 : array_like, optional
        Numpy array of shape (npts2, 3) containing 3-D positions.

    velocities2 : array_like, optional
        Numpy array of shape (npts2, 3) containing 3-D velocities.

    period : array_like, optional
        Length-3 array for periodic boundary conditions.

    num_threads : int, optional
        Number of threads (processes). Default is 1.

    approx_cell1_size : array_like, optional
        Hint for cell sizes of sample1.

    approx_cell2_size : array_like, optional
        Hint for cell sizes of sample2.

    Returns
    -------
    v12_dist : list of numpy.ndarray
        List of length ``num_rbins``. Element ``i`` is a 1-D array of all
        pairwise radial velocity values :math:`v_{12}` for pairs whose separation
        falls in the annular bin ``(rbins[i], rbins[i+1]]``.

    Notes
    -----
    The pairwise radial velocity is:

    .. math::

        v_{12} = (\vec{v}_1 - \vec{v}_2) \cdot \hat{r}_{12}

    where :math:`\hat{r}_{12}` is the unit vector from object 2 to object 1.
    Positive values correspond to pairs moving apart.
    """
    result = _process_args(sample1, velocities1, sample2, velocities2,
        rbins_absolute, rbins_normalized, normalize_rbins_by,
        period, num_threads, approx_cell1_size, approx_cell2_size)

    sample1, velocities1, sample2, velocities2, max_rbins_absolute, period, \
        num_threads, _sample1_is_sample2, PBCs, \
        approx_cell1_size, approx_cell2_size, rbins_normalized, normalize_rbins_by = result

    x1in, y1in, z1in = sample1[:, 0], sample1[:, 1], sample1[:, 2]
    x2in, y2in, z2in = sample2[:, 0], sample2[:, 1], sample2[:, 2]
    xperiod, yperiod, zperiod = period
    squared_normalize_rbins_by = normalize_rbins_by * normalize_rbins_by
    search_xlength = max_rbins_absolute
    search_ylength = max_rbins_absolute
    search_zlength = max_rbins_absolute

    approx_cell1_size, approx_cell2_size = (
        _set_approximate_cell_sizes(approx_cell1_size, approx_cell2_size, period)
    )
    approx_x1cell_size, approx_y1cell_size, approx_z1cell_size = approx_cell1_size
    approx_x2cell_size, approx_y2cell_size, approx_z2cell_size = approx_cell2_size

    x1in, y1in, z1in = sample1[:, 0], sample1[:, 1], sample1[:, 2]
    x2in, y2in, z2in = sample2[:, 0], sample2[:, 1], sample2[:, 2]
    vx1in, vy1in, vz1in = velocities1[:, 0], velocities1[:, 1], velocities1[:, 2]
    vx2in, vy2in, vz2in = velocities2[:, 0], velocities2[:, 1], velocities2[:, 2]

    double_mesh = RectangularDoubleMesh(x1in, y1in, z1in, x2in, y2in, z2in,
        approx_x1cell_size, approx_y1cell_size, approx_z1cell_size,
        approx_x2cell_size, approx_y2cell_size, approx_z2cell_size,
        search_xlength, search_ylength, search_zlength, xperiod, yperiod, zperiod, PBCs)

    engine = partial(v12_distribution_vs_r_engine, double_mesh,
        x1in, y1in, z1in, x2in, y2in, z2in,
        vx1in, vy1in, vz1in, vx2in, vy2in, vz2in,
        squared_normalize_rbins_by, rbins_normalized)

    num_threads, cell1_tuples = _cell1_parallelization_indices(
        double_mesh.mesh1.ncells, num_threads)

    num_annular_bins = len(rbins_normalized) - 1

    if num_threads > 1:
        pool = multiprocessing.Pool(num_threads)
        thread_results = pool.map(engine, cell1_tuples)
        pool.close()
        # Merge per-bin lists from all threads
        combined = [[] for _ in range(num_annular_bins)]
        for thread_vrad_lists in thread_results:
            for bin_idx, vrad_vals in enumerate(thread_vrad_lists):
                combined[bin_idx].extend(vrad_vals)
    else:
        combined = engine(cell1_tuples[0])

    return [np.array(lst, dtype=np.float64) for lst in combined]


def _process_args(sample1, velocities1, sample2, velocities2,
        rbins_absolute, rbins_normalized, normalize_rbins_by,
        period, num_threads, approx_cell1_size, approx_cell2_size):
    rbins_normalized, normalize_rbins_by = get_normalized_rbins(
        rbins_absolute, rbins_normalized, normalize_rbins_by, sample1)

    max_rbins_absolute = np.amax(rbins_normalized) * np.amax(normalize_rbins_by)

    sample1 = enforce_sample_has_correct_shape(sample1)
    velocities1 = np.atleast_1d(velocities1).astype('f4')

    if sample2 is not None:
        sample2 = np.atleast_1d(sample2)
        if velocities2 is None:
            msg = ("\n If `sample2` is passed as an argument, \n"
                   "`velocities2` must also be specified.")
            raise ValueError(msg)
        else:
            velocities2 = np.atleast_1d(velocities2)

        _sample1_is_sample2 = False
        if np.all(sample1.shape == sample2.shape):
            if np.all(sample1 == sample2):
                _sample1_is_sample2 = True
    else:
        sample2 = sample1
        velocities2 = velocities1
        _sample1_is_sample2 = True

    x1 = sample1[:, 0]
    y1 = sample1[:, 1]
    z1 = sample1[:, 2]
    x2 = sample2[:, 0]
    y2 = sample2[:, 1]
    z2 = sample2[:, 2]
    period, PBCs = get_period(period)

    if period is None:
        PBCs = False
        x1, y1, z1, x2, y2, z2, period = (
            _enclose_in_box(x1, y1, z1, x2, y2, z2,
                min_size=[max_rbins_absolute*3.0, max_rbins_absolute*3.0, max_rbins_absolute*3.0]))
    else:
        PBCs = True
        period = np.atleast_1d(period).astype(float)
        if len(period) == 1:
            period = np.array([period[0]]*3)
        try:
            assert np.all(period < np.inf)
            assert np.all(period > 0)
        except AssertionError:
            msg = "Input ``period`` must be a bounded positive number in all dimensions"
            raise ValueError(msg)

    sample1[:, 0] = x1
    sample1[:, 1] = y1
    sample1[:, 2] = z1
    sample2[:, 0] = x2
    sample2[:, 1] = y2
    sample2[:, 2] = z2

    num_threads = get_num_threads(num_threads)

    if approx_cell1_size is None:
        approx_cell1_size = [max_rbins_absolute, max_rbins_absolute, max_rbins_absolute]
    elif len(np.atleast_1d(approx_cell1_size)) == 1:
        approx_cell1_size = [approx_cell1_size, approx_cell1_size, approx_cell1_size]
    if approx_cell2_size is None:
        approx_cell2_size = [max_rbins_absolute, max_rbins_absolute, max_rbins_absolute]
    elif len(np.atleast_1d(approx_cell2_size)) == 1:
        approx_cell2_size = [approx_cell2_size, approx_cell2_size, approx_cell2_size]

    return sample1, velocities1, sample2, velocities2, max_rbins_absolute, period, \
        num_threads, _sample1_is_sample2, PBCs, approx_cell1_size, approx_cell2_size, \
        rbins_normalized, normalize_rbins_by
