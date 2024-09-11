import os
import numpy as np
import healpy as hp
import pymaster as nmt
import joblib


def get_workspace(wksp_dir, nmt_field1, nmt_field2, nmt_bins):
    """Get the NmtWorkspace for given fields and bins (with caching)."""
    # only need to hash based on masks
    hash_key = joblib.hash([nmt_field1.get_mask(), nmt_field2.get_mask()])
    wksp_file = f"{wksp_dir}/cl/{hash_key}.fits"

    try:
        # load from existing file
        wksp = nmt.NmtWorkspace.from_file(wksp_file)
        # update bins and beams after loading
        wksp.update_beams(nmt_field1.beam, nmt_field2.beam)
        wksp.update_bins(nmt_bins)
    except RuntimeError:
        # compute and save to file
        wksp = nmt.NmtWorkspace.from_fields(nmt_field1, nmt_field2, nmt_bins)
        os.makedirs(f"{wksp_dir}/cl", exist_ok=True)
        wksp.write_to(wksp_file)

    return wksp


def get_cov_workspace(wksp_dir, nmt_field1a, nmt_field2a, nmt_field1b=None, nmt_field2b=None):
    """
    Get the NmtCovarianceWorkspace object needed to calculate the covariance between the
    cross-spectra (field1a, field2a) and (field1b, field2b).
    """
    if nmt_field1b is None and nmt_field2b is None:
        nmt_field1b = nmt_field1a
        nmt_field2b = nmt_field2a
    elif nmt_field1b is None or nmt_field2b is None:
        raise ValueError("Must provide either 2 or 4 fields")

    # only need to hash masks
    hash_key = joblib.hash([nmt_field1a.get_mask(), nmt_field2a.get_mask(), nmt_field1b.get_mask(), nmt_field2b.get_mask()])
    wksp_file = f"{wksp_dir}/cov/{hash_key}.fits"

    try:
        wksp = nmt.NmtCovarianceWorkspace.from_file(wksp_file)
    except RuntimeError:
        wksp = nmt.NmtCovarianceWorkspace.from_fields(nmt_field1a, nmt_field2a, nmt_field1b, nmt_field2b)
        os.makedirs(f"{wksp_dir}/cov", exist_ok=True)
        wksp.write_to(wksp_file)

    return wksp


def compute_cl(wksp_dir, nmt_field1, nmt_field2, nmt_bins, return_bpw=False):
    """Calculate the x-spectrum between tracer1 and tracer2."""
    wksp = get_workspace(wksp_dir, nmt_field1, nmt_field2, nmt_bins)
    pcl = nmt.compute_coupled_cell(nmt_field1, nmt_field2)
    cl = wksp.decouple_cell(pcl)
    if return_bpw:
        return cl, wksp.get_bandpower_windows()
    return cl


def fsky(nmt_field1, nmt_field2):
    return np.mean(nmt_field1.get_mask() * nmt_field2.get_mask())


def compute_gaussian_cov(wksp_dir, nmt_field1a, nmt_field2a, nmt_field1b, nmt_field2b, nmt_bins):
    """Compute the Gaussian covariance between powerspectra A and B."""
    # get workspaces
    cov_wksp = get_cov_workspace(wksp_dir, nmt_field1a, nmt_field2a, nmt_field1b, nmt_field2b)
    wksp_a = get_workspace(wksp_dir, nmt_field1a, nmt_field2a, nmt_bins)
    wksp_b = get_workspace(wksp_dir, nmt_field1b, nmt_field2b, nmt_bins)

    spins = [nmt_field1a.spin, nmt_field2a.spin, nmt_field1b.spin, nmt_field2b.spin]

    # iNKA approximation: get coupled cls divded by mean of product of masks
    pcl1a1b = nmt.compute_coupled_cell(nmt_field1a, nmt_field1b) / fsky(nmt_field1a, nmt_field1b)
    pcl2a1b = nmt.compute_coupled_cell(nmt_field2a, nmt_field1b) / fsky(nmt_field2a, nmt_field1b)
    pcl1a2b = nmt.compute_coupled_cell(nmt_field1a, nmt_field2b) / fsky(nmt_field1a, nmt_field2b)
    pcl2a2b = nmt.compute_coupled_cell(nmt_field2a, nmt_field2b) / fsky(nmt_field2a, nmt_field2b)

    cov = nmt.gaussian_covariance(cov_wksp, *spins, pcl1a1b, pcl1a2b, pcl2a1b, pcl2a2b, wksp_a, wksp_b)
    return cov
