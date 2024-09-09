import os
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

    # only need to hash based on masks
    hash_key = joblib.hash([nmt_field1a.get_mask(), nmt_field2a.get_mask(), nmt_field1b.get_mask(), nmt_field2b.get_mask()])
    wksp_file = f"{wksp_dir}/cov/{hash_key}.fits"

    try:
        wksp = nmt.NmtCovarianceWorkspace.from_file(wksp_file)
    except RuntimeError:
        wksp = nmt.NmtCovarianceWorkspace.from_fields(nmt_field1a, nmt_field2a, nmt_field1b, nmt_field2b)
        os.makedirs(f"{wksp_dir}/cov", exist_ok=True)
        wksp.write_to(wksp_file)

    return wksp
