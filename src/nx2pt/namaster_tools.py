import os
import numpy as np
import healpy as hp
import pymaster as nmt
import joblib


def get_workspace(nmt_field1, nmt_field2, nmt_bins, wksp_cache=None):
    """Get the NmtWorkspace for given fields and bins (with caching)."""

    if wksp_cache is None:
        wksp = nmt.NmtWorkspace.from_fields(nmt_field1, nmt_field2, nmt_bins)
        return wksp

    # hash on mask alms (to support catalog fields) and spins
    hash_key = joblib.hash([nmt_field1.get_mask_alms(), nmt_field1.spin, nmt_field2.get_mask_alms(), nmt_field2.spin])
    wksp_file = f"{wksp_cache}/cl/{hash_key}.fits"

    try:
        # load from existing file
        wksp = nmt.NmtWorkspace.from_file(wksp_file)
        wksp.check_unbinned()
        print("Using cached workspace")
        # update bins and beams after loading
        wksp.update_beams(nmt_field1.beam, nmt_field2.beam)
        wksp.update_bins(nmt_bins)
    except RuntimeError:
        # compute and save to file
        wksp = nmt.NmtWorkspace.from_fields(nmt_field1, nmt_field2, nmt_bins)
        os.makedirs(f"{wksp_cache}/cl", exist_ok=True)
        wksp.write_to(wksp_file)

    return wksp


def get_cov_workspace(nmt_field1a, nmt_field2a, nmt_field1b=None, nmt_field2b=None, wksp_cache=None):
    """
    Get the NmtCovarianceWorkspace object needed to calculate the covariance between the
    cross-spectra (field1a, field2a) and (field1b, field2b).
    """
    if nmt_field1b is None and nmt_field2b is None:
        nmt_field1b = nmt_field1a
        nmt_field2b = nmt_field2a
    elif nmt_field1b is None or nmt_field2b is None:
        raise ValueError("Must provide either 2 or 4 fields")

    if wksp_cache is None:
        wksp = nmt.NmtCovarianceWorkspace.from_fields(nmt_field1a, nmt_field2a, nmt_field1b, nmt_field2b)
        return wksp

    # hash masks and spins
    hash_key = joblib.hash([nmt_field1a.get_mask(), nmt_field1a.spin, nmt_field2a.get_mask(), nmt_field2a.spin,
                            nmt_field1b.get_mask(), nmt_field1b.spin, nmt_field2b.get_mask(), nmt_field2b.spin])
    wksp_file = f"{wksp_cache}/cov/{hash_key}.fits"

    try:
        wksp = nmt.NmtCovarianceWorkspace.from_file(wksp_file)
        print("Using cached workspace")
    except RuntimeError:
        wksp = nmt.NmtCovarianceWorkspace.from_fields(nmt_field1a, nmt_field2a, nmt_field1b, nmt_field2b)
        os.makedirs(f"{wksp_cache}/cov", exist_ok=True)
        wksp.write_to(wksp_file)

    return wksp


def compute_cl(wksp_dir, nmt_field1, nmt_field2, nmt_bins, return_bpw=False):
    """Calculate the x-spectrum between tracer1 and tracer2."""
    wksp = get_workspace(wksp_dir, nmt_field1, nmt_field2, nmt_bins)
    print(wksp.wsp.lmax, wksp.wsp.lmax_mask)
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
    print(cov_wksp.wsp.lmax, cov_wksp.wsp.lmax_mask)
    wksp_a = get_workspace(wksp_dir, nmt_field1a, nmt_field2a, nmt_bins)
    wksp_b = get_workspace(wksp_dir, nmt_field1b, nmt_field2b, nmt_bins)
    print(wksp_a.wsp.lmax, wksp_a.wsp.lmax_mask)
    print(wksp_b.wsp.lmax, wksp_b.wsp.lmax_mask)

    spins = [nmt_field1a.spin, nmt_field2a.spin, nmt_field1b.spin, nmt_field2b.spin]
    print(spins)

    # iNKA approximation: get coupled cls divded by mean of product of masks
    pcl1a1b = nmt.compute_coupled_cell(nmt_field1a, nmt_field1b) / fsky(nmt_field1a, nmt_field1b)
    pcl2a1b = nmt.compute_coupled_cell(nmt_field2a, nmt_field1b) / fsky(nmt_field2a, nmt_field1b)
    pcl1a2b = nmt.compute_coupled_cell(nmt_field1a, nmt_field2b) / fsky(nmt_field1a, nmt_field2b)
    pcl2a2b = nmt.compute_coupled_cell(nmt_field2a, nmt_field2b) / fsky(nmt_field2a, nmt_field2b)

    if np.isnan(pcl1a1b).any(): print("pcl1a1b has nans")
    if np.isnan(pcl1a2b).any(): print("pcl1a2b has nans")
    if np.isnan(pcl2a1b).any(): print("pcl2a1b has nans")
    if np.isnan(pcl2a2b).any(): print("pcl2a2b has nans")

    cov = nmt.gaussian_covariance(cov_wksp, *spins, pcl1a1b, pcl1a2b, pcl2a1b, pcl2a2b, wksp_a, wksp_b)
    if np.isnan(cov).any(): print("cov has nans")
    return cov


def parse_tracer_bin(tracer_bin_key):
    """Takes a string of the form tracer_name_{int} and returns tracer_name, int."""
    key_split = tracer_bin_key.split('_')
    tracer_name = '_'.join(key_split[:-1])
    tracer_bin = int(key_split[-1])
    return tracer_name, tracer_bin


def parse_cl_key(cl_key):
    tracer_bin_keys = cl_key.split(', ')
    return list(map(parse_tracer_bin, tracer_bin_keys))


def compute_cls_cov(tracers, xspectra_list, bins, compute_cov=True, compute_interbin_cov=True, wksp_cache=None):
    """Calculate all cross-spectra and covariances from a list of tracers."""
    cls = dict()
    wksps = dict()
    bpws = dict()
    # loop over all cross-spectra
    for xspec in xspectra_list:
        tracer1_key, tracer2_key = xspec
        tracer1 = tracers[tracer1_key]
        tracer2 = tracers[tracer2_key]
        # loop over all bins
        for i in range(len(tracer1)):
            for j in range(i, len(tracer2)):
                cl_key = f"{tracer1_key}_{i}, {tracer2_key}_{j}"
                print("computing cross-spectrum", cl_key)
                wksp = get_workspace(tracer1[i].field, tracer2[j].field, bins, wksp_cache=wksp_cache)
                pcl = nmt.compute_coupled_cell(tracer1[i].field, tracer2[j].field)
                cl = wksp.decouple_cell(pcl)
                # save quantities
                cls[cl_key] = cl
                wksps[cl_key] = wksp
                bpws[cl_key] = wksp.get_bandpower_windows()

    covs = dict()
    if not compute_cov:
        return cls, bpws, covs

    # loop over all covariances
    cl_keys = list(cls.keys())
    for i in range(len(cl_keys)):
        cl_key_a = cl_keys[i]
        (tracer_a1_key, bin_a1), (tracer_a2_key, bin_a2) = parse_cl_key(cl_key_a)
        field_a1 = tracers[tracer_a1_key][bin_a1].field
        field_a2 = tracers[tracer_a2_key][bin_a2].field

        # skip covariances that involve a catalog field
        if tracers[tracer_a1_key][bin_a1].is_cat_field or tracers[tracer_a2_key][bin_a2].is_cat_field:
            print("Skipping covariances involving catalog field (not implemented yet)")
            continue

        for j in range(i, len(cl_keys)):
            cl_key_b = cl_keys[j]

            if not compute_interbin_cov:
                # skip all off-diagonal covs
                if cl_key_a != cl_key_b:
                    continue

            (tracer_b1_key, bin_b1), (tracer_b2_key, bin_b2) = parse_cl_key(cl_key_b)
            field_b1 = tracers[tracer_b1_key][bin_b1].field
            field_b2 = tracers[tracer_b2_key][bin_b2].field

            # skip covariances that involve a catalog field
            if tracers[tracer_b1_key][bin_b1].is_cat_field or tracers[tracer_b2_key][bin_b2].is_cat_field:
                print("Skipping covariances involving catalog field (not implemented yet)")
                continue

            cov_key = f"{cl_key_a}, {cl_key_b}"
            print("computing covariance", cov_key)
            cov_wksp = get_cov_workspace(field_a1, field_a2, field_b1, field_b2, wksp_cache=wksp_cache)
            wksp_a = wksps[cl_key_a]
            wksp_b = wksps[cl_key_b]
            spins = [field_a1.spin, field_a2.spin, field_b1.spin, field_b2.spin]
            pcl_a1b1 = nmt.compute_coupled_cell(field_a1, field_b1) / fsky(field_a1, field_b1)
            pcl_a1b2 = nmt.compute_coupled_cell(field_a1, field_b2) / fsky(field_a1, field_b2)
            pcl_a2b1 = nmt.compute_coupled_cell(field_a2, field_b1) / fsky(field_a2, field_b1)
            pcl_a2b2 = nmt.compute_coupled_cell(field_a2, field_b2) / fsky(field_a2, field_b2)

            cov = nmt.gaussian_covariance(cov_wksp, *spins, pcl_a1b1, pcl_a1b2, pcl_a2b1, pcl_a2b2,
                                          wksp_a, wksp_b)
            if np.isnan(cov).any(): print("cov has nans")
            covs[cov_key] = cov

    return cls, bpws, covs
