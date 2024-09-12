import os
import sys
import yaml
import numpy as np
import healpy as hp
import pymaster as nmt
import joblib
import sacc

import nx2pt
from nx2pt import MapTracer


def get_ell_bins(config):
    """Generate ell bins from config."""
    nside = config["nside"]
    ell_min = config["ell_min"]
    dl = config["delta_ell"]
    ell_bins = np.linspace(ell_min, 3*nside, int((3*nside - ell_min) / dl) + 1, dtype=int)
    return ell_bins


def get_tracer(config, key):
    """Load tracer information."""
    nside = config["nside"]
    name = config[key]["name"]
    data_dir = config[key]["data_dir"]
    if "bins" in config[key].keys():
        bins = config[key]["bins"]
    else:
        bins = 1
    if "use_mask_squared" in config[key].keys():
        use_mask_squared = config[key]["use_mask_squared"]
    else:
        use_mask_squared = False
    if "correct_qu_sign" in config[key].keys():
        correct_qu_sign = config[key]["correct_qu_sign"]
    else:
        correct_qu_sign = False

    print(name, f"({bins} bins)" if bins > 1 else '')

    tracer_bins = []
    for bin in range(bins):
        bin_name = name if bins == 1 else f"{name} (bin {bin})"
        map_file = data_dir + '/' + config[key]["map"].format(bin=bin, nside=nside)
        mask_file = data_dir + '/' + config[key]["mask"].format(bin=bin, nside=nside)
        if "beam" in config[key].keys():
            beam_file = data_dir + '/' + config[key]["beam"].format(bin=bin, nside=nside)
        else:
            beam = np.ones(3*nside)

        maps = np.atleast_2d(hp.read_map(map_file, field=None))
        if correct_qu_sign and len(maps) == 2:
            maps = np.array([-maps[0], maps[1]])

        mask = hp.read_map(mask_file)
        if use_mask_squared: mask = mask**2

        tracer = MapTracer(bin_name, maps, mask, beam=beam)
        print(tracer)
        tracer_bins.append(tracer)

    return tracer_bins


def save_sacc(config):
    pass


def save_npz(file_name, ell_eff, cls, covs, bpws):
    """Save cross-spectra, covariances, and bandpower windows to a .npz file."""
    assert bpws.keys() == cls.keys(), "Each cross-spectrum should have a corresponding bandpower window"
    save_dict = {"cl_" + str(cl_key): cls[cl_key] for cl_key in cls.keys()} | \
                {"cov_" + str(cov_key): covs[cov_key] for cov_key in covs.keys()} | \
                {"bpw_" + str(cl_key): bpws[cl_key] for cl_key in cls.keys()} | \
                {"ell_eff": ell_eff}
    np.savez(file_name, **save_dict)


def main():
    with open(sys.argv[1]) as f:
        config = yaml.full_load(f)

    print(config)

    tracer_keys = [key for key in config.keys() if key.startswith("tracer")]
    print(f"Found {len(tracer_keys)} tracers")
    tracers = dict()
    for tracer_key in tracer_keys:
        tracer = get_tracer(config, tracer_key)
        tracers[tracer_key] = tracer

    xspec_keys = [key for key in config.keys() if key.startswith("cross_spectra")]
    print(f"Found {len(xspec_keys)} set(s) of cross-spectra to calculate")
    for xspec_key in xspec_keys:
        if "save_npz" not in config[xspec_key].keys() and "save_sacc" not in config[xspec_key].keys():
            print(f"Warning! No output will be saved for the block {xspec_key}")

    ell_bins = get_ell_bins(config)
    nmt_bins = nmt.NmtBin.from_edges(ell_bins[:-1], ell_bins[1:])
    ell_eff = nmt_bins.get_effective_ells()
    print(f"Will calculate {len(ell_eff)} bandpowers between ell = {ell_bins[0]} and ell = {ell_bins[-1]}")
    wksp_dir = config["workspace_dir"]

    for xspec_key in xspec_keys:
        xspec_list = config[xspec_key]["list"]
        print("Computing set", xspec_list)

        calc_cov = False
        calc_interbin_cov = False
        if "covariance" in config[xspec_key].keys():
            calc_cov = config[xspec_key]["covariance"]
        if "interbin_cov" in config[xspec_key].keys():
            calc_interbin_cov = config[xspec_key]["interbin_cov"]

        cls, bpws, covs = nx2pt.compute_cls_cov(tracers, xspec_list, nmt_bins, compute_cov=calc_cov, compute_interbin_cov=calc_interbin_cov)

        # save all cross-spectra
        if "save_npz" in config[xspec_key].keys():
            save_npz_file = config[xspec_key]["save_npz"].format(nside=config["nside"])
            print("Saving to", save_npz_file)
            save_npz(save_npz_file, ell_eff, cls, covs, bpws)

        # create sacc file
        #if "save_sacc" in config.keys():
            #print("Creating sacc file")
