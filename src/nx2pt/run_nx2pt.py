import argparse
import os
import sys
import yaml
import numpy as np
import healpy as hp
import pymaster as nmt
import joblib
import sacc
from astropy.table import Table

from .data import Data
from .tracer import MapTracer, CatalogTracer
from .namaster_tools import get_bpw_edges, get_nmtbins
from .namaster_tools import compute_cls_cov
from .utils import get_ul_key, parse_cl_key, parse_tracer_bin


def get_ell_bins(nside, bin_config):
    """Generate ell bins from config."""
    bpw_edges = bin_config.get("bpw_edges", None)
    if bpw_edges is None:
        kind = bin_config.get("kind", "linear")
        lmin = bin_config.get("ell_min", 2)
        lmax = bin_config.get("ell_max", 3*nside-1)
        nbpws = bin_config.get("nbpws", None)
        if nbpws is None and kind == "linear":
            delta_ell = bin_config["delta_ell"]
            nbpws = (lmax - lmin) // delta_ell
        elif nbpws is None:
            raise ValueError("Must specify nbpws for non-linear binning")
        bpw_edges = get_bpw_edges(lmin, lmax, nbpws, kind)
    nmt_bin = get_nmtbins(nside, bpw_edges)
    return nmt_bin


def get_tracer(nside, tracer_config):
    """Load tracer information."""
    name = tracer_config["name"]
    data_dir = tracer_config["data_dir"]
    if "healpix" in tracer_config.keys():
        tracer_type = "healpix"
    elif "catalog" in tracer_config.keys():
        tracer_type = "catalog"
    else:
        raise ValueError(f"Tracer {key} must have either a 'healpix' or 'catalog' section")
    bins = tracer_config.get("bins", 1)
    use_mask_squared = tracer_config.get("use_mask_squared", False)
    correct_qu_sign = tracer_config.get("correct_qu_sign", False)

    print(name, f"({bins} bins)" if bins > 1 else '')

    tracer_bins = []
    for bin_i in range(bins):
        bin_name = name if bins == 1 else f"{name} (bin {bin_i})"

        if "beam" in tracer_config.keys():
            if tracer_config["beam"] == "pixwin":
                beam = hp.pixwin(nside)
            #beam_file = data_dir + '/' + config[key]["beam"].format(bin=bin_i, nside=nside)
        else:
            beam = np.ones(3*nside)

        if tracer_type == "healpix":
            map_file = data_dir + '/' + tracer_config["healpix"]["map"].format(bin=bin_i, nside=nside)
            mask_file = data_dir + '/' + tracer_config["healpix"]["mask"].format(bin=bin_i, nside=nside)

            maps = np.atleast_2d(hp.read_map(map_file, field=None))
            if correct_qu_sign and len(maps) == 2:
                maps = np.array([-maps[0], maps[1]])
            mask = hp.read_map(mask_file)
            if use_mask_squared: mask = mask**2
            tracer = MapTracer(bin_name, maps, mask, beam=beam)
            noise_est = tracer_config["healpix"].get("noise_est", 0)
            if not isinstance(noise_est, list):
                noise_est = bins * [noise_est,]
            tracer.noise_est = noise_est[bin_i]

        elif tracer_type == "catalog":
            cat_file = data_dir + '/' + tracer_config["catalog"]["file"].format(bin=bin_i)
            catalog = Table.read(cat_file)
            pos = [get_ul_key(catalog, "ra"), get_ul_key(catalog, "dec")]
            try:
                weights = get_ul_key(catalog, "weight")
            except KeyError:
                weights = np.ones(len(catalog))
            if "fields" in tracer_config["catalog"].keys():
                fields = [catalog[f] for f in tracer_config["catalog"]["fields"]]
                if correct_qu_sign and len(fields) == 2:
                    fields = [-fields[0], fields[1]]
                pos_rand = None
                weights_rand = None
            elif "randoms" in tracer_config["catalog"].keys():
                fields = None
                rand_file = data_dir + '/' + tracer_config["catalog"]["randoms"].format(bin=bin_i)
                rand_cat = Table.read(rand_file)
                pos_rand = [get_ul_key(rand_cat, "ra"), get_ul_key(rand_cat, "dec")]
                try:
                    weights_rand = get_ul_key(rand_cat, "weight")
                except KeyError:
                    weights_rand = np.ones(len(rand_cat))
            else:
                raise ValueError(f"Must specify either fields or randoms in {tracer_key}")
            tracer = CatalogTracer(bin_name, pos, weights, 3*nside-1, fields=fields, beam=beam,
                                   pos_rand=pos_rand, weights_rand=weights_rand)

        print(tracer)
        tracer_bins.append(tracer)
    return tracer_bins


def main():
    parser = argparse.ArgumentParser(description="Run a Nx2-point analysis pipeline")
    parser.add_argument("config_file", help="YAML file specifying pipeline to run")
    parser.add_argument("--nside", help="overrides nside in config file", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true", help="Don't use the workspace cache")
    args = parser.parse_args()
    print(args)

    with open(args.config_file) as f:
        config = yaml.full_load(f)

    print(config)
    nside = config["nside"] if args.nside is None else args.nside
    print("Nside", nside)

    nmt_bins = get_ell_bins(nside, config["binning"])
    ell_eff = nmt_bins.get_effective_ells()

    tracer_keys = [key for key in config["tracers"].keys()]
    print(f"Found {len(tracer_keys)} tracers")
    tracers = dict()
    for tracer_key in tracer_keys:
        tracer_bins = get_tracer(nside, config["tracers"][tracer_key])
        tracers[tracer_key] = tracer_bins

    xspec_keys = [key for key in config.keys() if key.startswith("cross_spectra")]
    print(f"Found {len(xspec_keys)} set(s) of cross-spectra to calculate")
    for xspec_key in xspec_keys:
        if "save_npz" not in config[xspec_key].keys() and "save_sacc" not in config[xspec_key].keys():
            print(f"Warning! No output will be saved for the block {xspec_key}")

    wksp_dir = None if args.no_cache else config["workspace_dir"]
    print("Using workspace cache:", wksp_dir)

    for xspec_key in xspec_keys:
        xspec_list = config[xspec_key]["list"]
        print("Computing set", xspec_list)

        calc_cov = config[xspec_key].get("covariance", False)
        calc_interbin_cov = config[xspec_key].get("interbin_cov", False)
        subtract_noise = config[xspec_key].get("subtract_noise", False)

        # calculate everything
        cls, bpws, covs = compute_cls_cov(tracers, xspec_list, nmt_bins, subtract_noise=subtract_noise,
                                          compute_cov=calc_cov, compute_interbin_cov=calc_interbin_cov,
                                          wksp_cache=wksp_dir)

        data = Data(ell_eff, cls, covs, bpws, tracers=tracers)

        # save all cross-spectra
        if "save_npz" in config[xspec_key].keys():
            save_npz_file = config[xspec_key]["save_npz"].format(nside=nside)
            print("Saving to", save_npz_file)
            data.write_to_npz(save_npz_file)

        # create sacc file
        if "save_sacc" in config[xspec_key].keys():
            save_sacc_file = config[xspec_key]["save_sacc"]["file"]
            print("Saving to", save_sacc_file)
            metadata = config[xspec_key]["save_sacc"].get("metadata", None)
            data.write_to_sacc(save_sacc_file, metadata=metadata)
