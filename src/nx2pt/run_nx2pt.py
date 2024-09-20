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

from .tracer import MapTracer, CatalogTracer
from .namaster_tools import parse_cl_key, parse_tracer_bin, compute_cls_cov


def get_ell_bins(config):
    """Generate ell bins from config."""
    nside = config["nside"]
    ell_min = config["ell_min"]
    dl = config["delta_ell"]
    ell_bins = np.linspace(ell_min, 3*nside, int((3*nside - ell_min) / dl) + 1, dtype=int)
    return ell_bins


def get_ul_key(dict_like, key):
    """Get a value using a case-insensitive key."""
    key_list = list(dict_like.keys())
    key_list_lower = [k.lower() for k in key_list]
    if key.lower() not in key_list_lower:
        raise KeyError(f"could not find {key} in {dict_like}")
    ind = key_list_lower.index(key.lower())
    return dict_like[key_list[ind]]


def get_tracer(config, key):
    """Load tracer information."""
    nside = config["nside"]
    name = config[key]["name"]
    data_dir = config[key]["data_dir"]
    if "healpix" in config[key].keys():
        tracer_type = "healpix"
    elif "catalog" in config[key].keys():
        tracer_type = "catalog"
    else:
        raise ValueError(f"Tracer {key} must have either a 'healpix' or 'catalog' section")
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
    for bin_i in range(bins):
        bin_name = name if bins == 1 else f"{name} (bin {bin_i})"

        if "beam" in config[key].keys():
            if config[key]["beam"] == "pixwin":
                beam = hp.pixwin(nside)
            beam_file = data_dir + '/' + config[key]["beam"].format(bin=bin_i, nside=nside)
        else:
            beam = np.ones(3*nside)

        if tracer_type == "healpix":
            map_file = data_dir + '/' + config[key]["healpix"]["map"].format(bin=bin_i, nside=nside)
            mask_file = data_dir + '/' + config[key]["healpix"]["mask"].format(bin=bin_i, nside=nside)

            maps = np.atleast_2d(hp.read_map(map_file, field=None)).astype(float)
            if correct_qu_sign and len(maps) == 2:
                maps = np.array([-maps[0], maps[1]])
            mask = hp.read_map(mask_file).astype(float)
            if use_mask_squared: mask = mask**2
            tracer = MapTracer(bin_name, maps, mask, beam=beam)

        elif tracer_type == "catalog":
            cat_file = data_dir + '/' + config[key]["catalog"]["file"].format(bin=bin_i)
            catalog = Table.read(cat_file)
            pos = [get_ul_key(catalog, "ra"), get_ul_key(catalog, "dec")]
            try:
                weights = get_ul_key(catalog, "weight")
            except KeyError:
                weights = np.ones(len(catalog))
            if "fields" in config[key]["catalog"].keys():
                fields = [catalog[f] for f in config[key]["catalog"]["fields"]]
                if correct_qu_sign and len(fields) == 2:
                    fields = [-fields[0], fields[1]]
                pos_rand = None
                weights_rand = None
            elif "randoms" in config[key]["catalog"].keys():
                fields = None
                rand_file = data_dir + '/' + config[key]["catalog"]["randoms"].format(bin=bin_i)
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


def save_sacc(file_name, tracers, ell_eff, cls, covs, bpws, ignore_b_modes=True, metadata=None):
    s = sacc.Sacc()
    # metadata
    s.metadata["creation"] = datetime.date.today().isoformat()
    if metadata is not None:
        for key in metadata:
            s.metadata[key] = metadata[key]
    # tracers (currently only save as misc tracers)
    for tracer_key in tracers.keys():
        for i in range(len(tracers[tracer_key])):
            sacc_name = tracer_key.rstrip("tracer_") + f"_{i}"
            s.add_tracer("Misc", sacc_name)
    # data
    for cl_key in cls.keys():
        (tracer1, bin1), (tracer2, bin2) = parse_cl_key(cl_key)
        sacc_name1 = tracer1.rstrip("tracer_") + f"_{bin1}"
        sacc_name2 = tracer2.rstrip("tracer_") + f"_{bin2}"
        bpw = bpws[cl_key]
        # possible spin combinations
        if tracers[tracer1][bin1].spin == 0 and tracers[tracer2][bin2].spin == 0:
            s.add_ell_cl("cl_00", sacc_name1, sacc_name2, ell_eff, cls[cl_key][0])
        elif tracers[tracer1][bin1].spin == 2 and tracers[tracer2][bin2].spin == 0:
            s.add_ell_cl("cl_e0", sacc_name1, sacc_name2, ell_eff, cls[cl_key][0])

    # covariance

    # write



def save_npz(file_name, ell_eff, cls, covs, bpws):
    """Save cross-spectra, covariances, and bandpower windows to a .npz file."""
    assert bpws.keys() == cls.keys(), "Each cross-spectrum should have a corresponding bandpower window"
    save_dict = {"cl_" + str(cl_key): cls[cl_key] for cl_key in cls.keys()} | \
                {"cov_" + str(cov_key): covs[cov_key] for cov_key in covs.keys()} | \
                {"bpw_" + str(cl_key): bpws[cl_key] for cl_key in cls.keys()} | \
                {"ell_eff": ell_eff}
    np.savez(file_name, **save_dict)


def main():
    parser = argparse.ArgumentParser(description="Run a Nx2-point analysis pipeline")
    parser.add_argument("config_file", help="YAML file specifying pipeline to run")
    parser.add_argument("--no-cache", action="store_true", help="Don't use the workspace cache")
    args = parser.parse_args()
    print(args)

    with open(args.config_file) as f:
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
    if args.no_cache:
        wksp_dir = None
    else:
        wksp_dir = config["workspace_dir"]
    print("Using workspace cache:", wksp_dir)

    for xspec_key in xspec_keys:
        xspec_list = config[xspec_key]["list"]
        print("Computing set", xspec_list)

        calc_cov = False
        calc_interbin_cov = False
        if "covariance" in config[xspec_key].keys():
            calc_cov = config[xspec_key]["covariance"]
        if "interbin_cov" in config[xspec_key].keys():
            calc_interbin_cov = config[xspec_key]["interbin_cov"]

        # calculate everything
        cls, bpws, covs = compute_cls_cov(tracers, xspec_list, nmt_bins, compute_cov=calc_cov,
                                          compute_interbin_cov=calc_interbin_cov, wksp_cache=wksp_dir)

        # save all cross-spectra
        if "save_npz" in config[xspec_key].keys():
            save_npz_file = config[xspec_key]["save_npz"].format(nside=config["nside"])
            print("Saving to", save_npz_file)
            save_npz(save_npz_file, ell_eff, cls, covs, bpws)

        # create sacc file
        #if "save_sacc" in config.keys():
            #print("Creating sacc file")
