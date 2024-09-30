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
from .namaster_tools import get_bpw_edges, get_nmtbins
from .namaster_tools import compute_cls_cov
from .utils import get_ul_key, parse_cl_key, parse_tracer_bin


def get_ell_bins(nside, config):
    """Generate ell bins from config."""
    bpw_edges = config.get("bpw_edges", None)
    if bpw_edges is None:
        kind = config.get("kind", "linear")
        lmin = config.get("ell_min", 2)
        lmax = config.get("ell_max", 3*nside-1)
        nbpws = config.get("nbpws", None)
        if nbpws is None and kind == "linear":
            delta_ell = config["delta_ell"]
            nbpws = (lmax - lmin) // delta_ell
        elif nbpws is None:
            raise ValueError("Must specify nbpws for non-linear binning")
        bpw_edges = get_bpw_edges(lmin, lmax, nbpws, kind)
    nmt_bin = get_nmtbins(nside, bpw_edges)
    return nmt_bin


def get_tracer(nside, config, key):
    """Load tracer information."""
    name = config[key]["name"]
    data_dir = config[key]["data_dir"]
    if "healpix" in config[key].keys():
        tracer_type = "healpix"
    elif "catalog" in config[key].keys():
        tracer_type = "catalog"
    else:
        raise ValueError(f"Tracer {key} must have either a 'healpix' or 'catalog' section")
    bins = config[key].get("bins", 1)
    use_mask_squared = config[key].get("use_mask_squared", False)
    correct_qu_sign = config[key].get("correct_qu_sign", False)

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
            noise_est = config[key]["healpix"].get("noise_est", 0)
            if not isinstance(noise_est, list):
                noise_est = [noise_est,]
            tracer.noise_est = noise_est[bin_i]

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
    parser.add_argument("--nside", help="overrides nside in config file", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true", help="Don't use the workspace cache")
    args = parser.parse_args()
    print(args)

    with open(args.config_file) as f:
        config = yaml.full_load(f)

    print(config)
    nside = config["nside"]
    if args.nside is not None: nside = args.nside
    nmt_bins = get_ell_bins(nside, config["binning"])
    ell_eff = nmt_bins.get_effective_ells()

    tracer_keys = [key for key in config.keys() if key.startswith("tracer")]
    print(f"Found {len(tracer_keys)} tracers")
    tracers = dict()
    for tracer_key in tracer_keys:
        tracer = get_tracer(nside, config, tracer_key)
        tracers[tracer_key] = tracer

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

        # save all cross-spectra
        if "save_npz" in config[xspec_key].keys():
            save_npz_file = config[xspec_key]["save_npz"].format(nside=config["nside"])
            print("Saving to", save_npz_file)
            save_npz(save_npz_file, ell_eff, cls, covs, bpws)

        # create sacc file
        #if "save_sacc" in config.keys():
            #print("Creating sacc file")
