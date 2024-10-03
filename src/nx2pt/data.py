import datetime
import os
import warnings
import numpy as np
import sacc


def bin_theory_cl(theory_cl, bpws):
    """Bin a theory Cl given some bandpower windows."""
    nells = bpws.shape[1]
    if len(theory_cl) < nells:
        raise ValueError("theory Cl has fewer ells than the bandpower windows.")
    wsum = np.sum(bpws, axis=1)
    return np.sum(np.expand_dims(theory_cl[:nells], 0) * bpws, axis=1) / wsum


def get_cl_dtypes(ncls):
    """Return the sacc data types that correspond to a given number of Cls."""
    if ncls == 1:
        return ["cl_00"]
    if ncls == 2:
        return ["cl_0e", "cl_0b"]
    if ncls == 4:
        return ["cl_ee", "cl_eb", "cl_be", "cl_bb"]
    raise ValueError("ncls must be 1, 2, or 4")


class Data:

    def __init__(self, ell_eff, cls, covs, bpws):
        self.ell_eff = ell_eff
        self.cls = cls
        self.covs = covs
        self.bpws = bpws
        self.nbpws = len(ell_eff)

    @property
    def tracers(self):
        tracers = set()
        for key in self.cls.keys():
            cl_tracers = key.split(", ")
            tracers.add(cl_tracers[0])
            tracers.add(cl_tracers[1])
        return sorted(list(tracers))

    @property
    def tracer_pairs(self):
        tracer_pairs = []
        for key in self.cls.keys():
            cl_tracers = key.split(", ")
            tracer_pairs.append(tuple(cl_tracers))
        return tracer_pairs

    @classmethod
    def from_npz(cls, filename):
        cls = dict()
        covs = dict()
        bpws = dict()
        with np.load(filename) as f:
            for key in f.keys():
                if key.startswith("cl_"):
                    cls[key.removeprefix("cl_")] = f[key]
                elif key.startswith("cov_"):
                    covs[key.removeprefix("cov_")] = f[key]
                elif key.startswith("bpw_"):
                    bpws[key.removeprefix("bpw_")] = f[key]
            ell_eff = f["ell_eff"]
        return Data(ell_eff, cls, covs, bpws)

    @classmethod
    def from_sacc(cls, filename):
        pass

    @classmethod
    def from_theory_cls(cls, theory_cls, covs, bpws):
        pass

    def get_cl(self, tracer1, tracer2, dtype=None):
        cl_key1 = f"{tracer1}, {tracer2}"
        cl_key2 = f"{tracer2}, {tracer1}"
        try:
            cl = self.cls[cl_key1]
        except KeyError:
            try:
                cl = self.cls[cl_key2]
            except KeyError:
                raise KeyError(f"could not find Cl for tracers {tracer1} and {tracer2}")
        if dtype is None:
            return cl
        ind = get_cl_dtypes(len(cl)).index(dtype)
        return cl[ind]


    def get_cov(self, cl1, cl2, dtype1=None, dtype2=None):
        cov_key1 = f"{cl1}, {cl2}"
        cov_key2 = f"{cl2}, {cl1}"
        try:
            cov = self.covs[cov_key1]
        except KeyError:
            try:
                # need to transpose if switching order
                cov = self.covs[cov_key2].T
            except KeyError:
                raise KeyError(f"could not find Cov for Cls {cl1} and {cl2}")
        if dtype1 is None and dtype2 is None:
            return cov
        ncls = np.array(cov.shape) // self.nbpws
        ind1 = get_cl_dtypes(ncls[0]).index(dtype1)
        ind2 = get_cl_dtypes(ncls[1]).index(dtype2)
        cov_shape = (self.nbpws, ncls[0], self.nbpws, ncls[1])
        return cov.reshape(cov_shape)[:,ind1,:,ind2]

    def build_full_cov_e(self, cls, scale_cuts=None, fill_off_diag=True):
        covs = []
        if scale_cuts is not None:
            use_ell = (self.ell_eff >= scale_cuts[0]) * (self.ell_eff < scale_cuts[1])
        else:
            use_ell = np.ones(self.nbpws, dtype=bool)
        for i in range(len(cls)):
            covs_i = []
            for j in range(len(cls)):
                if i == j:
                    cov = self.get_cov(cls[i], cls[i])[:,0,:,0]
                else:
                    try:
                        cov = self.get_cov(cls[i], cls[j])[:,0,:,0]
                    except KeyError:
                        cov = np.zeros((self.nbpws, self.nbpws))
                covs_i.append(cov[use_ell][:,use_ell])
            covs.append(covs_i)
        return np.block(covs)

    def write_to_npz(self, filename):
        """Save cross-spectra, covariances, and bandpower windows to a .npz file."""
        save_dict = {"cl_" + cl_key: self.cls[cl_key] for cl_key in self.cls.keys()} | \
                    {"cov_" + cov_key: self.covs[cov_key] for cov_key in self.covs.keys()} | \
                    {"bpw_" + cl_key: self.bpws[cl_key] for cl_key in self.cls.keys()} | \
                    {"ell_eff": self.ell_eff}
        np.savez(filename, **save_dict)

    def write_to_sacc(self, filename, metadata=None, overwrite=True):
        """Create a sacc fits file containing the cross-spectra and covariance."""
        s = sacc.Sacc()
        # metadata
        s.metadata["creation"] = datetime.date.today().isoformat()
        if metadata is not None:
            for key in metadata.keys():
                s.metadata[key] = metadata[key]
        # tracers (currently only save as misc tracers)
        for tracer in self.tracers:
            s.add_tracer("Misc", tracer)
        # data
        for tracer1, tracer2 in self.tracer_pairs:
            cl_key = f"{tracer1}, {tracer2}"
            cl = self.cls[cl_key]
            if self.bpws == {}:
                warnings.warn("Data has no bandpower information")
                bpws = [None for i in range(len(cl))]
            else:
                nmt_bpws = self.bpws[cl_key]
                ell = np.arange(nmt_bpws.shape[-1])
                bpws = [sacc.BandpowerWindow(ell, nmt_bpws[i,:,i,:].T) for i in range(len(cl))]
            # possible spin combinations
            if len(cl) == 1:
                s.add_ell_cl("cl_00", tracer1, tracer2, self.ell_eff, cl[0], window=bpws[0])
            elif len(cl) == 2:
                for i, dtype in enumerate(["cl_0e", "cl_0b"]):
                    s.add_ell_cl(dtype, tracer1, tracer2, self.ell_eff, cl[i], window=bpws[i])
            elif len(cl) == 4:
                for i, dtype in enumerate(["cl_ee", "cl_eb", "cl_be", "cl_bb"]):
                    s.add_ell_cl(dtype, tracer1, tracer2, self.ell_eff, cl[i], window=bpws[i])
            else:
                raise ValueError("number of Cls should be 1, 2, or 4")
        # covariance
        if self.covs == {}:
            warnings.warn("Data has no covariance information")
        else:
            if len(self.cls.keys()) == len(self.covs.keys()):
                # block diagonal covariance
                s.add_covariance([self.get_cov(cl_key, cl_key, reshape_cov=False) for cl_keys in self.cls.keys()])
            else:
                # loop over all possible cross-spectra
                full_cov = np.zeros((len(s.mean), len(s.mean)))
                for tracers1 in s.get_tracer_combinations():
                    for dtype1 in s.get_data_types(tracers1):
                        inds1 = s.indices(tracers=tracers1, data_type=dtype1)
                        for tracers2 in s.get_tracer_combinations():
                            for dtype2 in s.get_data_types(tracers2):
                                inds2 = s.indices(tracers=tracers2, data_type=dtype2)
                                cov = self.get_cov(", ".join(tracers1), ", ".join(tracers2), dtype1=dtype1, dtype2=dtype2)
                                full_cov[np.ix_(inds1, inds2)] = cov
                s.add_covariance(full_cov)
        # write
        s.save_fits(filename, overwrite=overwrite)
