import os
import numpy as np
import sacc


class Data:

    def __init__(self, ell_eff, cls, covs, bpws):
        self.ell_eff = ell_eff
        self.cls = cls
        self.covs = covs
        self.bpws = bpws
        self.nbpws = len(ell_eff)

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

    def get_tracers(self):
        tracers = set()
        for key in self.cls.keys():
            cl_tracers = key.split(", ")
            tracers.add(cl_tracers[0])
            tracers.add(cl_tracers[1])
        return sorted(list(tracers))

    def get_cl(self, tracer1, tracer2):
        cl_key1 = f"{tracer1}, {tracer2}"
        cl_key2 = f"{tracer2}, {tracer1}"
        try:
            return self.cls[cl_key1]
        except KeyError:
            try:
                return self.cls[cl_key2]
            except KeyError:
                raise KeyError(f"could not find Cl for tracers {tracer1} and {tracer2}")

    def get_cov(self, cl1, cl2):
        cov_key1 = f"{cl1}, {cl2}"
        cov_key2 = f"{cl2}, {cl1}"
        try:
            cov = self.covs[cov_key1]
        except KeyError:
            try:
                cov = self.covs[cov_key2]
            except KeyError:
                raise KeyError(f"could not find Cov for Cls {cl1} and {cl2}")
        ncls = len(cov) // self.nbpws
        cov_shape = (self.nbpws, ncls, self.nbpws, ncls)
        return cov.reshape(cov_shape)

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

    def write_to_sacc(self, filename):
        pass

    def write_to_npz(self, filename):
        pass

