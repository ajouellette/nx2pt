from dataclasses import dataclass, field
import numpy as np
import healpy as hp
import pymaster as nmt


@dataclass
class Tracer:
    """
    Base class (not useable on its own) representing a field defined on the sky.
    """

    name: str
    beam: np.array = field(repr=False, default=None, kw_only=True)
    dndz: tuple[np.array, np.array] = field(repr=False, default=None, kw_only=True)
    spin: int = field(init=False)


@dataclass
class MapTracer(Tracer):
    """
    A class representing a map-based field defined on the sky.

    Required arguments:
    name: str - a name describing the tracer
    maps: list of arrays - maps (1 for spin-0 fields or 2 for spin-2 fields) defining the field values on the sky
    mask: array - sky mask

    Optional arguments:
    beam: array - instrument beam or smoothing that has been applied to the field
    dndz: tuple of arrays - (z, dndz), redshift distribution of the tracer

    Attributes:
    nside: int - the healpix nside parameter for this tracer
    spin: int - spin of this tracer
    field: NmtField - namaster field object for this tracer
    """

    nside: int = field(init=False)
    maps: list[np.array] = field(repr=False)
    mask: np.array = field(repr=False)

    def __post_init__(self):
        for m in self.maps:
            assert len(m) == len(self.mask), "Maps and masks must all be the same size"
        self.nside = hp.npix2nside(len(self.mask))
        if self.beam is None:
            self.beam = np.ones(3*self.nside)
        assert len(self.beam) == 3*self.nside, "Beam is incorrect size for given nside"
        if len(self.maps) == 1:
            self.spin = 0
        elif len(self.maps) == 2:
            self.spin = 2
        else:
            raise ValueError("Only spin-0 or spin-2 supported")
        # namaster field
        self.field = nmt.NmtField(self.mask, self.maps, spin=self.spin, beam=self.beam)


@dataclass
class CatalogTracer(Tracer):
    """
    A class representing a catalog-based field defined on the sky.
    """

    pos: np.array = field(repr=False)
    weights: np.array = field(repr=False)
    fields: list[np.array] = field(repr=False)
    lmax: int
    field_is_weighted: bool = field(repr=False, default=False, kw_only=True)
    lonlat: bool = field(repr=False, default=True, kw_only=True)

    def __post_init__(self):
        assert len(pos) == 2, "angular positions should have shape (2, N)"
        if len(self.fields) == 1:
            self.spin = 0
        elif len(self.fields) == 2:
            self.spin = 2
        else:
            raise ValueError("Only spin-0 or spin-2 supported")
        assert len(weights) == len(pos[0]), "mismatch between shapes of weights and positions arrays"
        assert len(fields[0]) == len(pos[0]), "mismatch between shapes of fields and positions arrays"
        if self.beam is None:
            self.beam = np.ones(self.lmax+1)
        assert len(self.beam) >= self.lmax+1, "beam is incorrect size for given lmax"
        # namaster field
        self.field = nmt.NmtFieldCatalog(pos, weights, fields, lmax, spin=self.spin, beam=self.beam, field_is_weighted=self.field_is_weighted, lonlat=self.lonlat)
