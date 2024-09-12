from dataclasses import dataclass, field
import numpy as np
import healpy as hp
import pymaster as nmt
from astropy.table import Table


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

    catalog: Table = field(repr=False)

    def __post_init__(self):
        pass
