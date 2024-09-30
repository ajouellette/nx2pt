# nx2pt

Nx2pt is designed to be a simple to use wrapper around the LSST DESC tools [NaMASTER](https://github.com/LSSTDESC/NaMaster) and [SACC](https://github.com/LSSTDESC/sacc) to calculate potentially large sets of cross-correlations and their covariance.

## Installation
Installation should be as simple as
```
git clone https://github.com/ajouellette/nx2pt
cd nx2pt
python -m pip install . [--user] [-e]
```

## Usage
Most uses of Nx2pt will simply involve running the script `run_nx2pt` which takes a YAML file that defines all the tracers of interest and specifies which cross-spectra and covariances to calculate.

A very simple example of a `pipeline.yaml` file is given below:
```yaml
nside: 1024
# binning scheme for cross-spectra
binning:
    kind = "linear"
    delta_ell = 50
# where to save namaster workspaces containg the mode coupling matrices
workspace_dir: "workspaces"

# define first tracer (some galaxy sample, for example)
tracer1:
  name: "Galaxy sample 1"
  data_dir: "galaxies1"
  map: "galaxy1_delta.fits"
  mask: "galaxy1_mask.fits"

# define second tracer (some other galaxy sample)
tracer2:
  name: "Galaxy sample 2"
  data_dir: "galaxies2"
  map: "galaxy2_delta.fits"
  mask: "galaxy2_mask.fits"

cross_spectra:
  # calculate all auto- and cross-spectra
  list:
    - [tracer1, tracer2]
    - [tracer1, tracer1]
    - [tracer2, tracer2]
  # calculate full covariance
  covariance: True
  # save everything to a .npz file
  save_npz: "galaxy_3x2pt_spectra.npz"
```

It is very easy to setup significantly more complicated pipelines with arbitrary number of tracers (each possibly with tomographic bins and/or spin). More pipeline examples are given in `examples`.

After specifying the tracers and various configuration options, all of the cross-spectra and covariances can be calculated by simply running:
```
run_nx2pt pipeline.yaml
```


## Credits
If you use this code in a publication, make sure to cite all the relevant papers that describe NaMASTER and SACC. Thank you to all of the LSST DESC members who have developed and maintain these tools.
