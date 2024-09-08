# nx2pt-pipeline

Nx2pt is designed to be a simple to use wrapper around the LSST DESC tools [NaMASTER](https://github.com/LSSTDESC/NaMaster) and [SACC](https://github.com/LSSTDESC/sacc) to calculate potentially large sets of cross-correlations and their covariance.

## Installation
Installation should be as simple as
```
python -m pip install . [--user] [-e]
```

## Usage
Most uses of nx2pt will simply involve running the script `run_nx2pt` which takes a YAML file that defines all the tracers of interest and specifies which cross-spectra and covariances to calculate.

A very simple example of a `pipeline.yaml` file is given below:
```yaml
nside: 1024

tracer1:

tracer2:

cross_spectra:
  list: [tracer1, tracer2]
```

More pipeline examples are given in `examples`.

After specifying the tracers and various configuration options, all of the cross-spectra and covariances can be calculated by simply running:
```
run_nx2pt pipeline.yaml
```


## Credits
If you use this code in a publication, make sure to cite all the relevant papers that describe NaMASTER and SACC. Thank you to all of the LSST DESC members who have developed and maintain these tools.
