# bittorchinfo

This project is an extended version of [torchinfo](https://github.com/tyleryep/torchinfo).
However we added support for binary layers, as used in [bitorch](https://github.com/hpi-xnor/bitorch).
Please check out and use the original project, unless you specifically want to use this package in conjunction with [bitorch](https://github.com/hpi-xnor/bitorch).

# Usage

```
pip install bitorchinfo
```

# Contributing

All issues and pull requests are much appreciated! If you are wondering how to build the project:

- torchinfo is actively developed using the lastest version of Python.
  - Changes should be backward compatible with Python 3.6, but this is subject to change in the future.
  - Run `pip install -r requirements-dev.txt`. We use the latest versions of all dev packages.
  - Run `pre-commit install`.
  - To use auto-formatting tools, use `pre-commit run -a`.
  - To run unit tests, run `pytest`.
  - To update the expected output files, run `pytest --overwrite`.
  - To skip output file tests, use `pytest --no-output`

# References

- Thanks to @tyleryep for providing the [torchinfo](https://github.com/tyleryep/torchinfo) package.
