# climate_utils
Python library of custom utilities for climate &amp; atmospheric science. Includes methods for computing common climate indices, preforming data transformations, and plotting vertical/horizontal atmospheric slices with sensible default visual choices

See docstrings at function definitions for arguments and usage

## Installation:

1. Clone the package

2. Locally install the package

```bash
cd climate_utils
pip install -e .
```

The `-e` flag to `pip install` installs the package with a symlink to the soruce code, so that later updates via git pull do not require re-installation.

## Usage:

Import the functions as e.g.

```python
import climate_utils.climate_toolbox as ctb
import climate_utils.climate_artist as cart
import climate_utils.artist_utils as aut
```
