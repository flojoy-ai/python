# Flojoy - visual python function

## @flojoy

Python client for Flojoy desktop app and `@flojoy` decorator

Install with `pip install flojoy`

### Usage:

```
from scipy import signal
import numpy as np
from flojoy import flojoy, DataContainer

@flojoy
def BUTTER(v, params):
    ''' Apply a butterworth filter to an input vector '''

    print('Butterworth inputs:', v)

    x = v[0].x
    sig = v[0].y
    
    sos = signal.butter(10, 15, 'hp', fs=1000, output='sos')
    filtered = signal.sosfilt(sig, sig)

    return DataContainer(x = x, y = filtered)
```

The `@flojoy` decorator automatically injects vector(s) passed from the previous node, as well as any control parameters set in the CTRL panel UI.

Please see https://github.com/flojoy-io/flojoy-desktop/tree/main/PYTHON/FUNCTIONS for more usage examples.

### Supported data types

See https://github.com/flojoy-io/flojoy-python/issues/4


## Publish Package on PYPI
### Uploading file via *Twine*
You have to install `twine` package first if not installed already.
To install `twine` run following command: 
  `pip install twine`
  
- Update version in [setup.py](setup.py#L5). For example: `version = '0.0.1'` for prod version and `version = '0.0.1-dev'` for dev version release,
- Run following command to make a distribution file:
  `python3 setup.py sdist`
- To upload files on `PYPI`, run following command: 
`twine upload dist/*`

**Note:** You'll need `username` and `password` to make a release. Please get in touch with the team for credentials.