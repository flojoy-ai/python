# @flojoy

Python client for Flojoy desktop app and `@flojoy` decorator

Install with `pip install flojoy`

## Usage:

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

## Supported types

TODO

