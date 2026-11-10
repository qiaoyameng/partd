# Reproducible offline environment

Upstream: https://github.com/dask/partd
Upstream SHA: e832b655606342dc742ec1c564b07abf1ad58383
Runtime: 3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)]
Platform: win32

Create an isolated virtual environment, then run from this checkout:
```
python -m pip install -r requirements-domain.lock
python -m pip install -e .
```
Set `PYTHONUTF8=1` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` before tests.
```
python -m pytest -q -o addopts= -o log_cli=false partd/tests/test_file.py partd/tests/test_encode.py partd/tests/test_buffer.py partd/tests/test_dict.py
```
Measured selected pytest process: 0.628 seconds.
Only the listed core tests are claimed. Optional integrations are not required.

Tests use synthetic in-memory or temporary-file inputs under the upstream license.
No account, cloud service, browser or external database is needed. Recreate the
environment from this immutable commit to reset the starting state.
