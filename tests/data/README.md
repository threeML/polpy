# Real test data

`POLAR_160325A.pevt` / `POLAR_160325A.prsp` are real POLAR event/response
files for GRB160325A, used by `tests/test_real_data_joint_fit.py` to check
that data loading, `DataList` assembly, and joint likelihood evaluation
still work against real data (not just the synthetic `conftest.py`
fixtures).

## `NCHANS`/`NSABINS` patch

`PolData.__init__` (`polpy/poldata.py`) reads `self.n_channels` and
`self.n_scattering_bins` from the `NCHANS`/`NSABINS` header keywords on the
`POLEVENTS` HDU. The original POLAR file does not carry these keywords (only
`TIME`/`CHANNEL`/`SABIN`/`DEADFRAC` columns), so loading it raises a
`KeyError`. The copy checked in here has been patched in place to add
`NCHANS = 150` and `NSABINS = 360` (matching the energy/scattering-angle
binning of the paired `.prsp` response file) so `PolData` can load it.

This is a real interoperability bug, not a quirk of this specific file —
`PolData` should not hard-require header keywords that real POLAR data
doesn't carry. It's tracked as a known issue rather than fixed here; the
patched header is a workaround so the real-data test suite has something to
run against in the meantime.
