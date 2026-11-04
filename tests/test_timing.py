"Rustymath's timing lock, seen from Python."
import os
import pytest, rustymath as rm

@pytest.mark.skipif(os.name == 'nt', reason='Python timing coordination requires Unix flock')
def test_timing_lock_protocol():
    "Rust's timing hold excludes Python readers of the same lock file."
    import fcntl
    from fastws.timelock import LOCK
    with rm.timing(), open(LOCK, 'a') as f, pytest.raises(BlockingIOError): fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
