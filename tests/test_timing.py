"Rustymath's timing lock, seen from Python."
import os
import pytest, rustymath as rm

@pytest.mark.skipif(os.name == 'nt', reason='Python timing coordination requires Unix flock')
def test_timing_lock_protocol(tmp_path):
    "Rust's timing hold excludes Python readers of the same lock file. The test points the lock at its own file, so builds holding the machine's lock don't hold it up."
    import fcntl
    lock = str(tmp_path/'timing.lock')
    with rm.timing(lock), open(lock, 'a') as f, pytest.raises(BlockingIOError): fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
