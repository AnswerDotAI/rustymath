import pytest, rustymath as rm

@pytest.fixture(params=[1, 4])
def threads(request):
    "Run a test on one thread and on four, then restore the thread setting"
    previous = rm.set_threads(request.param)
    yield request.param
    rm.set_threads(previous)
