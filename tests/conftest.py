import os
import tempfile
import pytest
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

# Never touch the user's runtime or experiments while running tests.
_runtime=tempfile.TemporaryDirectory(prefix='neuroforge-tests-')
os.environ['NEUROFORGE_INLINE_WORKER']='1'
os.environ['NEUROFORGE_DATA']=_runtime.name


@pytest.fixture(scope='session',autouse=True)
def cleanup_worker():
    yield
    from neuroforge import worker
    worker.POOL.shutdown(wait=True)
    _runtime.cleanup()
