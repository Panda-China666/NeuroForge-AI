"""Fresh process per compute task: CUDA failures cannot poison the API process."""
import os
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
if __name__=='__main__':
    from .worker import execute
    execute(sys.argv[1])
