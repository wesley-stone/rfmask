import multiprocessing as mp
from queue import Queue

def foo(queue):
    pass

pool=mp.Pool()
q=Queue()

pool.map(foo,(q,))