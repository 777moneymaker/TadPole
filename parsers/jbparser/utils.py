import functools
import inspect
import pickle
import sys
from functools import wraps
from itertools import chain
from multiprocessing import cpu_count
from pathlib import Path
from timeit import default_timer as timer
from typing import Callable, Collection, Dict

import joblib
from tqdm import tqdm

from joblib.externals.loky import set_loky_pickler
from joblib import wrap_non_picklable_objects

# DEFAULTS
default_threads = max(cpu_count() - 1, 1)


# PARALLELIZATION
class Parallel(joblib.Parallel):
    """
    The modification of joblib.Parallel
    with a TQDM proigress bar
    according to Nth
    (https://stackoverflow.com/questions/37804279/how-can-we-use-tqdm-in-a-parallel-execution-with-joblib)
    """

    def __init__(self,
                 parallelized_function: Callable,
                 input_collection: Collection,
                 kwargs: Dict = None,
                 n_jobs=None,
                 backend=None,
                 description: str = None,
                 verbose=0,
                 timeout=None,
                 pre_dispatch='2 * n_jobs',
                 batch_size='auto',
                 temp_folder=None, max_nbytes='1M', mmap_mode='r',
                 prefer=None,
                 require=None,
                 bar: bool = False):
        if not n_jobs:
            n_jobs = default_threads
        joblib.Parallel.__init__(self, n_jobs, backend, verbose, timeout,
                                 pre_dispatch, batch_size, temp_folder,
                                 max_nbytes, mmap_mode, prefer, require)
        kwargs = {} if not kwargs else kwargs

        if bar:
            self._progress = tqdm(total=len(input_collection))
            if description:
                self._progress.set_description(description)
            else:
                self._progress.set_description(parallelized_function.__name__)
        else:
            self._progress = None

        if all([e is None for e in input_collection]):
            print('\nExecutor found empty input_collection (assuming that no input is required)\n', flush=True)
            input_empty = True
        else:
            input_empty = False

        if input_empty:
            self.result = self.__call__((joblib.delayed(parallelized_function)(**kwargs)) for _ in input_collection)
        else:
            self.result = self.__call__((joblib.delayed(parallelized_function)(e, **kwargs)) for e in input_collection)

        if bar:
            self._progress.close()

    def print_progress(self):
        if self._progress:
            self._progress.n = self.n_completed_tasks
            self._progress.refresh()


class BatchParallel(Parallel):

    def __init__(self,
                 parallelized_function: Callable,
                 input_collection: Collection,
                 partition_size: int = None,
                 kwargs: Dict = {},
                 n_jobs=None,
                 backend=None,
                 description: str = None,
                 verbose=0,
                 timeout=None,
                 pre_dispatch='2 * n_jobs',
                 batch_size='auto',
                 temp_folder=None, max_nbytes='1M', mmap_mode='r',
                 prefer=None,
                 require=None):

        if description is None:
            description = parallelized_function.__name__

        if all([e is None for e in input_collection]):
            print('\nExecutor found empty input_collection (assuming that no input is required)\n', flush=True)
            input_empty = True
        else:
            input_empty = False

        def wrapper_function(batch):
            if input_empty:
                return tuple([parallelized_function(**kwargs) for _ in batch])
            else:
                return tuple([parallelized_function(element, **kwargs) for element in batch])

        if partition_size is None:
            partition_size = max(int(len(input_collection) / 3 * default_threads), 1)
        elif partition_size < 1:
            partition_size = int(len(input_collection) * partition_size)
        batches = [input_collection[i * partition_size:(i + 1) * partition_size] for i in
                   range((len(input_collection) + partition_size - 1) // partition_size)]

        Parallel.__init__(self,
                          parallelized_function=wrapper_function,
                          input_collection=batches,
                          n_jobs=n_jobs,
                          backend=backend,
                          verbose=verbose,
                          timeout=timeout,
                          pre_dispatch=pre_dispatch,
                          batch_size=batch_size,
                          temp_folder=temp_folder,
                          max_nbytes=max_nbytes,
                          mmap_mode=mmap_mode,
                          prefer=prefer,
                          require=require,
                          description=description)

        self.result = chain.from_iterable(self.result)


def time_this(func):
    """
    Decorator which returns information about execution of decorated function.
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = timer()
        values = func(*args, **kwargs)
        end = timer()
        runtime = end - start
        if values is None:
            print(f"{func.__name__!r} execution error", flush=True)
        else:
            print(f"{func.__name__!r} executed successfully in {runtime:.6f} seconds", flush=True)
            return values[0]
        return wrapper_timer


def checkpoint(funct: callable):
    """
    Simple serialisation decorator
    that saves function result
    if exacted output file don't exist or is empty
    or read it if it is non-empty
    @param funct: function to wrap
    @param pickle_path: a path to an output file
    @param serialisation_method: a module used for serialiation (either joblib or pickle)
    @return:
    """

    signature = inspect.signature(funct)

    @wraps(funct)
    def save_checkpoint(*args, **kwargs):

        bound_args = signature.bind(*args, **kwargs)
        pickle_path = Path(bound_args.arguments.get('pickle_path',
                                                    signature.parameters['pickle_path'].default))
        print(f'\nrunning {funct.__name__}', flush=True)
        if pickle_path:
            try:
                with pickle_path.open('rb') as file_object:
                    result = pickle.load(file_object)
                print(f'\ntemporary file read from: {pickle_path.as_posix()}\n', flush=True)
                return result
            except (FileNotFoundError, IOError, EOFError):
                sys.setrecursionlimit(5000)
                result = funct(*args, **kwargs)
                with pickle_path.open('wb') as out:
                    pickle.dump(result, out)
                print(f'\ntemporary file stored at: {pickle_path.as_posix()}\n', flush=True)
                return result

    return save_checkpoint
