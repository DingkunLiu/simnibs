from concurrent.futures import ThreadPoolExecutor
import functools
import multiprocessing as mp
from typing import Callable


def run_in_new_thread(fn):
    """Decorator that runs `fn` in a new thread."""

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(fn, *args, **kwargs).result()

    return wrapped_fn


@run_in_new_thread
def run_in_multiprocessing_pool(
    n_workers: int,
    fn: Callable,
    iterable,
    pool_kwargs: dict | None = None,
    start_method: str | None = None,
):
    """Submit an iterable to a pool of workers.  `fn` is executed as

            fn(iterable[0]), fn(iterable[1]), ...

    and the result returned as a list.

    Parameters
    ----------
    n_workers : int
        Number of workers.
    fn : Callable
        The function to call.
    iterable : _type_
        Iterable of arguments to `fn`.

    Returns
    -------
    result
        The concatenated result from running `fn`.
    """
    start_method = mp.get_start_method() if start_method is None else start_method
    pool_kwargs = pool_kwargs or {}
    with mp.get_context(start_method).Pool(processes=n_workers, **pool_kwargs) as pool:
        result = pool.starmap_async(fn, iterable)
        pool.close()
        pool.join()
    return result.get()
