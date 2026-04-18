#!/usr/bin/env python
# Copyright (C) 2017, Carlo de Franchis <carlo.de-franchis@polytechnique.org>

import os
import sys
import logging
import multiprocessing
import multiprocessing.context

from s2p import common
from s2p.gpu_memory_manager import GPUMemoryManager

logger = logging.getLogger(__name__)


def show_progress(a):
    """
    Print the number of tiles that have been processed.

    Args:
        a: useless argument, but since this function is used as a callback by
            apply_async, it has to take one argument.
    """
    show_progress.counter += 1
    status = "done {:{fill}{width}} / {} tiles".format(show_progress.counter,
                                                       show_progress.total,
                                                       fill='',
                                                       width=len(str(show_progress.total)))
    if show_progress.counter < show_progress.total:
        status += chr(8) * len(status)
    else:
        status += '\n'
    sys.stdout.write(status)
    sys.stdout.flush()


# this is biggest hack ever, because python's multiprocessing.Value cannot be passed to Pool.apply_async
# so we use the initializer/initargs mechanism, and patch the arguments in tilewise_wrapper
substituted_args = []
INIT_ARG_SENTINEL = 'INIT_ARG_SENTINEL'


def expand_initargs(*initargs):
    global substituted_args
    substituted_args = initargs


def remap_extra_args(extra_args):
    out_args = []
    init_args = []
    for a in extra_args:
        if isinstance(a, GPUMemoryManager):
            out_args.append(INIT_ARG_SENTINEL)
            init_args.append(a)
        else:
            out_args.append(a)
    return tuple(out_args), init_args


def undo_remap_extra_args(extra_args):
    out_args = []
    subargs = iter(substituted_args)
    for a in extra_args:
        if a == INIT_ARG_SENTINEL:
            out_args.append(next(subargs))
        else:
            out_args.append(a)
    return out_args


def tilewise_wrapper(cfg, fun, *args, stdout: str, tile_label: str, **kwargs):
    args = undo_remap_extra_args(args)

    root = logging.getLogger()
    prevhandlers = list(root.handlers)
    prevfilters = list(root.filters)
    for h in prevhandlers:
        root.removeHandler(h)
        h.close()
    for f in prevfilters:
        root.removeFilter(f)
        f.close()

    root.setLevel(logging.INFO)
    f = logging.Formatter('%(asctime)s %(name)s.%(funcName)s %(levelname)-8s %(message)s')
    h = logging.FileHandler(stdout)
    h.setFormatter(f)
    root.addHandler(h)

    if cfg['debug']:
        # if debug is true, then redirect everything to the stderr
        h = logging.StreamHandler(sys.stderr)
        f = logging.Formatter(f'{tile_label} | %(name)s.%(funcName)s | %(message)s')
        h.setFormatter(f)
        root.addHandler(h)
    else:
        # if debug is false, then redirect only the warning and errors messages
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(logging.ERROR)
        f = logging.Formatter(f'{tile_label} | %(name)s.%(funcName)s | %(message)s')
        h.setFormatter(f)
        root.addHandler(h)

    try:
        out = fun(*args)
    except Exception:
        logging.exception("Exception in %s" % fun.__name__)
        raise
    finally:
        # restore the previous loggers
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()
        for f in list(root.filters):
            root.removeFilter(f)
        for h in prevhandlers:
            root.addHandler(h)
        for f in prevfilters:
            root.addFilter(f)

    return out


def get_mp_context() -> multiprocessing.context.BaseContext:
        # use a `spawn` strategy, because 'fork' is unsafe when threads are involved
        # and forkserver is only available on linux (and might also be unsafe anyway)
    return multiprocessing.get_context("spawn")


def launch_calls(cfg, fun, list_of_args, nb_workers, *extra_args, tilewise=True,
                 timeout=600):
    """
    Run a function several times in parallel with different given inputs.

    Args:
        fun: function to be called several times in parallel.
        list_of_args: list of (first positional) arguments passed to fun, one
            per call
        nb_workers: number of calls run simultaneously
        extra_args (optional): tuple containing extra arguments to be passed to
            fun (same value for all calls)
        tilewise (bool): whether the calls are run tilewise or not
        timeout (int): timeout for each function call (in seconds)

    Return:
        list of outputs
    """
    results = []
    outputs = []
    show_progress.counter = 0
    show_progress.total = len(list_of_args)

    extra_args, init_args = remap_extra_args(extra_args)
    if init_args:
        # the init_args hack only work when calling tilewise_wrapper
        assert tilewise

    def tile_label_from_dir(tile_dir: str) -> str:
        """convert:
               /path/to/output_s2p/tiles/row_0002145_height_715/col_0000000_width_667
           to:
               row_0002145_height_715/col_0000000_width_667
        """
        root = os.path.dirname(os.path.dirname(tile_dir))
        return tile_dir.replace(root, '')

    if nb_workers != 1:
        pool = get_mp_context().Pool(nb_workers, initializer=expand_initargs, initargs=init_args)

        for x in list_of_args:
            args = tuple()
            if type(x) == tuple:
                args += x
            else:
                args += (x,)
            args += extra_args
            if tilewise:
                if type(x) == tuple:
                    # we expect x = (cfg, tile_dictionary, ?)
                    tile_dir = x[1].dir
                    tile_label = tile_label_from_dir(tile_dir)
                    if len(x) == 3:  # we expect x = (cfg, tile_dictionary, pair_id)
                        tile_dir = os.path.join(tile_dir, 'pair_%d' % x[2])
                        tile_label = os.path.join(tile_label, 'pair_%d' % x[2])
                else:  # we expect x = tile_dictionary
                    tile_dir = x.dir
                    tile_label = tile_label_from_dir(tile_dir)

                log = os.path.join(tile_dir, 'stdout.log')
                args = (cfg, fun,) + args
                results.append(pool.apply_async(tilewise_wrapper, args=args,
                                                kwds={'stdout': log, 'tile_label': tile_label},
                                                callback=show_progress))
            else:
                results.append(pool.apply_async(fun, args=args, callback=show_progress))

        for r in results:
            o = r.get(timeout)
            outputs.append(o)

        pool.close()
        pool.join()

    else:
        # In single-process mode, manually call the initializer that
        # multiprocessing.Pool would normally call in each worker.
        if init_args:
            expand_initargs(*init_args)

        outputs = []
        for x in list_of_args:
            args = tuple()
            if type(x) == tuple:
                args += x
            else:
                args += (x,)
            args += extra_args
            if tilewise:
                if type(x) == tuple:
                    # we expect x = (cfg, tile_dictionary, ?)
                    tile_dir = x[1].dir
                    tile_label = tile_label_from_dir(tile_dir)
                    if len(x) == 3:  # we expect x = (cfg, tile_dictionary, pair_id)
                        tile_dir = os.path.join(tile_dir, 'pair_%d' % x[2])
                        tile_label = os.path.join(tile_label, 'pair_%d' % x[2])
                else:  # we expect x = tile_dictionary
                    tile_dir = x.dir
                    tile_label = tile_label_from_dir(tile_dir)

                log = os.path.join(tile_dir, 'stdout.log')
                args = (cfg, fun,) + args
                outputs.append(tilewise_wrapper(*args, stdout=log, tile_label=tile_label))
            else:
                outputs.append(fun(*args))


    common.print_elapsed_time()
    return outputs
