# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from .cudf_cpp cimport *
from .cudf_cpp import *
from cudf.bindings.json cimport *
from libc.stdlib cimport free
from libcpp.vector cimport vector

from cudf.dataframe.column import Column
from cudf.dataframe.numerical import NumericalColumn
from cudf.dataframe.dataframe import DataFrame
from cudf.dataframe.datetime import DatetimeColumn
from cudf.utils import ioutils
from cudf.bindings.nvtx import nvtx_range_push, nvtx_range_pop
from librmm_cffi import librmm as rmm

import nvstrings
import numpy as np
import collections.abc
import os

cpdef cpp_read_json(path_or_buf, lines, dtype, compression):
    """
    Cython function to call into libcudf API, see `read_json`.
    See Also
    --------
    cudf.io.json.read_json
    """

    # Setup arguments
    cdef json_read_arg args = json_read_arg()

    #if not os.path.isfile(path) and not os.path.exists(path):
    #    raise FileNotFoundError(
    #        errno.ENOENT, os.strerror(errno.ENOENT), path
    #    )
    path = str(path_or_buf).encode()
    source_ptr = <char*>path
    args.source_type = HOST_BUFFER
    args.source = source_ptr
    args.buffer_size = len(path)

    args.lines = lines
    if compression is None:
        compression_bytes = <char*>NULL
    else:
        compression = compression.encode()
        compression_bytes = <char*>compression

    # Call read_json
    with nogil:
        result = read_json(&args)

    check_gdf_error(result)

    out = args.data
    if out is NULL:
        raise ValueError("Failed to parse Json")

    # Extract parsed columns
    outcols = []
    new_names = []
    for i in range(args.num_cols_out):
        data_mem, mask_mem = gdf_column_to_column_mem(out[i])
        outcols.append(Column.from_mem_views(data_mem, mask_mem))
        new_names.append(out[i].col_name.decode())
        free(out[i].col_name)
        free(out[i])

    # Construct dataframe from columns
    df = DataFrame()
    for k, v in zip(new_names, outcols):
        df[k] = v

    # Set column to use as row indexes if available
    if args.index_col is not NULL:
        df = df.set_index(df.columns[args.index_col[0]])
        free(args.index_col)

    return df
