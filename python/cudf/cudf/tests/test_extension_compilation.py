import operator
import pytest

from numba import types
from numba.cuda import compile_ptx


arith_ops = (
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
)

number_types = (
    types.float32,
    types.float64,
    types.int8,
    types.int16,
    types.int32,
    types.int64,
    types.uint8,
    types.uint16,
    types.uint32,
    types.uint64,
    types.complex64,
    types.complex128,
)


@pytest.mark.parametrize('op', arith_ops)
@pytest.mark.parametrize('ty', number_types, ids=[str(t) for t in number_types])
@pytest.mark.parametrize('constant', [1, 1.5])
def test_arith_masked_vs_constant(op, ty, constant):

    def func(x):
        return op(x, constant)

    cc = (7, 5)
    ptx, resty = compile_ptx(func, (ty,), cc=cc)
