# String UDF memory management

Some functions used on strings inside UDFs create new strings; these include
functions like ``concat()``, ``replace()``, and various others. Creating new
strings requires memory allocations, and the semantics of these allocations must
match the semantics of Python allocations. Specifically, memory management
(allocation and freeing of the underlying data) is handled transparently for the
user, via a reference counting mechanism.


## Numba memory management and the Numba Runtime (NRT)

cuDF does not implement its own model for memory management in UDFs. It uses
[Numba's memory management for nopython mode
code](https://numba.readthedocs.io/en/stable/developer/numba-runtime.html#memory-management).
Memory management is implemented through a combination of functions provided via
a runtime library (the Numba Runtime, or *NRT*), and the emission of calls to
NRT functions during code generation. These calls are emitted during:

- **The creation of a new object**: During object creation, memory is allocated
  and a structure to track the memory is created and initialized.
- **When new references are created:** For example during assignments, the
  reference count of the assigned object is incremented.
- **When references are destroyed:** For example when an object goes out of
  scope, or when an object holding a reference is destroyed. During these
  events, the reference count of the tracked object is decremented. If the
  reference count of an object falls to zero, its destructor will be invoked via
  the Numba Runtime.

The cuDF implementation of string functions is required to implement the
appropriate support for the cases above, which it provides through its lowering,
its CUDA C++ library code, and ts Numba shim functions.

This document does not provide a detailed description of Numba memory
management, but only touches on the parts needed to understand the cuDF string
function implementations.


## Preliminaries

## Data structures

### cuDF string data structures

On the C++ side, strings are represented using the ``cudf::string_view`` class,
and the ``cudf::strings::udf::udf_string`` class. The ``string_view`` class does
not own the underlying string data; the ``udf_string`` does own the underlying
string data.

The cuDF extensions to Numba generate code to manipulate instances of these
classes, so we outline the members of these classes to aid in understanding
them. These classes also have various methods; consult the cuDF C++ Developer
Documentation for further details of these structures.

```c++
class string_view {
    // A pointer to the underlying string data
    char const* p{};
    // The length of the underlying string data in bytes
    size_type bytes{};
    // The offset into the underlying string data in characters
    size_type char_pos{};
    // The offset into the underlying string data in bytes
    size_type byte_pos{};
};
```

```c++
class udf_string {
  // A pointer to the underlying string data
  char* m_data{};
  // The length of the string data in bytes
  cudf::size_type m_bytes{};
  // The size of the underlying allocation in bytes
  cudf::size_type m_capacity{};
};
```

```{note}
A ``udf_string`` has a destructor that frees the underlying string data. This is
important, because the C++ destructor is invoked during destruction of a
Python-side Managed UDF String object.
```

### Numba data structures

The Numba ``NRT_MemInfo`` structure is used to manage the lifetime of
allocations. It is represented [in Numba-CUDA's CUDA C++
code](https://github.com/NVIDIA/numba-cuda/blob/c015cabd114b6b3e7b2701a484ce7daa909d7f87/numba_cuda/numba/cuda/memory_management/nrt.cuh#L6-L14)
as:

```C++
struct NRT_MemInfo {
  // The reference count. Initialized to 1 for a new object
  cuda::atomic<size_t, cuda::thread_scope_device> refct;
  // The destructor: a function that is called when the reference count drops to
  // zero.
  NRT_dtor_function dtor;
  // A pointer to any data the destructor requires; may be null if none is
  needed.
  void* dtor_info;
  // A pointer to the allocation managed by this object.
  void* data;
  // The size of the allocation. This may be unspecified (zero) if the
  destructor functions does not need to know the size of the allocation.
  size_t size;
};
```

The ``NRT_dtor_function`` has the following prototype:

```C++
void (*NRT_dtor_function)(void* ptr, size_t size, void* info);
```

### cuDF String UDF data structures

These data structures are part of the Python cuDF code and are only used to
support UDFs operating on strings.

A ``managed_udf_string`` combines an ``NRT_MemInfo`` and a ``udf_string``,
providing an entity that can represent a string using cuDF C++ code, with its
lifetime managed by Numba-generated and Numba extension code in cuDF.

```C++
struct managed_udf_string {
  // Points to the NRT_MemInfo object managing the UDF string
  // XXX: A void pointer looks odd here. Can this be an `NRT_MemInfo*`?
  void* meminfo;
  // The UDF string being managed.
  udf_string udf_str;
};
```

There is also a "special case" of a Managed UDF String, an
``mi_str_allocation``, that stores the ``NRT_MemInfo`` inside itself instead of
pointing to an ``NRT_MemInfo`` elsewhere in memory. This is only
stack-allocated during the creation of a new Managed UDF String.

```c++
struct mi_str_allocation {
  NRT_MemInfo mi;
  udf_string st;
};
```

### NRT functions

The NRT functions that the cuDF implementation uses are:

```CUDA
// Allocates `size` bytes on the heap using on-device malloc, and returns a
// pointer to the allocated data.
//
// If memory system stats are enabled, tracked stats are updated.
__device__ void* NRT_Allocate(size_t size);
```

```CUDA
// Set the fields of a new NRT_MemInfo, `mi`. The reference count is initialized
// to 1. Other fields are initialized as per the other function arguments.
__device__ void NRT_MemInfo_init(
  NRT_MemInfo* mi,
  void* data,
  size_t size,
  NRT_dtor_function dtor,
  void* dtor_info
);
```

## Implementation

The cuDF implementations for Managed UDF Strings is required to provide:

- Typing and lowering for Managed UDF String operations.
  - The typing has no special properties; it is similar to any other typing
    implementation in a Numba extension.
  - The lowering is required to ensure that ``NRT_MemInfo`` objects for each
    managed object are created and initialized correctly.
- C++ implementations of string functions:
  - Some of these are provided by cuDF / libcudf's C++ string functionality
  - Other functions are provided by the ``strings_udf`` C++ library in cuDF
    Python. These help with the allocation of data and implement the required
    destructors.
  - Numba shim functions to adapt calls to C++ code for use in Numba code and
    Numba extensions are also required.
- Conversion from String UDF data back into cudf C++ column format.

Use of C++ code for string functionality is not a hard requirement for
implementing string support in a Numba extension - it is instead a pragmatic
requirement that the Python and C++ sides of cuDF share a single implementation
for string operations, instead of trying to keep two separately-maintained
implementations in sync.

cuDF generally does not need to provide any implementation to support reference
counting operations during the lifetime of a managed object after its
construction. Because the standard ``NRT_MemInfo`` data structure is used in
conjuction with Numba's built-in lowerings and NRT library, it automatically
emits code to manage reference counts during "normal" Python code execution
including assignments, scoping, function calls, etc. Another way to view this is
that the cuDF code is really providing the implementation and details at the
lifetime boundaries, the beginning and ending, of Managed UDF Strings.

The majority of the complexity in the implementation comes from two areas:

- Combining the requirement to use C++ implementations, with the need to provide
  correct initialization of `NRT_MemInfo` object, and
- Conversion of Managed UDF String objects back into cuDF columns when a UDF
  returns strings.

### Managed UDF String creation

Most of the work of Managed UDF String creation is handled in a single
function, ``make_meminfo_for_new_udf_string()``:

```CUDA
// Given a UDF String, return the MemInfo for a newly-created Managed UDF
// String.
__device__ NRT_MemInfo* make_meminfo_for_new_udf_string(udf_string* udf_str);
```

This function does the following:

- A new ``mi_str_allocation``, ``mi_and_str``, is allocated on the heap using
  the ``NRT_Allocate`` function.
- The ``NRT_MemInfo`` in ``mi_and_str`` is initialized with
  ``NRT_MemInfo_init``. This:
  - Initializes the refcount to 1.
  - Sets the destructor to the ``udf_str_dtor()`` function. ``udf_str_dtor()``
    acts like a shim function that calls the C++ destructor for the
    ``udf_sting`` ``st``, deallocating its string data.
  - Initializes the ``size`` to 0 - the destructor does not need information
    about the size of the C++ object.
  - Provides a null pointer for destructor information, as no additional data
    is needed to destroy a ``udf_string``.
- It then copies the ``mi_and_str`` 


Key points:

- **Ownership transfer:** Ownership of the underlying string data that
  ``udf_str`` was passed in with, is transferred to the Managed UDF String.
- **Co-location of MemInfo and UDF String:** Why do we do this?
- **When we cast from ``string_view`` to ``managed_udf_string``, why is the
  ``managed_udf_string`` allowed to take ownership?**
  - It looks like creating a UDF String from a string view just copies the
    pointers, so it seems to be stealing ownership? Don't we get a double free?

## Notes

Placement new is needed so we can initialize an object without it getting
destructed at the end of the scope, because we want its ownership to transfer to
Numba management, not get blown up by C++.
