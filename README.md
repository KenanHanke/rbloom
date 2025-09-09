# rBloom

[![PyPI](https://img.shields.io/pypi/v/rbloom)](https://pypi.org/project/rbloom/)
[![license](https://img.shields.io/github/license/KenanHanke/rbloom)](https://github.com/KenanHanke/rbloom/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17084925.svg)](https://doi.org/10.5281/zenodo.17084925)
[![build](https://img.shields.io/github/actions/workflow/status/KenanHanke/rbloom/CI.yml)](https://github.com/KenanHanke/rbloom/actions)

A fast, simple and lightweight
[Bloom filter](https://en.wikipedia.org/wiki/Bloom_filter) library for
Python, implemented in Rust. It's designed to be as pythonic as possible,
mimicking the built-in `set` type where it can, and works with any
hashable object. While it's a new library (this project was started in
2023), it's currently the fastest option for Python by
a long shot (see the section
[Benchmarks](#benchmarks)). Releases are published on
[PyPI](https://pypi.org/project/rbloom/).

## Quickstart

This library defines only one class, which can be used as follows:

```python
>>> from rbloom import Bloom
>>> bf = Bloom(200, 0.01)  # 200 items max, false positive rate of 1%
>>> bf.add("hello")
>>> "hello" in bf
True
>>> "world" in bf
False
>>> bf.update(["hello", "world"])  # "hello" and "world" now in bf
>>> other_bf = Bloom(200, 0.01)

### add some items to other_bf

>>> third_bf = bf | other_bf    # third_bf now contains all items in
                                # bf and other_bf
>>> third_bf = bf.copy()
... third_bf.update(other_bf)   # same as above
>>> bf.issubset(third_bf)    # bf <= third_bf also works
True
```

For the full API, see the section [Documentation](#documentation).

## Installation

On almost all platforms, simply run:

```sh
pip install rbloom
```

If you're on an uncommon platform, this may cause pip to build the library
from source, which requires the Rust
[toolchain](https://www.rust-lang.org/tools/install). You can also build
`rbloom` by cloning this repository and running
[maturin](https://github.com/PyO3/maturin):

```sh
maturin build --release
```

This will create a wheel in the `target/wheels/` directory, which can
subsequently also be passed to pip.

## Why rBloom?

Why should you use this library instead of one of the other
Bloom filter libraries on PyPI?

- **Simple:** Almost all important methods work exactly like their
  counterparts in the built-in `set`
  [type](https://docs.python.org/3/library/stdtypes.html#set-types-set-frozenset).
- **Fast:** `rbloom` is implemented in Rust, which makes it
  blazingly fast. See section [Benchmarks](#benchmarks) for more
  information.
- **Lightweight:** `rbloom` has no dependencies of its own.
- **Maintainable:** This library is very concise, and it's written
  in idiomatic Rust. Even if I were to stop maintaining `rbloom` (which I
  don't intend to), it would be trivially easy for you to fork it and keep
  it working for you.

I started `rbloom` because I was looking for a simple Bloom filter
dependency for a project, and the only sufficiently fast option
(`pybloomfiltermmap3`) was segfaulting on recent Python versions. `rbloom`
ended up being twice as fast and has grown to encompass a more complete
API (e.g. with set comparisons like `issubset`). Do note that it doesn't
use mmapped files, however. This shouldn't be an issue in most cases, as
the random access heavy nature of a Bloom filter negates the benefits of
mmap after very few operations, but it is something to keep in mind for
edge cases.

## Benchmarks

The following simple benchmark was implemented in the respective API of
each library (see the [comparison benchmarks](benchmarks/compare.py)):

```python
bf = Bloom(10_000_000, 0.01)

for i in range(10_000_000):
    bf.add(i + 0.5)  # floats because ints are hashed as themselves

for i in range(10_000_000):
    assert i + 0.5 in bf
```

This resulted in the following average runtimes on an M1 Pro (confirmed to be proportional to runtimes on an Intel machine):

| Library                                                            | Time    | Notes                                 |
| ------------------------------------------------------------------ | ------- | ------------------------------------- |
| [rBloom](https://pypi.org/project/rbloom/)                         | 2.52s   | works out-of-the-box                  |
| [pybloomfiltermmap3](https://pypi.org/project/pybloomfiltermmap3/) | 4.78s   | unreliable [1]                        |
| [pybloom3](https://pypi.org/project/pybloom3/)                     | 46.76s  | works out-of-the-box                  |
| [Flor](https://pypi.org/project/Flor/)                             | 76.94s  | doesn't work on arbitrary objects [2] |
| [bloom-filter2](https://pypi.org/project/bloom-filter2/)           | 165.54s | doesn't work on arbitrary objects [2] |

[1] The official package failed to install on Python 3.11 and kept segfaulting on 3.10 (Linux, January 2023). It seems to be fine for now (October 2024).
[2] I was forced to convert to a byte representation, which is bad default behavior as it presents
the problems mentioned below in the section "Cryptographic security".

Also note that `rbloom` is compiled against a stable ABI for
portability, and that you can get a small but measurable speedup by
removing the `"abi3-py37"` flag from `Cargo.toml` and building
it yourself.

## Documentation

This library defines only one class, the signature of which should be
thought of as follows. Note that only the first few methods differ from
the built-in `set` type:

```python
class Bloom:

    # expected_items:  max number of items to be added to the filter
    # false_positive_rate:  max false positive rate of the filter
    # hash_func:  optional argument, see section "Cryptographic security"
    def __init__(self, expected_items: int, false_positive_rate: float,
                 hash_func=__builtins__.hash)

    @property
    def size_in_bits(self) -> int      # number of buckets in the filter

    @property
    def hash_func(self) -> Callable[[Any], int]   # retrieve the hash_func
                                                  # given to __init__

    @property
    def approx_items(self) -> float    # estimated number of items in
                                       # the filter

    # see section "Persistence" for more information on these four methods
    @classmethod
    def load(cls, filepath: str, hash_func) -> Bloom
    def save(self, filepath: str)
    @classmethod
    def load_bytes(cls, data: bytes, hash_func) -> Bloom
    def save_bytes(self) -> bytes

    #####################################################################
    #                    ALL SUBSEQUENT METHODS ARE                     #
    #              EQUIVALENT TO THE CORRESPONDING METHODS              #
    #                     OF THE BUILT-IN SET TYPE                      #
    #####################################################################

    def add(self, obj)                            # add obj to self
    def __contains__(self, obj) -> bool           # check if obj in self
    def __bool__(self) -> bool                    # False if empty
    def __repr__(self) -> str                     # basic info

    def __or__(self, other: Bloom) -> Bloom       # self | other
    def __ior__(self, other: Bloom)               # self |= other
    def __and__(self, other: Bloom) -> Bloom      # self & other
    def __iand__(self, other: Bloom)              # self &= other

    # these extend the functionality of __or__, __ior__, __and__, __iand__
    def union(self, *others: Union[Iterable, Bloom]) -> Bloom        # __or__
    def update(self, *others: Union[Iterable, Bloom])                # __ior__
    def intersection(self, *others: Union[Iterable, Bloom]) -> Bloom # __and__
    def intersection_update(self, *others: Union[Iterable, Bloom])   # __iand__

    # these implement <, >, <=, >=, ==, !=
    def __lt__, __gt__, __le__, __ge__, __eq__, __ne__(self,
                                                       other: Bloom) -> bool
    def issubset(self, other: Bloom) -> bool      # self <= other
    def issuperset(self, other: Bloom) -> bool    # self >= other

    def clear(self)                               # remove all items
    def copy(self) -> Bloom                       # duplicate self
```

To prevent death and destruction, the bitwise set operations only work on
filters where all parameters are equal (including the hash functions being
the exact same object). Because this is a Bloom filter, the `__contains__`
and `approx_items` methods are probabilistic, as are all the methods that
compare two filters (such as `__le__` and `issubset`).

## Cryptographic security

Python's built-in hash function is designed to be fast, not maximally
collision-resistant, so if your program depends on the false positive rate
being perfectly correct, you may want to supply your own hash function.
This is especially the case when working with very large filters (more
than a few tens of millions of items) or when false positives are very
costly and could be exploited by an adversary. Just make sure that your
hash function returns an integer between -2^127 and 2^127 - 1. Feel free
to use the following example in your own code:

```python
from rbloom import Bloom
from hashlib import sha256
from pickle import dumps

def hash_func(obj):
    h = sha256(dumps(obj)).digest()
    # use sys.byteorder instead of "big" for a small speedup when
    # reproducibility across machines isn't a concern
    return int.from_bytes(h[:16], "big", signed=True)

bf = Bloom(100_000_000, 0.01, hash_func)
```

When you throw away Python's built-in hash function and start hashing
serialized representations of objects, however, you open up a breach into
the scary realm of the unpythonic:

- Numbers like `1`, `1.0`, `1 + 0j` and `True` will suddenly no longer
  be equal.
- Instances of classes with custom hashing logic (e.g. to stop
  caches inside instances from affecting their hashes) will suddenly
  display undefined behavior.
- Objects that can't be serialized simply won't be hashable at all.

Making you supply your own hash function in this case is a deliberate
design decision intended to show you what you're doing and prevent
you from shooting yourself in the foot.

Also note that using a custom hash will incur a performance penalty over
using the built-in hash.

## Persistence

The `save` and `load` methods, along with their byte-oriented counterparts
`save_bytes` and `load_bytes`, allow you to save and load filters to and
from disk/Python `bytes` objects. However, as the built-in hash function's
salt changes between invocations of Python, they only work on filters with
custom hash functions. Note that it is your responsibility to ensure that
the hash function you supply to the loading functions is the same as the
one originally used by the filter you're loading!

```python
bf = Bloom(10_000, 0.01, some_hash_func)
bf.add("hello")
bf.add("world")

# saving to a file
bf.save("bf.bloom")

# loading from a file
loaded_bf = Bloom.load("bf.bloom", some_hash_func)
assert loaded_bf == bf

# saving to bytes
bf_bytes = bf.save_bytes()

# loading from bytes
loaded_bf_from_bytes = Bloom.load_bytes(bf_bytes, some_hash_func)
assert loaded_bf_from_bytes == bf
```

The size of the file is `bf.size_in_bits / 8 + 8` bytes.

---

**Statement of attribution:** Bloom filters were originally proposed in
[(Bloom, 1970)](https://doi.org/10.1145/362686.362692). Furthermore, this
implementation makes use of a constant recommended by
[(L'Ecuyer, 1999)](https://doi.org/10.1090/S0025-5718-99-00996-5) for
redistributing the entropy of a single hash over multiple integers using a
linear congruential generator.
