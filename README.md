# rBloom

[![PyPI](https://img.shields.io/pypi/v/rbloom?color=blue)](https://pypi.org/project/rbloom/)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/kenbyte/rbloom?color=blue)](https://github.com/kenbyte/rbloom)

Ultralightweight, blazing fast, minimalistic
[Bloom filter](https://en.wikipedia.org/wiki/Bloom_filter) library for
Python, fully implemented in Rust.

## Usage

This library defines only one class, the signature of which should be thought of as:

```python
class Bloom:

    __init__(self, expected_items: int, false_positive_rate: float,
                 hash_func=__builtins__.hash)

    add(self, object)

    __contains__(self, object) -> bool

    clear(self)

```

Also see the section [Examples](#examples).

To prevent death and destruction, the bitwise set operations only work on
filters where all parameters are equal (including the hash functions being
the exact same object).

## Building

Use [maturin](https://github.com/PyO3/maturin) to build this library.
As of the time of writing, this can be performed with:

```bash
$ pip install maturin
$ maturin build --release
```

This will result in the creation of a wheel, which can be found in `target/wheels`.

## Examples

Most primitive example:

```python
from rbloom import Bloom

filter = Bloom(200, 0.01)

assert "hello" not in filter

filter.add("hello")

assert "hello" in filter
```

Print the first 1000 squares as well as around 0.001 = 0.1% of the numbers in between:

```python
from rbloom import Bloom

filter = Bloom(1000, 0.001)

for i in range(1, 1001):
    filter.add(i*i)

for i in range(1, 1000**2 + 1):
    if i in filter:
        print(i, end=" ")
```

---

When you throw away Python's built-in hash function and hash a serialized
representation, however, you open up a breach to the scary realm of the
unpythonic:

- Numbers like `2`, `2.0` and `2 + 0j` will suddenly no longer be equal.
- Instances of classes with custom hashing logic (e.g. to stop
  caches inside instances from affecting their hashes) will suddenly
  display undefined behavior.
- Objects that can't be serialized simply won't be hashable at all.

## Implementation details

Instead of using multiple hash functions, this program redistributes the
entropy of a single hash over multiple integers by using the single hash
as the seed of a multiplicative linear congruential generator (MLCG). The
constant used is one proposed by
[(L'Ecuyer, 1999)](https://doi.org/10.1090/S0025-5718-99-00996-5).
