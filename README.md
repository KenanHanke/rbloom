# rBloom

[![PyPI](https://img.shields.io/pypi/v/rbloom?color=blue)](https://pypi.org/project/rbloom/)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/kenbyte/rbloom?color=blue)](https://github.com/kenbyte/rbloom)

Ultralightweight, blazing fast, minimalistic bloom filter library for Python, implemented in Rust.

## Usage

This library defines only one class, the signature of which should be thought of as:

```python
class Bloom:

    def __init__(self, size_in_bits):
        ...

    def __contains__(self, object):
        ...

    def add(self, object):
        ...
```

See [examples](#examples).

The size in bits is equal to the theoretical maximum amount of objects that could be
contained in the filter. However, the filter should ideally be significantly larger
than this to reduce the likelihood of birthday collisions, which in practice result
in a false positive `True` returned by the `__contains__` method. To decide on an ideal
size, calculate `size_in_bits` by dividing the maximum number of expected items by the
maximum acceptable likelihood of a false positive
(e.g. 200 items / 0.01 likelihood = 20000 bits).

## Building

Use [maturin](https://github.com/PyO3/maturin) to build this library.
As of the time of writing, this can be performed with:

```sh
$ pip install maturin
$ maturin build --release
```

This will result in the creation of a wheel, which can be found in `target/wheels`.

## Examples

Most primitive example:

```python
from rbloom import Bloom

filter = Bloom(200)

assert "hello" not in filter

filter.add("hello")

assert "hello" in filter
```

Print the first 1000 squares as well as around 0.001 = 0.1% of the numbers in between:

```python
from rbloom import Bloom

filter = Bloom(int(1000 / 0.001))

for i in range(1, 1001):
    filter.add(i*i)

for i in range(1, 1000**2 + 1):
    if i in filter:
        print(i, end=" ")
```
