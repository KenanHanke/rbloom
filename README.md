# rBloom

Ultralightweight, blazing fast, minimalistic bloom filter library for Python implemented in Rust.

This library defines only one class, the signature of which should be thought of as:

```
class Bloom:

    def __init__(self, size_in_bits):
        ...

    def __contains__(self, object):
        ...

    def add(self, object):
        ...
```
