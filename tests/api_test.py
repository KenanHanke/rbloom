#!/usr/bin/env python3

from rbloom import Bloom
from hashlib import sha256
from pickle import dumps
import os
import io


def test_bloom(bloom: Bloom):
    assert not bloom
    assert bloom.approx_items == 0.0

    bloom.add('foo')
    assert bloom
    assert bloom.approx_items > 0.0

    bloom.add('bar')

    assert 'foo' in bloom
    assert 'bar' in bloom
    assert 'baz' not in bloom

    bloom.update(['baz', 'qux'])
    assert 'baz' in bloom
    assert 'qux' in bloom

    other = bloom.copy()
    assert other == bloom
    assert other is not bloom

    other.clear()
    assert not other
    assert other.approx_items == 0.0

    other.update(['foo', 'bar', 'baz', 'qux'])
    assert other == bloom

    other.update(str(i).encode()*500 for i in range(100000))
    for i in range(100000):
        assert str(i).encode()*500 in other
    assert bloom != other
    assert bloom & other == bloom
    assert bloom | other == other

    bloom &= other
    assert bloom < other

    orig = bloom.copy()
    bloom |= other
    assert bloom == other
    assert bloom > orig
    assert bloom >= orig
    assert bloom.issuperset(other)
    assert orig <= bloom
    assert orig.issubset(bloom)
    assert bloom >= bloom
    assert bloom.issuperset(bloom)
    assert bloom <= bloom
    assert bloom.issubset(bloom)

    bloom = orig.copy()
    bloom.update(other)
    assert bloom == other
    assert bloom > orig

    bloom = orig.copy()
    assert other == bloom.union(other)
    assert bloom == bloom.intersection(other)

    bloom.intersection_update(other)
    assert bloom == orig

    # TEST PERSISTENCE
    if not bloom.hash_func is hash:
        # find a filename that doesn't exist
        i = 0
        while os.path.exists(f'UNIT_TEST_{i}.bloom'):
            i += 1
        filename = f'test{i}.bloom'

        try:
            # save and load
            bloom.save(filename)
            bloom2 = Bloom.load(filename, bloom.hash_func)
            assert bloom == bloom2
        finally:
            # remove the file
            os.remove(filename)

        # TEST bytes PERSISTENCE
        bloom_bytes = bloom.save()
        assert type(bloom_bytes) == bytes
        bloom3 = Bloom.load(bloom_bytes, bloom.hash_func)
        assert bloom == bloom3

        # TEST fileobj PERSISTENCE
        fileobj = io.BytesIO()

        bloom.save(fileobj)
        fileobj.seek(0)

        bloom4 = Bloom.load(fileobj, bloom.hash_func)
        assert bloom == bloom4


def sha_based(obj):
    h = sha256(dumps(obj)).digest()
    return int.from_bytes(h[:16], "big") - 2**127


def api_suite():
    assert repr(Bloom(27_000, 0.0317)) == "<Bloom size_in_bits=193960 approx_items=0.0>"
    assert Bloom(1140, 0.999).hash_func == hash
    assert Bloom(102, 0.01, hash_func=hash).hash_func is hash
    assert Bloom(103100, 0.51, hash_func=sha_based).hash_func is sha_based

    test_bloom(Bloom(13242, 0.0000001))
    test_bloom(Bloom(9874124, 0.01, hash_func=sha_based))
    test_bloom(Bloom(2837, 0.5, hash_func=hash))

    print('All API tests passed')


if __name__ == '__main__':
    api_suite()
