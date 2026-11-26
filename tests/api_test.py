#!/usr/bin/env python3
import gc
import mmap
import pathlib
import tempfile
import threading
import weakref

from rbloom import Bloom
from hashlib import sha256
from pickle import dumps
import os


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
        bloom_bytes = bloom.save_bytes()
        assert type(bloom_bytes) == bytes
        bloom3 = Bloom.load_bytes(bloom_bytes, bloom.hash_func)
        assert bloom == bloom3


def sha_based(obj):
    h = sha256(dumps(obj)).digest()
    return int.from_bytes(h[:16], "big") - 2**127


def circular_ref():
    def loop_hash_func(x):
        return sha_based(x)
    weak_ref = weakref.ref(loop_hash_func)
    bloom = Bloom(1000, 0.1, hash_func=loop_hash_func)
    # In Newer pyo3 versions, instances keep a reference to their type, as CPython requires.
    assert [r for r in gc.get_referents(bloom) if r is not Bloom] == [loop_hash_func]
    loop_hash_func.bloom = bloom
    del bloom
    del loop_hash_func
    gc.collect()
    assert weak_ref() is None

def operations_with_self():
    bloom = Bloom(1000, 0.1)
    bloom.add('foo')
    assert 'foo' in bloom
    bloom |= bloom
    bloom &= bloom
    bloom.update(bloom)
    bloom.update(bloom, bloom)
    bloom.update(bloom, ['bob'], bloom)
    assert 'foo' in bloom
    assert 'bob' in bloom

    bloom.intersection_update(bloom)
    bloom.intersection_update(bloom, bloom)
    bloom.intersection_update(bloom, ['foo'], bloom)
    assert 'foo' in bloom
    assert 'bob' not in bloom
    assert bloom == bloom
    assert bloom <= bloom
    assert bloom >= bloom
    assert not (bloom > bloom)
    assert not (bloom < bloom)
    assert bloom.issubset(bloom)
    assert bloom.issuperset(bloom)
    assert bloom.union(bloom) == bloom
    assert bloom.union(bloom, bloom) == bloom
    assert bloom.intersection(bloom) == bloom
    assert bloom.intersection(bloom, bloom) == bloom


def expect_error(error, operation):
    try:
        operation()
    except error:
        pass
    else:
        raise AssertionError(f'expected {error.__name__}')


def buffer_backed():
    n = Bloom.buffer_size(1000, 0.01)
    # an 8 byte header holding k, followed by the bits, as written by save
    assert n == Bloom(1000, 0.01).size_in_bits // 8 + 8
    for expected_items, false_positive_rate in [(0, 0.01), (1000, 0.0), (1000, 1.0)]:
        expect_error(ValueError, lambda: Bloom.buffer_size(expected_items, false_positive_rate))

    assert not Bloom(1000, 0.01).readonly
    buf = bytearray(b'\xff' * n)
    bloom = Bloom.init_buffer(buf, 1000, 0.01, sha_based)
    assert not bloom.readonly
    # init_buffer starts an empty filter, whatever the buffer held
    assert not bloom
    bloom.add('foo')
    assert 'foo' in bloom
    assert any(buf[8:])

    # load_buffer adopts the existing contents, through any object exporting the memory
    alias = Bloom.load_buffer(memoryview(buf), sha_based)
    assert 'foo' in alias

    # filters sharing a buffer see each other's writes, including through |= with themselves
    alias.add('bar')
    assert 'bar' in bloom
    bloom |= alias
    bloom &= alias
    bloom |= bloom
    assert bloom == alias

    # results of copies and operators own their bits
    copy = bloom.copy()
    copy.add('baz')
    assert 'baz' not in bloom
    assert 'baz' not in bloom | Bloom(1000, 0.01, sha_based)

    with tempfile.TemporaryFile() as f:
        f.truncate(n)
        m = mmap.mmap(f.fileno(), n)
        with Bloom.init_buffer(m, 1000, 0.01, sha_based) as mapped:
            mapped.add('foo')
            # like a memoryview, the filter holds the buffer until it's released
            expect_error(BufferError, m.close)
        with Bloom.load_buffer(m, sha_based) as loaded:
            assert 'foo' in loaded
        m.close()

    for bad_buffer, error in [
        (bytearray(n + 1), ValueError),
        (bytes(n), BufferError),
        (memoryview(bytearray(2 * n))[::2], BufferError),
        ('not a buffer', TypeError),
    ]:
        expect_error(error, lambda: Bloom.init_buffer(bad_buffer, 1000, 0.01))

    expect_error(BufferError, lambda: buf.extend(b'\0'))
    alias.release()
    bloom.release()
    bloom.release()
    buf.extend(b'\0')
    assert repr(bloom) == f'<Bloom size_in_bits={bloom.size_in_bits} released>'
    for operation in [
        lambda: bloom.add('foo'),
        lambda: 'foo' in bloom,
        lambda: bool(bloom),
        lambda: bloom.readonly,
        lambda: bloom == alias,
        lambda: bloom | copy,
        lambda: copy.update(bloom),
        bloom.copy,
        bloom.save_bytes,
    ]:
        expect_error(ValueError, operation)
    # releasing a filter that owns its bits does nothing
    copy.release()
    assert 'foo' in copy

    def new_buffer_backed(expected_items, false_positive_rate, hash_func=hash):
        buf = bytearray(Bloom.buffer_size(expected_items, false_positive_rate))
        return Bloom.init_buffer(buf, expected_items, false_positive_rate, hash_func)

    test_bloom(new_buffer_backed(13242, 0.0000001))
    test_bloom(new_buffer_backed(9874124, 0.01, sha_based))


def buffer_matches_saved_format():
    bloom = Bloom(1000, 0.01, sha_based)
    bloom.update(['foo', 'bar'])
    saved = bloom.save_bytes()
    assert len(saved) == Bloom.buffer_size(1000, 0.01)

    # init_buffer lays out exactly the header and bits that save writes
    buf = bytearray(Bloom.buffer_size(1000, 0.01))
    backed = Bloom.init_buffer(buf, 1000, 0.01, sha_based)
    backed.update(['foo', 'bar'])
    assert bytes(buf) == saved
    assert backed.save_bytes() == saved
    assert Bloom.load_bytes(bytes(buf), sha_based) == bloom

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'saved.bloom')
        bloom.save(path)

        # a file written by save can be memory-mapped and used directly
        with open(path, 'r+b') as f:
            m = mmap.mmap(f.fileno(), 0)
        loaded = Bloom.load_buffer(m, sha_based)
        assert loaded == bloom
        assert loaded.size_in_bits == bloom.size_in_bits
        loaded.add('baz')
        m.flush()
        assert 'baz' in Bloom.load(path, sha_based)
        del loaded
        m.close()

        # read-only buffers support everything except changing the filter
        expected = Bloom.load(path, sha_based)
        with open(path, 'rb') as f:
            m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            contents = f.read()
        expect_error(BufferError, lambda: Bloom.init_buffer(m, 1000, 0.01, sha_based))
        for read_only in [Bloom.load_buffer(m, sha_based), Bloom.load_buffer(contents, sha_based)]:
            assert read_only.readonly
            assert read_only == expected
            assert 'foo' in read_only and 'baz' in read_only
            assert read_only.approx_items == expected.approx_items
            assert read_only.save_bytes() == expected.save_bytes()
            assert read_only | Bloom(1000, 0.01, sha_based) == expected
            other = Bloom(1000, 0.01, sha_based)
            other |= read_only
            assert other == expected
            copy = read_only.copy()
            copy.add('qux')
            assert 'qux' in copy
            for change in [
                lambda: read_only.add('qux'),
                lambda: read_only.update(['qux']),
                lambda: read_only.update(Bloom(1000, 0.01, sha_based)),
                lambda: read_only.intersection_update(Bloom(1000, 0.01, sha_based)),
                read_only.clear,
                lambda: read_only.__ior__(read_only),
            ]:
                expect_error(BufferError, change)
            assert read_only == expected
        del read_only
        m.close()

    expect_error(ValueError, lambda: Bloom.load_buffer(bytearray(saved), hash))
    expect_error(ValueError, lambda: Bloom.load_buffer(bytearray(8), sha_based))


def mmap_helpers():
    n = Bloom.buffer_size(1000, 0.01)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'bf.bloom')

        # a path is created, and an existing file is resized to fit
        for _ in range(2):
            with Bloom(1000, 0.01, sha_based, mmap_file=path) as bloom:
                assert not bloom
                bloom.add('foo')
            assert os.path.getsize(path) == n
        with open(path, 'ab') as f:
            f.write(b'\xff' * 100)
        with Bloom(1000, 0.01, sha_based, mmap_file=path) as bloom:
            assert not bloom
            bloom.add('foo')
        assert os.path.getsize(path) == n
        assert 'foo' in Bloom.load(path, sha_based)

        with Bloom.load_mmap(path, sha_based) as loaded:
            assert not loaded.readonly
            loaded.add('bar')
        with Bloom.load_mmap(pathlib.Path(path), sha_based, mode='r') as loaded:
            assert loaded.readonly
            assert 'foo' in loaded and 'bar' in loaded
            expect_error(BufferError, lambda: loaded.add('baz'))

        # file objects are used as they are, including unflushed writes, and left open
        Bloom(1000, 0.01, sha_based).save(path)
        with open(path, 'r+b') as f:
            saved = f.read()
            with Bloom.load_mmap(f, sha_based) as loaded:
                assert not loaded
                loaded.add('foo')
            assert not f.closed
        with open(path, 'rb') as f:
            assert f.read() != saved

        with tempfile.TemporaryFile() as f:
            f.write(b'partial')
            with Bloom(1000, 0.01, sha_based, mmap_file=f) as bloom:
                bloom.add('foo')
            assert os.fstat(f.fileno()).st_size == n
            with Bloom.load_mmap(f, sha_based, mode='r') as loaded:
                assert 'foo' in loaded

        with open(path, 'rb') as f:
            expect_error(PermissionError, lambda: Bloom.load_mmap(f, sha_based))

        expect_error(ValueError, lambda: Bloom.load_mmap(path, sha_based, mode='w'))
        expect_error(TypeError, lambda: Bloom.load_mmap(42, sha_based))
        expect_error(TypeError, lambda: Bloom(1000, 0.01, mmap_file=42))
        expect_error(FileNotFoundError, lambda: Bloom.load_mmap(os.path.join(tmp, 'missing'), sha_based))
        expect_error(TypeError, lambda: Bloom(1000, 0.01, sha_based, path))


def buffer_circular_ref():
    class Buffer(bytearray):
        pass

    buf = Buffer(Bloom.buffer_size(1000, 0.1))
    weak_ref = weakref.ref(buf)
    buf.bloom = Bloom.init_buffer(buf, 1000, 0.1)
    assert [r for r in gc.get_referents(buf.bloom) if r is not Bloom] == [buf]
    del buf
    gc.collect()
    assert weak_ref() is None


def concurrent_access():
    n = Bloom.buffer_size(100_000, 0.01)
    shared = bytearray(n)
    filters = [
        Bloom(100_000, 0.01),
        Bloom.init_buffer(bytearray(n), 100_000, 0.01),
        # several filters over one buffer
        *[Bloom.init_buffer(shared, 100_000, 0.01) for _ in range(4)],
    ]
    for bloom in filters:
        others = [Bloom(100_000, 0.01) for _ in range(2)]
        for i, other in enumerate(others):
            other.update(range(-i - 1, -1000, -2))

        def worker(start):
            for i in range(start, 20_000, 4):
                bloom.add(i)
                assert i in bloom
            for other in others:
                bloom.__ior__(other)
                other.__ior__(bloom)
                assert bloom >= other
                assert bloom == bloom

        threads = [threading.Thread(target=worker, args=(start,)) for start in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert all(i in bloom for i in range(20_000))
        assert all(i in bloom for i in range(-1, -1000, -1))


def api_suite():
    assert repr(Bloom(27_000, 0.0317)) == "<Bloom size_in_bits=193960 approx_items=0.0>"
    assert Bloom(1140, 0.999).hash_func == hash
    assert Bloom(102, 0.01, hash_func=hash).hash_func is hash
    assert Bloom(103100, 0.51, hash_func=sha_based).hash_func is sha_based

    test_bloom(Bloom(13242, 0.0000001))
    test_bloom(Bloom(9874124, 0.01, hash_func=sha_based))
    test_bloom(Bloom(2837, 0.5, hash_func=hash))

    circular_ref()
    operations_with_self()
    buffer_backed()
    buffer_matches_saved_format()
    mmap_helpers()
    buffer_circular_ref()
    concurrent_access()

    print('All API tests passed')


if __name__ == '__main__':
    api_suite()
