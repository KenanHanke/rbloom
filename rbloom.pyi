import os
from typing import Any, Callable, ClassVar, Iterable, Literal, Protocol, Union, final

from typing_extensions import Buffer


class _HasFileno(Protocol):
    def fileno(self) -> int: ...


_MmapFile = Union[str, os.PathLike[str], _HasFileno]


@final
class Bloom:

    # expected_items:  max number of items to be added to the filter
    # false_positive_rate:  max false positive rate of the filter
    # hash_func:  optional argument, see section "Cryptographic security"
    # mmap_file:  optional path or file object to store the filter in, see
    #             section "Buffer-backed filters"
    def __init__(self, expected_items: int, false_positive_rate: float,
                 hash_func: Callable[[Any], int] = ..., *,
                 mmap_file: _MmapFile | None = None) -> None: ...

    # number of buckets in the filter
    @property
    def size_in_bits(self) -> int: ...

    # retrieve the hash_func given to __init__
    @property
    def hash_func(self) -> Callable[[Any], int]: ...

    # estimated number of items in the filter
    @property
    def approx_items(self) -> float: ...

    # load from file, see section "Persistence"
    @classmethod
    def load(cls, filepath: Union[str, os.PathLike[str]], hash_func: Callable[[Any], int]) -> Bloom: ...

    # load from bytes(), see section "Persistence"
    @classmethod
    def load_bytes(cls, data: bytes, hash_func: Callable[[Any], int]) -> Bloom: ...

    # save to file, see section "Persistence"
    def save(self, filepath: Union[str, os.PathLike[str]]) -> None: ...

    # save to a bytes(), see section "Persistence"
    def save_bytes(self) -> bytes: ...

    # number of bytes of a buffer holding a filter with these parameters,
    # see section "Buffer-backed filters"
    @staticmethod
    def buffer_size(expected_items: int, false_positive_rate: float) -> int: ...

    # create an empty filter stored in a writable buffer, see section
    # "Buffer-backed filters"
    @classmethod
    def init_buffer(cls, buffer: Buffer, expected_items: int,
                    false_positive_rate: float,
                    hash_func: Callable[[Any], int] = ...) -> Bloom: ...

    # use a buffer holding a saved filter, see section "Buffer-backed filters"
    @classmethod
    def load_buffer(cls, buffer: Buffer, hash_func: Callable[[Any], int]) -> Bloom: ...

    # memory-map a file holding a saved filter and use it like load_buffer,
    # see section "Buffer-backed filters"
    @classmethod
    def load_mmap(cls, file: _MmapFile, hash_func: Callable[[Any], int], *,
                  mode: Literal["r", "rw"] = "rw") -> Bloom: ...

    # release the buffer of a buffer-backed filter, see section
    # "Buffer-backed filters"
    def release(self) -> None: ...
    def __enter__(self) -> Bloom: ...
    def __exit__(self, *args: object) -> None: ...

    # whether the filter is stored in a read-only buffer
    @property
    def readonly(self) -> bool: ...

    #####################################################################
    #                    ALL SUBSEQUENT METHODS ARE                     #
    #              EQUIVALENT TO THE CORRESPONDING METHODS              #
    #                     OF THE BUILT-IN SET TYPE                      #
    #####################################################################

    def add(self, obj: Any, /) -> None: ...

    def __contains__(self, obj: Any) -> bool: ...

    def __bool__(self) -> bool: ...                   # False if empty

    def __repr__(self) -> str: ...                    # basic info

    def __or__(self, other: Bloom) -> Bloom: ...      # self | other

    def __ior__(self, other: Bloom) -> Bloom: ...     # self |= other

    def __and__(self, other: Bloom) -> Bloom: ...     # self & other

    def __iand__(self, other: Bloom) -> Bloom: ...    # self &= other

    # extension of __or__
    def union(self, *others: Union[Iterable, Bloom]) -> Bloom: ...

    # extension of __ior__
    def update(self, *others: Union[Iterable, Bloom]) -> None: ...

    # extension of __and__
    def intersection(self, *others: Union[Iterable, Bloom]) -> Bloom: ...

    # extension of __iand__
    def intersection_update(self, *others: Union[Iterable, Bloom]) -> None: ...

    # these implement <, >, <=, >=, ==, !=
    def __lt__(self, other: Bloom) -> bool: ...
    def __gt__(self, other: Bloom) -> bool: ...
    def __le__(self, other: Bloom) -> bool: ...
    def __ge__(self, other: Bloom) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...

    # filters are mutable, so they are unhashable like sets
    __hash__: ClassVar[None]  # type: ignore[assignment]

    def issubset(self, other: Union[Iterable, Bloom], /) -> bool: ...    # self <= other

    def issuperset(self, other: Union[Iterable, Bloom], /) -> bool: ...  # self >= other

    def clear(self) -> None: ...                          # remove all items

    def copy(self) -> Bloom: ...                          # duplicate self
