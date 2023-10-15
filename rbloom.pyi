from typing import Any, Callable, Iterable, Union, final, Optional, BinaryIO, overload


@final
class Bloom:

    # expected_items:  max number of items to be added to the filter
    # false_positive_rate:  max false positive rate of the filter
    # hash_func:  optional argument, see section "Cryptographic security"
    def __init__(self, expected_items: int, false_positive_rate: float,
                 hash_func: Callable[[Any], int]=__builtins__.hash) -> None: ...

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
    def load(cls, dest: Union[bytes, str, BinaryIO], hash_func: Callable[[Any], int]) -> Bloom: ...

    # save to file, file object, or return bytes, see section "Persistence"
    @overload
    def save(self, source: Union[str, BinaryIO]) -> None: ...

    @overload
    def save(self) -> bytes: ...

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

    def __ior__(self, other: Bloom) -> None: ...      # self |= other

    def __and__(self, other: Bloom) -> Bloom: ...     # self & other

    def __iand__(self, other: Bloom) -> None: ...     # self &= other

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

    def issubset(self, other: Bloom, /) -> bool: ...      # self <= other

    def issuperset(self, other: Bloom, /) -> bool: ...    # self >= other

    def clear(self) -> None: ...                          # remove all items

    def copy(self) -> Bloom: ...                          # duplicate self
