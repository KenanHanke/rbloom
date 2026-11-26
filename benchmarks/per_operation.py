import os
import tempfile
import timeit

import time

NUMBER = 1000000


def format_time(time_ns: float) -> str:
    return f"{time_ns / 1000:.04} us"


def run(bloom_expr: str):
    setup = f"from rbloom import Bloom; b = {bloom_expr}"

    res = timeit.timeit(
        setup=setup,
        stmt="b.add(object())",
        timer=time.perf_counter_ns,
        number=NUMBER,
    )
    print("Time to insert an element:")
    print(format_time(res / NUMBER))

    results = timeit.repeat(
        setup=f"{setup}; objects = [object() for _ in range({NUMBER})]",
        stmt="b.update(objects)",
        timer=time.perf_counter_ns,
        number=1,
        repeat=20,
    )
    res = min(results)
    print("Time to insert each element in a batch:")
    print(format_time(res / NUMBER))

    results = timeit.repeat(
        setup=f"{setup}; objects = (object() for _ in range({NUMBER}))",
        stmt="b.update(objects)",
        timer=time.perf_counter_ns,
        number=1,
        repeat=20,
    )
    res = min(results)
    print("Time to insert each element in a batch via an iterable:")
    print(format_time(res / NUMBER))

    res = timeit.timeit(
        setup=f"{setup}; stored_obj = object(); b.add(stored_obj); b.update(object() for _ in range({NUMBER}))",
        stmt="stored_obj in b",
        timer=time.perf_counter_ns,
        number=NUMBER,
    )
    print("Time to check if an object is present:")
    print(format_time(res / NUMBER))


def main():
    print("== In-memory filter ==")
    run(f"Bloom({NUMBER}, 0.01)")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bench.bloom")
        print()
        print("== mmap-backed filter ==")
        run(f"Bloom({NUMBER}, 0.01, mmap_file={path!r})")


if __name__ == "__main__":
    main()
