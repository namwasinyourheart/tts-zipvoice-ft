# read_lca_probe.py
import sys
import numpy as np
from pathlib import Path

def try_read_lca(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    # Import lazily so we don't fail if lhotse not installed
    try:
        from lhotse import LilcomChunkyReader
    except Exception as e:
        raise RuntimeError("Failed to import LilcomChunkyReader from lhotse: " + str(e))

    reader = LilcomChunkyReader(str(path))

    print("LilcomChunkyReader repr:", repr(reader))
    print("\nAvailable attributes and methods on reader:")
    print(dir(reader))

    # Try a number of common method names/approaches:
    attempts = []

    # 1) try .load()
    if hasattr(reader, "load"):
        try:
            arr = reader.load()
            print("Succeeded with reader.load() ->", type(arr), getattr(arr, "shape", None))
            return np.asarray(arr)
        except Exception as e:
            attempts.append(("load()", e))

    # 2) try .read()
    if hasattr(reader, "read"):
        try:
            arr = reader.read()
            print("Succeeded with reader.read() ->", type(arr), getattr(arr, "shape", None))
            return np.asarray(arr)
        except Exception as e:
            attempts.append(("read()", e))

    # 3) try calling the object if callable
    if callable(reader):
        try:
            arr = reader()
            print("Succeeded by calling reader() ->", type(arr), getattr(arr, "shape", None))
            return np.asarray(arr)
        except Exception as e:
            attempts.append(("call()", e))

    # 4) try converting to numpy directly
    try:
        arr = np.asarray(reader)
        print("Succeeded with np.asarray(reader) ->", type(arr), getattr(arr, "shape", None))
        return arr
    except Exception as e:
        attempts.append(("np.asarray(reader)", e))

    # 5) try iterator/list conversion
    try:
        lst = list(reader)   # some versions make it iterable
        arr = np.array(lst)
        print("Succeeded with list(reader) ->", arr.shape)
        return arr
    except Exception as e:
        attempts.append(("list(reader)", e))

    # 6) try indexing/slicing
    try:
        arr = reader[:]
        print("Succeeded with reader[:] ->", getattr(arr, "shape", None))
        return np.asarray(arr)
    except Exception as e:
        attempts.append(("reader[:]", e))

    # 7) fallback: try lhotse.features.io helpers
    try:
        import lhotse.features.io as lio
        if hasattr(lio, "read_lilcom_chunky") or hasattr(lio, "LilcomChunkyReader"):
            # try generic read function names if present
            if hasattr(lio, "read_lilcom_chunky"):
                arr = lio.read_lilcom_chunky(str(path))
                print("Succeeded with lhotse.features.io.read_lilcom_chunky ->", getattr(arr, "shape", None))
                return np.asarray(arr)
    except Exception as e:
        attempts.append(("lhotse.features.io.read_lilcom_chunky", e))

    # If we reached here, nothing worked. Print attempts for debugging:
    print("\nAll attempts failed. Collected exceptions (method, exception):")
    for name, ex in attempts:
        print(f" - {name}: {type(ex).__name__}: {ex}")

    raise RuntimeError(
        "Unable to read .lca with available reader methods. "
        "Please reply with `python -c \"import lhotse; print(lhotse.__version__)\"` "
        "and the output of `python -c \"from lhotse import LilcomChunkyReader; import inspect; print(inspect.getsource(LilcomChunkyReader))\"` "
        "or paste `dir(LilcomChunkyReader)` output here so I can provide exact correct code for your Lhotse version."
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_lca_probe.py /path/to/file.lca")
        sys.exit(2)
    arr = try_read_lca(sys.argv[1])
    print("Final array shape:", getattr(arr, "shape", None))
