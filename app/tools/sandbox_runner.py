import builtins
import io
import json
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr

ALLOWED_MODULES = {
    "math",
    "statistics",
    "random",
    "datetime",
    "time",
    "re",
    "functools",
    "itertools",
    "collections",
    "decimal",
    "fractions",
    "json",
}

ALLOWED_BUILTINS = {
    "abs",
    "all",
    "any",
    "bin",
    "bool",
    "bytes",
    "chr",
    "complex",
    "dict",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hash",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "Exception",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "ArithmeticError",
}


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in ALLOWED_MODULES:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Module '{name}' is not allowed in this environment")


def _build_globals():
    safe_builtins = {key: getattr(builtins, key) for key in ALLOWED_BUILTINS if hasattr(builtins, key)}
    safe_builtins["__import__"] = _restricted_import

    safe_globals = {"__builtins__": safe_builtins}
    for module_name in ALLOWED_MODULES:
        try:
            safe_globals[module_name] = __import__(module_name)
        except Exception:
            # Ignore modules that fail to load (e.g., statistics may not be available)
            pass

    safe_globals["__name__"] = "__sandbox__"
    safe_globals["__package__"] = None
    return safe_globals


def _apply_time_limit(seconds: int = 4):
    try:
        import signal

        def _raise_timeout(signum, frame):
            raise TimeoutError("execution timed out")

        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(seconds)
        return signal
    except Exception:
        return None


def main():
    code = sys.stdin.read()
    if not code.strip():
        json.dump({"ok": False, "stdout": "", "stderr": "", "error": "代码内容为空"}, sys.stdout)
        return

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    safe_globals = _build_globals()

    signal_module = _apply_time_limit()
    start = time.time()
    ok = True
    error_message = None

    try:
        compiled = compile(code, "<sandbox>", "exec")
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(compiled, safe_globals, {})
    except TimeoutError as exc:
        ok = False
        error_message = str(exc)
    except Exception as exc:
        ok = False
        error_message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    finally:
        if signal_module is not None:
            try:
                signal_module.alarm(0)
            except Exception:
                pass

    execution_time = time.time() - start

    result = {
        "ok": ok,
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
        "execution_time": round(execution_time, 4),
    }

    if error_message:
        result["error"] = error_message

    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
