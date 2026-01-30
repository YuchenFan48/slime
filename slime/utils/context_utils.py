from functools import wraps


def with_defer(deferred_func):
    """
    Decorator that ensures deferred_func is called after the decorated function,
    similar to Go's defer statement.
    
    Usage:
        @with_defer(cleanup_function)
        def my_function():
            # do something
            pass
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            finally:
                deferred_func()

        return wrapper

    return decorator
