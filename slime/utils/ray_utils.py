class Box:
    """
    A simple container class that wraps a value.
    
    Useful for passing mutable references or delayed initialization.
    
    Example:
        box = Box(initial_value)
        print(box.inner)  # Access the wrapped value
    """
    
    def __init__(self, inner):
        self._inner = inner

    @property
    def inner(self):
        return self._inner
