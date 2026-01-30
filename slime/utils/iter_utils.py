from collections import defaultdict


# details: https://stackoverflow.com/questions/773/how-do-i-use-itertools-groupby
def group_by(iterable, key=None):
    """
    Similar to itertools.groupby, but does not require iterable to be sorted.
    
    Args:
        iterable: The iterable to group
        key: A function to extract the grouping key from each item.
             If None, the item itself is used as the key.
    
    Returns:
        A dict mapping keys to lists of items with that key.
    
    Example:
        >>> group_by([1, 2, 3, 4, 5], key=lambda x: x % 2)
        {1: [1, 3, 5], 0: [2, 4]}
    """
    ret = defaultdict(list)
    for item in iterable:
        ret[key(item) if key is not None else item].append(item)
    return dict(ret)
