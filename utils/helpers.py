import itertools as it


def nwise(iterable, n=2):
    """
    Construct a list of n-tuples of elements which are at distance `n` from the first element in the tuple. 
    It can be viewed as a sliding window of size `n` over a list.
     
    Args:
        iterable: a list or a generator 
        n: window size 

    Returns:
        An iterator which is a iterable of tuples each containing a pair as described above.
    """
    iters = it.tee(iterable, n)
    for i, obj in enumerate(iters):
        next(it.islice(obj, i, i), None)
    return it.izip(*iters)


def pairs_of_neighbours(iterable, window_size=2, add_reciprocal=True):
    """
    Construct a list of pairs (tuples) with the neighbouring elements from a window. 
    
    Args:
        iterable: an iterable object which contains all the elements in fixed order 
        window_size: the size of the neighbourhood (sliding window)
        add_reciprocal: bool flag whether the reciprocal connections should be added too 

    Returns:
        a list of pairs
    """
    pairs = list(set(sum([list(it.combinations(x, 2)) for x in nwise(iterable, n=window_size)], [])))
    if add_reciprocal:
        pairs = pairs + map(lambda y: (y[1], y[0]), pairs)

    return pairs
