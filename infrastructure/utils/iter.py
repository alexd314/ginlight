from collections import deque
import typing as tp
import itertools

T = tp.TypeVar('T')

def _cached_iterator(iterable : tp.Iterable, dq:tp.Deque) -> tp.Iterator:
    it = iter(iterable)
    while True:
        v = next(it,None)
        if v is not None:
            dq.append(v)
            yield v
        else:
            return

def takewhile_withlast(pred: tp.Callable, iterable: tp.Iterable[tp.T], last : tp.Deque) -> tp.Iterator[tp.T]:
    return itertools.takewhile(pred,_cached_iterator(iterable,last))

def first(iterable : tp.Iterable[T]) -> T:
    return next(iter(iterable))

def test():

    last = deque(maxlen=1)
    str = "params.obj1[0].value"
    iterable = iter(str)
    part1 = "".join(takewhile_withlast(lambda x: x not in ['.','[',']'],iterable, last))
    print(part1)
    print(last.pop())
    part2 = "".join(takewhile_withlast(lambda x: x not in ['.','[',']'],iterable, last))
    print(part2)
    print(last.pop())