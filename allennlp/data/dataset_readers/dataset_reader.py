from typing import Iterable, Iterator, Callable

from allennlp.data.instance import Instance
from allennlp.common import Params
from allennlp.common.registrable import Registrable
from allennlp.common.util import ensure_list

class _LazyInstances(Iterable):
    """
    An ``Iterable`` that just wraps a thunk and calls it for
    each call to ``__iter__``.
    """
    def __init__(self, thunk: Callable[[], Iterator[Instance]]) -> None:
        super().__init__()
        self.thunk = thunk

    def __iter__(self) -> Iterator[Instance]:
        return self.thunk()

class DatasetReader(Registrable):
    """
    A ``DatasetReader`` reads data from some location and constructs an :class:`InstanceGenerator`
    that returns an ``Iterable`` of the dataset's instances each time it's called. All parameters
    necessary to _read the data apart from the filepath should be passed to the constructor of the
    ``DatasetReader``.
    """
    lazy = False

    def instances(self, file_path: str) -> Iterable[Instance]:
        """
        Returns an ``Iterable`` containing all the instances
        in the specified dataset.

        If ``self.lazy`` is False, this calls ``self._read()``,
        calls ``ensure_list`` on the result (which is a no-op if
        self._read() returns a list, which most of the existing
        dataset readers do), and returns the resulting list.

        If ``self.lazy`` is True, this returns an object whose
        ``__iter__`` method calls ``self._read()`` each iteration.
        Note that if your implementation of ``_read()`` is not lazy
        (i.e. it loads all instances into memory at once), then your
        "lazy" dataset reader is just loading the entire dataset into
        a list each time you iterate over it, which is probably not
        what you want.

        In either case, the returned ``Iterable`` can be iterated
        over multiple times. It's unlikely you want to override this function,
        but if you do your result should likewise be repeatedly iterable.
        """
        if self.lazy:
            return _LazyInstances(lambda: iter(self._read(file_path)))
        else:
            return ensure_list(self._read(file_path))

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Reads the instances from the given file_path and returns them as an
        `Iterable` (which could be a list or could be a generator).
        """
        raise NotImplementedError

    def text_to_instance(self, *inputs) -> Instance:
        """
        Does whatever tokenization or processing is necessary to go from textual input to an
        ``Instance``.  The primary intended use for this is with a
        :class:`~allennlp.service.predictors.predictor.Predictor`, which gets text input as a JSON
        object and needs to process it to be input to a model.

        The intent here is to share code between :func:`_read` and what happens at
        model serving time, or any other time you want to make a prediction from new data.  We need
        to process the data in the same way it was done at training time.  Allowing the
        ``DatasetReader`` to process new text lets us accomplish this, as we can just call
        ``DatasetReader.text_to_instance`` when serving predictions.

        The input type here is rather vaguely specified, unfortunately.  The ``Predictor`` will
        have to make some assumptions about the kind of ``DatasetReader`` that it's using, in order
        to pass it the right information.
        """
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Params) -> 'DatasetReader':
        """
        Static method that constructs the dataset reader described by ``params``.
        """
        choice = params.pop_choice('type', cls.list_available())
        return cls.by_name(choice).from_params(params)
