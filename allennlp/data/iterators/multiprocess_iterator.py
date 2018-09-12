from typing import Iterable, Iterator, List, Optional
import copy
import logging

from torch.multiprocessing import Process, Queue, get_logger

from allennlp.common.checks import ConfigurationError
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.dataset import Batch
from allennlp.data.vocabulary import Vocabulary

logger = get_logger()  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

def _create_tensor_dicts(input_queue: Queue,
                         output_queue: Queue,
                         iterator: DataIterator) -> None:
    """
    Pulls at most ``max_instances_in_memory`` from the input_queue,
    groups them into batches of size ``batch_size``, converts them
    to ``TensorDict`` s, and puts them on the ``output_queue``.
    """
    def instances() -> Iterator[Instance]:
        instance = input_queue.get()
        while instance is not None:
            yield instance
            instance = input_queue.get()

    for tensor_dict in iterator(instances(), num_epochs=1, shuffle=False):
        output_queue.put(tensor_dict)

    output_queue.put(None)

def _queuer(instances: Iterable[Instance],
            input_queue: Queue,
            num_workers: int,
            num_epochs: Optional[int]) -> None:
    """
    Reads Instances from the iterable and puts them in the input_queue.
    """
    epoch = 0

    while num_epochs is None or epoch < num_epochs:
        epoch += 1
        for instance in instances:
            input_queue.put(instance)

    # Now put a None for each worker, since each needs to receive one
    # to know that it's done.
    for _ in range(num_workers):
        input_queue.put(None)

@DataIterator.register("multiprocess")
class MultiprocessIterator(DataIterator):
    """
    A ``DataIterator`` that uses ``torch.multiprocessing`` to generate tensor dicts
    using multiple processes. It's currently less full-featured than some of our
    other data iterators.

    Parameters
    ----------
    iterator : ``DataIterator``
        The ``DataIterator`` for generating tensor dicts. It will be cloned
        once per worker.
    num_workers : ``int``, optional (default = 1)
        The number of processes used for generating tensor dicts.
    output_queue_size: ``int``
        The size of the output queue on which tensor dicts are placed to be consumed.
        You might need to increase this if you're generating tensor dicts too quickly.
    """
    def __init__(self,
                 iterator: DataIterator,
                 num_workers: int = 1,
                 output_queue_size: int = 1000,
                 read_from_queue: bool = False) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = iterator._batch_size  # pylint: disable=protected-access
        self.output_queue_size = output_queue_size
        self.iterators = [copy.deepcopy(iterator) for _ in range(num_workers)]

        self.processes: List[Process] = []
        self.queuer: Optional[Process] = None

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        raise RuntimeError("MultiprocessIterator doesn't use create_batches")

    def index_with(self, vocab: Vocabulary):
        for iterator in self.iterators:
            iterator.index_with(vocab)

    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = None,
                 shuffle: bool = True) -> Iterator[TensorDict]:

        # If you run it forever, the multiprocesses won't shut down correctly.
        # TODO(joelgrus) find a solution for this
        if num_epochs is None:
            raise ConfigurationError("Multiprocess Iterator must be run for a fixed number of epochs")

        input_queue = Queue(self.output_queue_size * self.batch_size)
        output_queue = Queue(self.output_queue_size)

        # Start the tensor-dict workers.
        for iterator in self.iterators:
            args = (input_queue, output_queue, iterator)
            process = Process(target=_create_tensor_dicts, args=args)
            process.start()
            self.processes.append(process)

        # Start the queue-populating worker.
        self.queuer = Process(target=_queuer, args=(instances, input_queue, self.num_workers, num_epochs))
        self.queuer.start()

        num_finished = 0
        while num_finished < self.num_workers:
            item = output_queue.get()
            if item is None:
                # finished
                num_finished += 1
                logger.info(f"{num_finished} / {self.num_workers} workers finished")
            else:
                yield item

        self.queuer.join()
        self.queuer = None

        for process in self.processes:
            process.join()
        self.processes.clear()
