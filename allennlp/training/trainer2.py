import logging
import copy
import math
import os
import time
import re
import datetime
import traceback
from typing import Dict, Optional, List, Tuple, Union, Any, NamedTuple, Iterable, Set

from typing_extensions import Protocol
import torch
import torch.optim.lr_scheduler

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, parse_cuda_device
from allennlp.common.util import (dump_metrics, gpu_memory_mb, peak_memory_mb,
                                  get_frozen_and_tunable_parameter_names, lazy_groups_of)
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training.callbacks import Callback, CallbackHandler, Events
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training.trainer_base import TrainerBase
from allennlp.training import util as training_util
from allennlp.training.moving_average import MovingAverage

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TensorboardHistogramsState(Protocol):
    model: Model
    tensorboard: TensorboardWriter

class LogTensorboardHistograms(Callback[TensorboardHistogramsState]):
    def __init__(self):
        self.histogram_parameters: Set[str] = set()
        self.param_updates: Dict[str, torch.Tensor] = {}

    def __call__(self, event: str, state: TensorboardHistogramsState) -> None:
        if event == Events.TRAINING_START:
            self.histogram_parameters = set(
                    state.model.get_parameters_for_histogram_tensorboard_logging()
            )
        elif event == Events.BATCH_START and state.tensorboard.should_log_histograms_this_batch():
            # get the magnitude of parameter updates for logging
            # We need a copy of current parameters to compute magnitude of updates,
            # and copy them to CPU so large models won't go OOM on the GPU.
            self.param_updates = {name: param.detach().cpu().clone()
                                  for name, param in state.model.named_parameters()}
        elif event == Events.BATCH_END and state.tensorboard.should_log_histograms_this_batch():
            for name, param in state.model.named_parameters():
                self.param_updates[name].sub_(param.detach().cpu())
                update_norm = torch.norm(self.param_updates[name].view(-1, ))
                param_norm = torch.norm(param.view(-1, )).cpu()
                state.tensorboard.add_train_scalar("gradient_update/" + name,
                                                   update_norm / (param_norm + 1e-7))
            self.param_updates.clear()
            state.tensorboard.log_histograms(state.model, self.histogram_parameters)


class TensorboardState(Protocol):
    model: Model
    tensorboard: TensorboardWriter
    optimizer: Optimizer
    batch_grad_norm: float
    batch_group: list
    cumulative_batch_size: int
    train_metrics: dict
    val_metrics: dict


class LogTensorboard(Callback[TensorboardState]):
    def __init__(self, log_batch_size_period: int = None) -> None:
        self.log_batch_size_period = log_batch_size_period
        self.epoch = 1

    def __call__(self, event: str, state: TensorboardState) -> None:
        # At epoch start get parameters for histogram logging
        if event == Events.BATCH_END:
            # Log parameter values to tensorboard
            if state.tensorboard.should_log_this_batch():
                state.tensorboard.log_parameter_and_gradient_statistics(state.model, state.batch_grad_norm)
                state.tensorboard.log_learning_rates(state.model, state.optimizer)

                state.tensorboard.add_train_scalar("loss/loss_train", state.train_metrics["loss"])
                state.tensorboard.log_metrics({"epoch_metrics/" + k: v for k, v in state.train_metrics.items()})


            if self.log_batch_size_period:
                cur_batch = sum([training_util.get_batch_size(batch) for batch in state.batch_group])
                state.cumulative_batch_size += cur_batch
                if (state.batches_this_epoch - 1) % self.log_batch_size_period == 0:
                    average = state.cumulative_batch_size / state.batches_this_epoch
                    logger.info(f"current batch size: {cur_batch} mean batch size: {average}")
                    state.tensorboard.add_train_scalar("current_batch_size", cur_batch)
                    state.tensorboard.add_train_scalar("mean_batch_size", average)

        elif event == Events.EPOCH_END:
            state.tensorboard.log_metrics(state.train_metrics,
                                          val_metrics=state.val_metrics,
                                          log_to_console=True,
                                          epoch=self.epoch)
            self.epoch += 1


class LrsState(Protocol):
    learning_rate_scheduler: LearningRateScheduler
    batch_num_total: int
    epoch_number: int
    val_metrics: dict
    validation_metric: str

class LrsCallback(Callback[LrsState]):
    def __call__(self, event: str, state: LrsState) -> None:
        # Don't do anything if there's no lr_scheduler
        if state.learning_rate_scheduler is None:
            return

        if event == Events.AFTER_BACKWARD:
            state.learning_rate_scheduler.step_batch(state.batch_num_total)
        elif event == Events.EPOCH_END:
            state.learning_rate_scheduler.step(state.val_metrics[state.validation_metric],
                                               state.epoch_number)

class CheckpointState(Protocol):
    checkpointer: Checkpointer
    checkpoint_epoch: Union[int, str]
    model: Model
    metric_tracker: MetricTracker
    epoch_number: int

class CheckpointCallback(Callback[CheckpointState]):
    def __init__(self,
                 state_dict_attrs: List[str] = None,
                 other_attrs: List[str] = None) -> None:
        self.state_dict_attrs = state_dict_attrs or []
        self.other_attrs = other_attrs or []

    def __call__(self, event: str, state: CheckpointState) -> None:
        if event == Events.SAVE_CHECKPOINT:
            training_states = {}
            for attr in self.state_dict_attrs:
                state_attr = getattr(state, attr)
                if state_attr is not None:
                    training_states[attr] = state_attr.state_dict()
            for attr in self.other_attrs:
                training_states[attr] = getattr(state, attr)

            state.checkpointer.save_checkpoint(
                    model_state=state.model.state_dict(),
                    epoch=state.checkpoint_epoch,
                    training_states=training_states,
                    is_best_so_far=state.metric_tracker.is_best_so_far())



@TrainerBase.register("default")
class Trainer(TrainerBase):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Iterable[Instance],
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 validation_iterator: DataIterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 keep_serialized_model_every_num_seconds: int = None,
                 checkpointer: Checkpointer = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,
                 momentum_scheduler: Optional[MomentumScheduler] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False,
                 log_batch_size_period: Optional[int] = None,
                 moving_average: Optional[MovingAverage] = None) -> None:
        """
        A trainer for doing supervised learning. It just takes a labeled dataset
        and a ``DataIterator``, and uses the supplied ``Optimizer`` to learn the weights
        for your model over some fixed number of epochs. You can also pass in a validation
        dataset and enable early stopping. There are many other bells and whistles as well.

        Parameters
        ----------
        model : ``Model``, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their ``forward`` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.

            If you are training your model using GPUs, your model should already be
            on the correct device. (If you use `Trainer.from_params` this will be
            handled for you.)
        optimizer : ``torch.nn.Optimizer``, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : ``DataIterator``, required.
            A method for iterating over a ``Dataset``, yielding padded indexed batches.
        train_dataset : ``Dataset``, required.
            A ``Dataset`` to train on. The dataset should have already been indexed.
        validation_dataset : ``Dataset``, optional, (default = None).
            A ``Dataset`` to evaluate on. The dataset should have already been indexed.
        patience : Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after ``patience`` epochs with no improvement. If given, it must be ``> 0``.
            If None, early stopping is disabled.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an ``is_best`` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        validation_iterator : ``DataIterator``, optional (default=None)
            An iterator to use for the validation set.  If ``None``, then
            use the training `iterator`.
        shuffle: ``bool``, optional (default=True)
            Whether to shuffle the instances in the iterator or not.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        num_serialized_models_to_keep : ``int``, optional (default=20)
            Number of previous model checkpoints to retain.  Default is to keep 20 checkpoints.
            A value of None or -1 means all checkpoints will be kept.
        keep_serialized_model_every_num_seconds : ``int``, optional (default=None)
            If num_serialized_models_to_keep is not None, then occasionally it's useful to
            save models at a given interval in addition to the last num_serialized_models_to_keep.
            To do so, specify keep_serialized_model_every_num_seconds as the number of seconds
            between permanently saved checkpoints.  Note that this option is only used if
            num_serialized_models_to_keep is not None, otherwise all checkpoints are kept.
        checkpointer : ``Checkpointer``, optional (default=None)
            An instance of class Checkpointer to use instead of the default. If a checkpointer is specified,
            the arguments num_serialized_models_to_keep and keep_serialized_model_every_num_seconds should
            not be specified. The caller is responsible for initializing the checkpointer so that it is
            consistent with serialization_dir.
        model_save_interval : ``float``, optional (default=None)
            If provided, then serialize models every ``model_save_interval``
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if ``serialization_dir`` is provided.
        cuda_device : ``Union[int, List[int]]``, optional (default = -1)
            An integer or list of integers specifying the CUDA device(s) to use. If -1, the CPU is used.
        grad_norm : ``float``, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : ``float``, optional (default = ``None``).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting ``NaNs`` in your gradients during training
            that are not solved by using ``grad_norm``, you may need this.
        learning_rate_scheduler : ``LearningRateScheduler``, optional (default = None)
            If specified, the learning rate will be decayed with respect to
            this schedule at the end of each epoch (or batch, if the scheduler implements
            the ``step_batch`` method). If you use :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`,
            this will use the ``validation_metric`` provided to determine if learning has plateaued.
            To support updating the learning rate on every batch, this can optionally implement
            ``step_batch(batch_num_total)`` which updates the learning rate given the batch number.
        momentum_scheduler : ``MomentumScheduler``, optional (default = None)
            If specified, the momentum will be updated at the end of each batch or epoch
            according to the schedule.
        summary_interval: ``int``, optional, (default = 100)
            Number of batches between logging scalars to tensorboard
        histogram_interval : ``int``, optional, (default = ``None``)
            If not None, then log histograms to tensorboard every ``histogram_interval`` batches.
            When this parameter is specified, the following additional logging is enabled:
                * Histograms of model parameters
                * The ratio of parameter update norm to parameter norm
                * Histogram of layer activations
            We log histograms of the parameters returned by
            ``model.get_parameters_for_histogram_tensorboard_logging``.
            The layer activations are logged for any modules in the ``Model`` that have
            the attribute ``should_log_activations`` set to ``True``.  Logging
            histograms requires a number of GPU-CPU copies during training and is typically
            slow, so we recommend logging histograms relatively infrequently.
            Note: only Modules that return tensors, tuples of tensors or dicts
            with tensors as values currently support activation logging.
        should_log_parameter_statistics : ``bool``, optional, (default = True)
            Whether to send parameter statistics (mean and standard deviation
            of parameters and gradients) to tensorboard.
        should_log_learning_rate : ``bool``, optional, (default = False)
            Whether to send parameter specific learning rate to tensorboard.
        log_batch_size_period : ``int``, optional, (default = ``None``)
            If defined, how often to log the average batch size.
        moving_average: ``MovingAverage``, optional, (default = None)
            If provided, we will maintain moving averages for all parameters. During training, we
            employ a shadow variable for each parameter, which maintains the moving average. During
            evaluation, we backup the original parameters and assign the moving averages to corresponding
            parameters. Be careful that when saving the checkpoint, we will save the moving averages of
            parameters. This is necessary because we want the saved model to perform as well as the validated
            model if we load it later. But this may cause problems if you restart the training from checkpoint.
        """
        super().__init__(serialization_dir, cuda_device)

        if checkpointer is not None:
            # We can't easily check if these parameters were passed in, so check against their default values.
            # We don't check against serialization_dir since it is also used by the parent class.
            if num_serialized_models_to_keep != 20 or \
                    keep_serialized_model_every_num_seconds is not None:
                raise ConfigurationError(
                        "When passing a custom Checkpointer, you may not also pass in separate checkpointer "
                        "args 'num_serialized_models_to_keep' or 'keep_serialized_model_every_num_seconds'.")
        else:
            checkpointer = Checkpointer(serialization_dir,
                                        keep_serialized_model_every_num_seconds,
                                        num_serialized_models_to_keep)

        if patience is None:  # no early stopping
            if validation_dataset:
                logger.warning('You provided a validation dataset but patience was set to None, '
                               'meaning that early stopping is disabled')
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError('{} is an invalid value for "patience": it must be a positive integer '
                                     'or None (if you want to disable early stopping)'.format(patience))

        tensorboard = TensorboardWriter(
                get_batch_num_total=lambda: self.batch_num_total,
                serialization_dir=serialization_dir,
                summary_interval=summary_interval,
                histogram_interval=histogram_interval,
                should_log_parameter_statistics=should_log_parameter_statistics,
                should_log_learning_rate=should_log_learning_rate)

        # Enable activation logging.
        if histogram_interval is not None:
            tensorboard.enable_activation_logging(model)

        # This is all state that the callbacks might want:
        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.model = model
        self.iterator = iterator
        self.validation_iterator = validation_iterator or iterator
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.train_data = train_dataset
        self.validation_data = validation_dataset

        # For capturing mid / end-of-epoch metrics
        self.train_metrics: Dict[str, float] = {}
        self.val_metrics: Dict[str, float] = {}

        # For capturing overall metrics
        self.metrics: Dict[str, Any] = {}

        self.batch_num_total = 0
        self.cumulative_batch_size = 0
        self.batch_group = None
        self.batches_this_epoch = 0
        # For tracking is_best_so_far and should_stop_early
        self.metric_tracker = MetricTracker(patience, validation_metric)
        # Get rid of + or -
        self.validation_metric = validation_metric[1:]
        self.num_epochs = num_epochs

        self.checkpointer = checkpointer
        self.checkpoint_epoch: Union[int, str] = 0
        self.model_save_interval = model_save_interval

        self.grad_norm = grad_norm
        self.grad_clipping = grad_clipping
        self.learning_rate_scheduler = learning_rate_scheduler
        self.momentum_scheduler = momentum_scheduler
        self.moving_average = moving_average
        self.tensorboard = tensorboard
        self.last_log = 0.0
        self.epoch_number = 0

        # Set up callback handler
        callbacks: List[Callback] = [
                LogTensorboard(log_batch_size_period),
                LogTensorboardHistograms(),
                LrsCallback(),
                CheckpointCallback(
                        state_dict_attrs=[
                                'metric_tracker',
                                'learning_rate_scheduler',
                                'momentum_scheduler',
                                'optimizer'
                        ],
                        other_attrs=[
                                'batch_num_total',
                        ]
                )
        ]

        self.handler = CallbackHandler(callbacks, self)


    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self.model, self.grad_norm)

    def batch_loss(self, batch_group: List[TensorDict], for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batches and returns the ``loss`` value in the result.
        If ``for_training`` is `True` also applies regularization penalty.
        """
        if self._multiple_gpu:
            output_dict = training_util.data_parallel(batch_group, self.model, self._cuda_devices)
        else:
            assert len(batch_group) == 1
            batch = batch_group[0]
            batch = nn_util.move_to_device(batch, self._cuda_devices[0])
            output_dict = self.model(**batch)

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError("The model you are trying to optimize does not contain a"
                                   " 'loss' key in the output of model.forward(inputs).")
            loss = None

        return loss

    def _train_epoch(self, epoch: int) -> None:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self.num_epochs - 1)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self.model.train()

        num_gpus = len(self._cuda_devices)

        # Get tqdm for the training batches
        raw_train_generator = self.iterator(self.train_data,
                                            num_epochs=1,
                                            shuffle=self.shuffle)
        train_generator = lazy_groups_of(raw_train_generator, num_gpus)
        num_training_batches = math.ceil(self.iterator.get_num_batches(self.train_data)/num_gpus)
        self.last_log = time.time()
        last_save_time = time.time()

        logger.info("Training")
        train_generator_tqdm = Tqdm.tqdm(train_generator,
                                         total=num_training_batches)
        self.handler.state.cumulative_batch_size = 0
        for batch_group in train_generator_tqdm:

            self.handler.fire_event(Events.BATCH_START)

            self.handler.state.batches_this_epoch += 1
            self.handler.state.batch_num_total += 1

            self.optimizer.zero_grad()

            loss = self.batch_loss(batch_group, for_training=True)

            self.handler.fire_event(Events.AFTER_FORWARD)

            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            loss.backward()

            self.handler.fire_event(Events.AFTER_BACKWARD)

            train_loss += loss.item()

            self.batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self.momentum_scheduler:
                self.momentum_scheduler.step_batch(self.handler.state.batch_num_total)

            self.optimizer.step()

            # Update moving averages
            if self.moving_average is not None:
                self.moving_average.apply(self.handler.state.batch_num_total)

            # Update the description with the latest metrics
            self.train_metrics = training_util.get_metrics(self.model, train_loss, self.batches_this_epoch)
            description = training_util.description_from_metrics(self.train_metrics)

            train_generator_tqdm.set_description(description, refresh=False)

            # Save model if needed.
            if self.model_save_interval is not None and (
                    time.time() - last_save_time > self.model_save_interval
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                        '{0}.{1}'.format(epoch, training_util.time_to_str(int(last_save_time)))
                )

            self.handler.fire_event(Events.BATCH_END)

        self.train_metrics = training_util.get_metrics(self.model,
                                                       train_loss,
                                                       self.batches_this_epoch,
                                                       reset=True)
        self.train_metrics['cpu_memory_MB'] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            self.train_metrics['gpu_'+str(gpu_num)+'_memory_MB'] = memory

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self.model.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self.moving_average is not None:
            self.moving_average.assign_average_value()

        if self.validation_iterator is not None:
            val_iterator = self.validation_iterator
        else:
            val_iterator = self.iterator

        num_gpus = len(self._cuda_devices)

        raw_val_generator = val_iterator(self.validation_data,
                                         num_epochs=1,
                                         shuffle=False)
        val_generator = lazy_groups_of(raw_val_generator, num_gpus)
        num_validation_batches = math.ceil(val_iterator.get_num_batches(self.validation_data)/num_gpus)
        val_generator_tqdm = Tqdm.tqdm(val_generator,
                                       total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        for batch_group in val_generator_tqdm:

            loss = self.batch_loss(batch_group, for_training=False)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(self.model, val_loss, batches_this_epoch)
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        # Now restore the original parameter values.
        if self.moving_average is not None:
            self.moving_average.restore()

        return val_loss, batches_this_epoch

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                     "a different serialization directory or delete the existing serialization "
                                     "directory?")

        training_util.enable_gradient_clipping(self.model, self.grad_clipping)

        logger.info("Beginning training.")
        self.handler.fire_event(Events.TRAINING_START)

        this_epoch_val_metric: float = None
        epochs_trained = 0
        training_start_time = time.time()

        self.metrics['best_epoch'] = self.metric_tracker.best_epoch or 0
        for key, value in self.metric_tracker.best_epoch_metrics.items():
            self.metrics["best_validation_" + key] = value

        for epoch in range(epoch_counter, self.num_epochs):
            self.epoch_number = epoch
            self.handler.fire_event(Events.EPOCH_START)
            epoch_start_time = time.time()
            self._train_epoch(epoch)

            # get peak of memory usage
            if 'cpu_memory_MB' in self.train_metrics:
                self.metrics['peak_cpu_memory_MB'] = max(self.metrics.get('peak_cpu_memory_MB', 0),
                                                         self.train_metrics['cpu_memory_MB'])
            for key, value in self.train_metrics.items():
                if key.startswith('gpu_'):
                    self.metrics["peak_"+key] = max(self.metrics.get("peak_"+key, 0), value)

            if self.validation_data is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()
                    self.val_metrics = training_util.get_metrics(self.model, val_loss, num_batches, reset=True)

                    # Check validation metric for early stopping
                    this_epoch_val_metric = self.val_metrics[self.validation_metric]
                    self.metric_tracker.add_metric(this_epoch_val_metric)

                    if self.metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break


            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            self.metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            self.metrics["training_start_epoch"] = epoch_counter
            self.metrics["training_epochs"] = epochs_trained
            self.metrics["epoch"] = epoch

            for key, value in self.train_metrics.items():
                self.metrics["training_" + key] = value
            for key, value in self.val_metrics.items():
                self.metrics["validation_" + key] = value

            if self.metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                self.metrics['best_epoch'] = epoch
                for key, value in self.val_metrics.items():
                    self.metrics["best_validation_" + key] = value

                self.metric_tracker.best_epoch_metrics = copy.deepcopy(self.val_metrics)

            if self._serialization_dir:
                dump_metrics(os.path.join(self._serialization_dir, f'metrics_epoch_{epoch}.json'), self.metrics)

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self.momentum_scheduler:
                self.momentum_scheduler.step(this_epoch_val_metric, epoch)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self.num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self.num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

            self.handler.fire_event(Events.EPOCH_END)

            self._save_checkpoint(epoch)

        # Load the best model state before returning
        best_model_state = self.checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        self.handler.fire_event(Events.TRAINING_END)

        return self.metrics

    def _save_checkpoint(self, epoch: Union[int, str]) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        Parameters
        ----------
        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        """
        # If moving averages are used for parameters, we save
        # the moving average values into checkpoint, instead of the current values.
        if self.moving_average is not None:
            self.moving_average.assign_average_value()

        self.checkpoint_epoch = epoch
        self.handler.fire_event(Events.BEFORE_SAVE_CHECKPOINT)
        self.handler.fire_event(Events.SAVE_CHECKPOINT)
        self.handler.fire_event(Events.AFTER_SAVE_CHECKPOINT)

        # Restore the original values for parameters so that training will not be affected.
        if self.moving_average is not None:
            self.moving_average.restore()

    def _restore_checkpoint(self) -> int:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        Returns
        -------
        epoch: int
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        model_state, training_state = self.checkpointer.restore_checkpoint()

        if not training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if self.learning_rate_scheduler is not None and "learning_rate_scheduler" in training_state:
            self.learning_rate_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
        if self.momentum_scheduler is not None and "momentum_scheduler" in training_state:
            self.momentum_scheduler.load_state_dict(training_state["momentum_scheduler"])
        training_util.move_optimizer_to_cuda(self.optimizer)

        # Currently the ``training_state`` contains a serialized ``MetricTracker``.
        if "metric_tracker" in training_state:
            self.metric_tracker.load_state_dict(training_state["metric_tracker"])
        # It used to be the case that we tracked ``val_metric_per_epoch``.
        elif "val_metric_per_epoch" in training_state:
            self.metric_tracker.clear()
            self.metric_tracker.add_metrics(training_state["val_metric_per_epoch"])
        # And before that we didn't track anything.
        else:
            self.metric_tracker.clear()

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get('batch_num_total')
        if batch_num_total is not None:
            self.handler.state.batch_num_total = batch_num_total

        return epoch_to_return

    # Requires custom from_params.
    @classmethod
    def from_params(cls,  # type: ignore
                    model: Model,
                    serialization_dir: str,
                    iterator: DataIterator,
                    train_data: Iterable[Instance],
                    validation_data: Optional[Iterable[Instance]],
                    params: Params,
                    validation_iterator: DataIterator = None) -> 'Trainer':
        # pylint: disable=arguments-differ
        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)
        momentum_scheduler_params = params.pop("momentum_scheduler", None)

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(model_device)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))
        if "moving_average" in params:
            moving_average = MovingAverage.from_params(params.pop("moving_average"), parameters=parameters)
        else:
            moving_average = None

        if lr_scheduler_params:
            lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            lr_scheduler = None
        if momentum_scheduler_params:
            momentum_scheduler = MomentumScheduler.from_params(optimizer, momentum_scheduler_params)
        else:
            momentum_scheduler = None

        if 'checkpointer' in params:
            if 'keep_serialized_model_every_num_seconds' in params or \
                    'num_serialized_models_to_keep' in params:
                raise ConfigurationError(
                        "Checkpointer may be initialized either from the 'checkpointer' key or from the "
                        "keys 'num_serialized_models_to_keep' and 'keep_serialized_model_every_num_seconds'"
                        " but the passed config uses both methods.")
            checkpointer = Checkpointer.from_params(params.pop("checkpointer"))
        else:
            num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
            keep_serialized_model_every_num_seconds = params.pop_int(
                    "keep_serialized_model_every_num_seconds", None)
            checkpointer = Checkpointer(
                    serialization_dir=serialization_dir,
                    num_serialized_models_to_keep=num_serialized_models_to_keep,
                    keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds)
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = params.pop_int("log_batch_size_period", None)

        params.assert_empty(cls.__name__)
        return cls(model, optimizer, iterator,
                   train_data, validation_data,
                   patience=patience,
                   validation_metric=validation_metric,
                   validation_iterator=validation_iterator,
                   shuffle=shuffle,
                   num_epochs=num_epochs,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   grad_norm=grad_norm,
                   grad_clipping=grad_clipping,
                   learning_rate_scheduler=lr_scheduler,
                   momentum_scheduler=momentum_scheduler,
                   checkpointer=checkpointer,
                   model_save_interval=model_save_interval,
                   summary_interval=summary_interval,
                   histogram_interval=histogram_interval,
                   should_log_parameter_statistics=should_log_parameter_statistics,
                   should_log_learning_rate=should_log_learning_rate,
                   log_batch_size_period=log_batch_size_period,
                   moving_average=moving_average)


class TrainerPieces(NamedTuple):
    """
    We would like to avoid having complex instantiation logic taking place
    in `Trainer.from_params`. This helper class has a `from_params` that
    instantiates a model, loads train (and possibly validation and test) datasets,
    constructs a Vocabulary, creates data iterators, and handles a little bit
    of bookkeeping. If you're creating your own alternative training regime
    you might be able to use this.
    """
    model: Model
    iterator: DataIterator
    train_dataset: Iterable[Instance]
    validation_dataset: Iterable[Instance]
    test_dataset: Iterable[Instance]
    validation_iterator: DataIterator
    params: Params

    @staticmethod
    def from_params(params: Params,
                    serialization_dir: str,
                    recover: bool = False,
                    cache_directory: str = None,
                    cache_prefix: str = None) -> 'TrainerPieces':
        all_datasets = training_util.datasets_from_params(params, cache_directory, cache_prefix)
        datasets_for_vocab_creation = set(params.pop("datasets_for_vocab_creation", all_datasets))

        for dataset in datasets_for_vocab_creation:
            if dataset not in all_datasets:
                raise ConfigurationError(f"invalid 'dataset_for_vocab_creation' {dataset}")

        logger.info("From dataset instances, %s will be considered for vocabulary creation.",
                    ", ".join(datasets_for_vocab_creation))

        if recover and os.path.exists(os.path.join(serialization_dir, "vocabulary")):
            vocab = Vocabulary.from_files(os.path.join(serialization_dir, "vocabulary"))
            params.pop("vocabulary", {})
        else:
            vocab = Vocabulary.from_params(
                    params.pop("vocabulary", {}),
                    (instance for key, dataset in all_datasets.items()
                     for instance in dataset
                     if key in datasets_for_vocab_creation)
            )

        model = Model.from_params(vocab=vocab, params=params.pop('model'))

        # If vocab extension is ON for training, embedding extension should also be
        # done. If vocab and embeddings are already in sync, it would be a no-op.
        model.extend_embedder_vocab()

        # Initializing the model can have side effect of expanding the vocabulary
        vocab.save_to_files(os.path.join(serialization_dir, "vocabulary"))

        iterator = DataIterator.from_params(params.pop("iterator"))
        iterator.index_with(model.vocab)
        validation_iterator_params = params.pop("validation_iterator", None)
        if validation_iterator_params:
            validation_iterator = DataIterator.from_params(validation_iterator_params)
            validation_iterator.index_with(model.vocab)
        else:
            validation_iterator = None

        train_data = all_datasets['train']
        validation_data = all_datasets.get('validation')
        test_data = all_datasets.get('test')

        trainer_params = params.pop("trainer")
        no_grad_regexes = trainer_params.pop("no_grad", ())
        for name, parameter in model.named_parameters():
            if any(re.search(regex, name) for regex in no_grad_regexes):
                parameter.requires_grad_(False)

        frozen_parameter_names, tunable_parameter_names = \
                    get_frozen_and_tunable_parameter_names(model)
        logger.info("Following parameters are Frozen  (without gradient):")
        for name in frozen_parameter_names:
            logger.info(name)
        logger.info("Following parameters are Tunable (with gradient):")
        for name in tunable_parameter_names:
            logger.info(name)

        return TrainerPieces(model, iterator,
                             train_data, validation_data, test_data,
                             validation_iterator, trainer_params)
