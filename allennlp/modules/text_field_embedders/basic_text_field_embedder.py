from typing import Dict
import warnings

import torch
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TextFieldEmbedder.register("basic")
class BasicTextFieldEmbedder(TextFieldEmbedder):
    """
    This is a ``TextFieldEmbedder`` that wraps a collection of :class:`TokenEmbedder` objects.  Each
    ``TokenEmbedder`` embeds or encodes the representation output from one
    :class:`~allennlp.data.TokenIndexer`.  As the data produced by a
    :class:`~allennlp.data.fields.TextField` is a dictionary mapping names to these
    representations, we take ``TokenEmbedders`` with corresponding names.  Each ``TokenEmbedders``
    embeds its input, and the result is concatenated in an arbitrary order.

    Parameters
    ----------

    token_embedders : ``Dict[str, TokenEmbedder]``, required.
        A dictionary mapping token embedder names to implementations.
        These names should match the corresponding indexer used to generate
        the tensor passed to the TokenEmbedder.
    allow_unmatched_keys : ``bool``, optional (default = False)
        If True, then don't enforce the keys of the ``text_field_input`` to
        match those in ``token_embedders`` (useful if the mapping is specified
        via ``embedder_to_indexer_map``).
    """
    def __init__(self,
                 token_embedders: Dict[str, TokenEmbedder],
                 allow_unmatched_keys: bool = False) -> None:
        super(BasicTextFieldEmbedder, self).__init__()
        self._token_embedders = token_embedders
        self.expected_keys = {index_name
                              for embedder in token_embedders.values()
                              for index_name in embedder.index_names}

        for key, embedder in token_embedders.items():
            name = 'token_embedder_%s' % key
            self.add_module(name, embedder)
        self._allow_unmatched_keys = allow_unmatched_keys

    @overrides
    def get_output_dim(self) -> int:
        output_dim = 0
        for embedder in self._token_embedders.values():
            output_dim += embedder.get_output_dim()
        return output_dim

    def forward(self, text_field_input: Dict[str, torch.Tensor], num_wrapping_dims: int = 0) -> torch.Tensor:
        if self.expected_keys != set(text_field_input.keys()):
            if not self._allow_unmatched_keys:
                message = "Mismatched token keys: %s and %s" % (str(self._token_embedders.keys()),
                                                                str(text_field_input.keys()))
                raise ConfigurationError(message)
        embedded_representations = []
        keys = sorted(self._token_embedders.keys())
        for key in keys:
            embedder = getattr(self, 'token_embedder_{}'.format(key))
            tensors = [text_field_input[index_name] for index_name in embedder.index_names]

            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(embedder)
            token_vectors = embedder(*tensors)
            embedded_representations.append(token_vectors)
        return torch.cat(embedded_representations, dim=-1)

    # This is some unusual logic, it needs a custom from_params.
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'BasicTextFieldEmbedder':  # type: ignore
        # pylint: disable=arguments-differ
        embedder_to_indexer_map = params.pop("embedder_to_indexer_map", None)
        if embedder_to_indexer_map is not None:
            warnings.warn(DeprecationWarning("embedder_to_indexer_map is deprecated, please specify "
                                             "index_names explicitly for your token_embedders"))
            embedder_to_indexer_map = embedder_to_indexer_map.as_dict(quiet=True)
        allow_unmatched_keys = params.pop_bool("allow_unmatched_keys", False)

        token_embedders = {}
        keys = list(params.keys())
        for key in keys:
            embedder_params = params.pop(key)
            import logging
            logging.info(f"key {key} {embedder_params.as_dict()}")

            # Hack to add `index_names` to params if it's not there already.
            if 'index_names' not in embedder_params:
                if embedder_to_indexer_map and key in embedder_to_indexer_map:
                    embedder_params['index_names'] = embedder_to_indexer_map[key]
                else:
                    embedder_params['index_names'] = [key]

            token_embedders[key] = TokenEmbedder.from_params(vocab=vocab, params=embedder_params)
        params.assert_empty(cls.__name__)
        return cls(token_embedders, allow_unmatched_keys)
