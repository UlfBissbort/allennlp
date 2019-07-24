# pylint: disable=no-self-use
from typing import Dict, List, Optional
import logging

from overrides import overrides
import torch
from pytorch_transformers.tokenization_utils import PreTrainedTokenizer
from pytorch_transformers.tokenization_bert import BertTokenizer

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

logger = logging.getLogger(__name__)

class PreTrainedTransformerIndexer(TokenIndexer[int]):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 offset_type: str = "starting",
                 max_pieces: Optional[int] = 512,
                 truncate_long_sequences: bool = True,
                 start_token: Optional[str] = None,
                 end_token: Optional[str] = None,
                 add_start_end_tokens: bool = True,
                 token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)
        self.tokenizer = tokenizer
        if offset_type not in ('starting', 'ending'):
            raise ConfigurationError("offsets must be 'starting' or 'ending' or None")
        self.offset_type = offset_type
        self.max_pieces = max_pieces

        # May be a thing or not
        self.sep_token = tokenizer.sep_token

        self.add_start_end_tokens = add_start_end_tokens
        self.start_token = start_token
        self.end_token = end_token

        # Helpers for sliding windows
        self.start_ids = self.tokenizer.encode(self.start_token) if self.start_token else []
        self.end_ids = self.tokenizer.encode(self.end_token) if self.end_token else []
        if self.max_pieces:
            self.window_size = self.max_pieces - len(self.start_ids) - len(self.end_ids)
            self.stride = self.window_size // 2
        else:
            self.window_size = self.stride = None

        self.truncate_long_sequences = truncate_long_sequences

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    def _compute_offsets(self, token_wordpiece_ids: List[List[int]]) -> List[int]:
        if self.offset_type == "starting":
            # start counter at 0, append, and increment
            offset = 0
            offsets = []
            for token_wpids in token_wordpiece_ids:
                offsets.append(offset)
                offset += len(token_wpids)
        elif self.offset_type == "ending":
            # start counter at -1, increment, and append
            offset = -1
            offsets = []
            for token_wpids in token_wordpiece_ids:
                offset += len(token_wpids)
                offsets.append(offset)
        else:
            offsets = None

        # Remove offsets corresponding to start token and end token
        if offsets and self.start_token:
            offsets.pop(0)
        if offsets and self.end_token:
            offsets.pop()

        return offsets

    def _compute_type_ids(self, tokens: List[Token], token_wordpiece_ids: List[List[int]]) -> List[int]:
        type_ids = []
        type_id = 0
        for token, token_wpids in zip(tokens, token_wordpiece_ids):
            type_ids.extend(type_id for _ in token_wpids)
            if token.text == self.sep_token:
                type_id += 1

        return type_ids

    def _remove_start_and_end(self, wpids: List[int]) -> List[int]:
        # remove start_ids
        wpids = wpids[len(self.start_ids):]
        for _ in self.end_ids:
            wpids.pop()
        return wpids

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        # pylint: disable=unused-argument
        # tokens are one per word, so this is a list per word
        if self.start_token:
            tokens = [Token(self.start_token)] + tokens
        if self.end_token:
            tokens.append(Token(self.end_token))

        token_wordpiece_ids = [self.tokenizer.encode(token.text) for token in tokens]

        # Compute offsets
        offsets = self._compute_offsets(token_wordpiece_ids)

        # Compute token types
        type_ids = self._compute_type_ids(tokens, token_wordpiece_ids)

        # Handle truncation
        total_length = sum(len(wpids) for wpids in token_wordpiece_ids)
        if self.max_pieces is not None and total_length > self.max_pieces and self.truncate_long_sequences:
            truncated_tokens = []
            wordpiece_ids = []

            current_length = 0
            for i, twpids in enumerate(token_wordpiece_ids):
                if current_length + len(twpids) + len(self.end_ids) > self.max_pieces:
                    break
                else:
                    truncated_tokens.append(tokens[i])
                    wordpiece_ids.extend(twpids)
                    current_length += len(twpids)

            # Add end ids
            if self.end_token:
                truncated_tokens.append(tokens[-1])
                wordpiece_ids.extend(self.end_ids)

            # truncate tokens, offsets, and type_ids
            tokens = truncated_tokens
            new_offsets_length = len(tokens) - (1 if self.start_ids else 0) - (1 if self.end_ids else 0)
            offsets = offsets[:new_offsets_length]
            type_ids = type_ids[:len(wordpiece_ids)]

        # handle sliding window
        elif self.max_pieces is not None and total_length > self.max_pieces:
            # TODO: implement sliding windows
            raise RuntimeError("not implemented (and impossible to get here)")

        else:
            # just flatten
            wordpiece_ids = [wpid for token_wpids in token_wordpiece_ids for wpid in token_wpids]


        # Our mask should correspond to the original tokens,
        # because calling util.get_text_field_mask on the
        # "wordpiece_id" tokens will produce the wrong shape.
        # However, because of the max_pieces constraint, we may
        # have truncated the wordpieces; accordingly, we want the mask
        # to correspond to the remaining tokens after truncation, which
        # is captured by the offsets.
        mask = [1 for _ in offsets]

        return {"wordpiece_ids": wordpiece_ids, "mask": mask, "offsets": offsets, "type_ids": type_ids}

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def as_padded_tensor(self,
                         tokens: Dict[str, List[int]],
                         desired_num_tokens: Dict[str, int],
                         padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:  # pylint: disable=unused-argument
        return {key: torch.LongTensor(pad_sequence_to_length(val, desired_num_tokens[key]))
                for key, val in tokens.items()}


    @overrides
    def get_keys(self, index_name: str) -> List[str]:
        """
        We need to override this because the indexer generates multiple keys.
        """
        # pylint: disable=no-self-use,unused-argument
        return ["wordpiece_ids", "offsets", "type_ids", "mask"]

@TokenIndexer.register("bert-transformer-pretrained")
class PretrainedBertIndexer(PreTrainedTransformerIndexer):
    # pylint: disable=line-too-long
    """
    A ``TokenIndexer`` corresponding to a pretrained BERT model.

    Parameters
    ----------
    pretrained_model: ``str``
        Either the name of the pretrained model to use (e.g. 'bert-base-uncased'),
        or the path to the .txt file with its vocabulary.

        If the name is a key in the list of pretrained models at
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py#L33
        the corresponding path will be used; otherwise it will be interpreted as a path or URL.
    use_starting_offsets: bool, optional (default: False)
        By default, the "offsets" created by the token indexer correspond to the
        last wordpiece in each word. If ``use_starting_offsets`` is specified,
        they will instead correspond to the first wordpiece in each word.
    max_pieces: int, optional (default: 512)
        The BERT embedder uses positional embeddings and so has a corresponding
        maximum length for its input ids. Any inputs longer than this will
        either be truncated (default), or be split apart and batched using a
        sliding window.
    truncate_long_sequences : ``bool``, optional (default=``True``)
        By default, long sequences will be truncated to the maximum sequence
        length. Otherwise, they will be split apart and batched using a
        sliding window.
    """

    def __init__(self,
                 pretrained_model: str,
                 add_start_end_tokens: bool = True,
                 use_starting_offsets: bool = False,
                 do_lowercase: bool = True,
                 max_pieces: int = 512,
                 truncate_long_sequences: bool = True) -> None:
        if pretrained_model.endswith("-cased") and do_lowercase:
            logger.warning("Your BERT model appears to be cased, "
                           "but your indexer is lowercasing tokens.")
        elif pretrained_model.endswith("-uncased") and not do_lowercase:
            logger.warning("Your BERT model appears to be uncased, "
                           "but your indexer is not lowercasing tokens.")

        bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=do_lowercase)

        if add_start_end_tokens:
            start_token = bert_tokenizer.cls_token
            end_token = bert_tokenizer.sep_token
        else:
            start_token, end_token = None, None
        super().__init__(tokenizer=bert_tokenizer,
                         offset_type="starting" if use_starting_offsets else "ending",
                         max_pieces=max_pieces,
                         start_token=start_token,
                         end_token=end_token,
                         add_start_end_tokens=add_start_end_tokens,
                         truncate_long_sequences=truncate_long_sequences)
