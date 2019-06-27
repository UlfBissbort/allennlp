# pylint: disable=no-self-use
from typing import Dict, List, Optional
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer

logger = logging.getLogger(__name__)


@TokenIndexer.register("subword")
class SubwordIndexer(TokenIndexer[int]):
    def __init__(self,
                 tokenizer: Tokenizer,
                 subword_to_id: Dict[str, int],
                 offset_type: Optional[str] = None,
                 max_pieces: int = 512,
                 truncate_long_sequences: bool = True,
                 namespace: str = "subword") -> None:
        if offset_type not in [None, 'starting', 'ending']:
            raise ConfigurationError(f"unknown offset type: {offset_type}")
        self.tokenizer = tokenizer
        self.subword_to_id = subword_to_id
        self.offset_type = offset_type
        self.truncate_long_sequences = truncate_long_sequences
        self.max_pieces = max_pieces
        self._namespace = namespace
        self._added_to_vocabulary = False

        # self.vocab = BertTokenizer(pretrained_model).vocab
        # self._sep_id = self.vocab['[SEP]']
        # self._cls_id = self.vocab['[CLS]']

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        # If we only use pretrained models, we don't need to do anything here.
        pass

    def _add_encoding_to_vocabulary(self, vocabulary: Vocabulary) -> None:
        # pylint: disable=protected-access
        for word, idx in self.subword_to_id.items():
            vocabulary._token_to_index[self._namespace][word] = idx
            vocabulary._index_to_token[self._namespace][idx] = word

    @overrides
    def tokens_to_indices(self,
                          tokens: List[Token],
                          vocabulary: Vocabulary,
                          index_name: str) -> Dict[str, List[int]]:
        if not self._added_to_vocabulary:
            self._add_encoding_to_vocabulary(vocabulary)
            self._added_to_vocabulary = True

        # BertTokenizer attached the id to the token
        flat_wordpiece_ids = [token.text_id for token in tokens]

        # Similarly, we want to compute the token_type_ids from the flattened wordpiece ids before
        # we do the windowing; otherwise [SEP] tokens would get counted multiple times.
        flat_token_type_ids = []
        type_id = 0

        for wordpiece_id in flat_wordpiece_ids:
            flat_token_type_ids.append(type_id)
            type_id += wordpiece_id == self._sep_id

        # offsets[i] will give us the index into wordpiece_ids
        # for the wordpiece "corresponding to" the i-th input token.
        offsets = []

        # Find offsets:     [CLS, a, b, ##b, c, ##c, SEP]
        # starting offsets: [     1, 2,      4,       ]
        # ending offsets:   [     1,      3,      5,  ]
        # don't want offsets for CLS and SEP
        if self.offset_type == "starting":
            offsets = []
            for i in range(1, len(tokens)-1):
                if tokens[i].pos_ > tokens[i-1].pos_:
                    offsets.append(i)
        elif self.offset_type == "ending" or self.offset_type is None:
            # Compute offsets even for None because we need them for the mask
            offsets = []
            for i in range(1, len(tokens)-1):
                if tokens[i].pos_ < tokens[i+1].pos_:
                    offsets.append(i)

        if len(flat_wordpiece_ids) <= self.max_pieces:
            # If all the wordpieces fit, then we don't need to do anything special
            wordpiece_windows = [flat_wordpiece_ids]
            token_type_ids = [flat_token_type_ids]
        elif self._truncate_long_sequences:
            logger.warning("Too many wordpieces, truncating sequence. If you would like a sliding window, set"
                           "`truncate_long_sequences` to False %s", str([token.text for token in tokens]))
            wordpiece_windows = [flat_wordpiece_ids[:self.max_pieces-1] + [self._sep_id]]
            token_type_ids = [token_type_ids[:self.max_pieces-1] + [token_type_ids[self.max_pieces-1]]]
        else:
            raise NotImplementedError("sliding windows are not implemented")

        # Flatten the wordpiece windows
        wordpiece_ids = [wordpiece for sequence in wordpiece_windows for wordpiece in sequence]


        # Our mask should correspond to the original tokens,
        # because calling util.get_text_field_mask on the
        # "wordpiece_id" tokens will produce the wrong shape.
        # However, because of the max_pieces constraint, we may
        # have truncated the wordpieces; accordingly, we want the mask
        # to correspond to the remaining tokens after truncation, which
        # is captured by the offsets.
        mask = [1 for _ in offsets]

        indexed = {index_name: wordpiece_ids,
                   f"{index_name}-type-ids": token_type_ids,
                   "mask": mask}
        if self.offset_type is not None:
            indexed[f"{index_name}-offsets"] = offsets

        return indexed

    @overrides
    def get_padding_token(self) -> int:
        return 0

    @overrides
    def get_padding_lengths(self, token: int) -> Dict[str, int]:  # pylint: disable=unused-argument
        return {}

    @overrides
    def pad_token_sequence(self,
                           tokens: Dict[str, List[int]],
                           desired_num_tokens: Dict[str, int],
                           padding_lengths: Dict[str, int]) -> Dict[str, List[int]]:  # pylint: disable=unused-argument
        return {key: pad_sequence_to_length(val, desired_num_tokens[key])
                for key, val in tokens.items()}

    @overrides
    def get_keys(self, index_name: str) -> List[str]:
        """
        We need to override this because the indexer generates multiple keys.
        """
        # pylint: disable=no-self-use
        return [index_name, f"{index_name}-offsets", f"{index_name}-type-ids", "mask"]
