# pylint: disable=no-self-use,invalid-name,protected-access,bad-whitespace,bad-continuation
from allennlp.common.testing import ModelTestCase
from allennlp.data.token_indexers.bert_indexer import BertIndexer
from allennlp.data.tokenizers.bert_tokenizer import BertTokenizer
from allennlp.data.vocabulary import Vocabulary


class TestBertIndexer2(ModelTestCase):
    def test_starting_ending_offsets(self):
        vocab = Vocabulary()
        vocab_path = self.FIXTURES_ROOT / 'bert' / 'vocab.txt'

        #           2   3     5     6   8      9    2  15 10 11 14   1
        sentence = "the quick brown fox jumped over the laziest lazy elmo"
        tokenizer = BertTokenizer(str(vocab_path))
        tokens = tokenizer.tokenize(sentence)

        token_indexer = BertIndexer(str(vocab_path), offset_type="ending")
        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # 16 = [CLS], 17 = [SEP]
        assert indexed_tokens["bert"] == [16, 2, 3, 5, 6, 8, 9, 2, 15, 10, 11, 14, 1, 17]
        assert indexed_tokens["bert-offsets"] == [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]

        token_indexer = BertIndexer(str(vocab_path), offset_type="starting")
        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        assert indexed_tokens["bert"] == [16, 2, 3, 5, 6, 8, 9, 2, 15, 10, 11, 14, 1, 17]
        assert indexed_tokens["bert-offsets"] == [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]

        # No offsets
        token_indexer = BertIndexer(str(vocab_path))
        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")
        assert indexed_tokens["bert"] == [16, 2, 3, 5, 6, 8, 9, 2, 15, 10, 11, 14, 1, 17]
        assert "bert-offsets" not in indexed_tokens


    def test_do_lowercase(self):
        vocab = Vocabulary()
        vocab_path = self.FIXTURES_ROOT / 'bert' / 'vocab.txt'
        token_indexer = BertIndexer(str(vocab_path))

        # Quick is UNK because of capitalization
        #           2   1     5     6   8      9    2  15 10 11 14   1
        sentence = "the Quick brown fox jumped over the laziest lazy elmo"
        tokenizer = BertTokenizer(str(vocab_path))
        tokens = tokenizer.tokenize(sentence)
        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # Quick should get 1 == OOV
        assert indexed_tokens["bert"] == [16, 2, 1, 5, 6, 8, 9, 2, 15, 10, 11, 14, 1, 17]

        # Lowercase
        tokenizer = BertTokenizer(str(vocab_path), do_lower_case=True)
        tokens = tokenizer.tokenize(sentence)
        indexed_tokens = token_indexer.tokens_to_indices(tokens, vocab, "bert")

        # Now Quick should get indexed correctly as 3 ( == "quick")
        assert indexed_tokens["bert"] == [16, 2, 3, 5, 6, 8, 9, 2, 15, 10, 11, 14, 1, 17]
