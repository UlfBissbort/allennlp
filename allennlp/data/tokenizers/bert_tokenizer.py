from typing import List

from pytorch_pretrained_bert.tokenization import BertTokenizer as PretrainedBertTokenizer
from overrides import overrides

from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer


@Tokenizer.register("bert")
class BertTokenizer(Tokenizer):
    def __init__(self,
                 pretrained_model: str,
                 do_lower_case: bool = False,
                 never_split: List[str] = None) -> None:
        print(do_lower_case)
        if never_split is None:
            self.bert_tokenizer = PretrainedBertTokenizer.from_pretrained(pretrained_model,
                                                                          do_lower_case=do_lower_case)
        else:
            self.bert_tokenizer = PretrainedBertTokenizer.from_pretrained(pretrained_model,
                                                                          do_lower_case=do_lower_case,
                                                                          never_split=never_split)
        self.vocab = self.bert_tokenizer.vocab

    def sep(self, idx: int = None) -> Token:
        return Token("[SEP]", idx=idx, text_id=self.vocab["[SEP]"])

    def cls(self, idx: int = None) -> Token:
        return Token("[CLS]", idx=idx, text_id=self.vocab["[CLS]"])

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        tokens = [self.cls(idx=0)]
        idx = 0
        for token in self.bert_tokenizer.tokenize(text):
            # TODO: how fragile is this logic
            if not token.startswith('##'):
                idx += 1
            tokens.append(Token(token, idx=idx, text_id=self.vocab[token]))
        idx += 1
        tokens.append(self.sep(idx=idx))
        return tokens

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [self.tokenize(text) for text in texts]
