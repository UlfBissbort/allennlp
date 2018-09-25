# pylint: disable=invalid-name,no-self-use
import torch
import numpy as np

from allennlp.common.testing import AllenNlpTestCase
from allennlp.modules.seq2seq_encoders.contextual_encoder import ContextualEncoder
from allennlp.modules.token_embedders.cnn_highway_encoder import CnnHighwayEncoder

from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.token_embedders import Embedding

class TestContextualEncoder(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        self.character_encoder = CnnHighwayEncoder(
                activation='relu',
                embedding_dim=4,
                filters=[[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                max_characters_per_token=50,
                num_characters=262,
                num_highway=2,
                projection_dim=16,
                projection_location='after_cnn'
        )

        lstm = torch.nn.LSTM(bidirectional=True,
                             num_layers=3,
                             input_size=16,
                             hidden_size=10,
                             batch_first=True)
        self.seq2seq = PytorchSeq2SeqWrapper(lstm)


    def test_char_level_contextual_encoder(self):
        ce = ContextualEncoder(contextual_encoder=self.seq2seq,
                               num_layers=4,
                               return_all_layers=False)

        character_ids = torch.from_numpy(np.random.randint(0, 262, size=(5, 6, 50)))
        character_ids[0, 3:] = 0
        character_ids[1, -1, 40:] = 0
        char_mask = (character_ids > 0).long()
        mask = (char_mask.sum(dim=-1) > 0).long()
        embedding = self.character_encoder(character_ids)

        result = ce(embedding, mask)
        assert tuple(result.shape) == (5, 6, 20)

    def test_token_level_contextual_encoder(self):
        token_embedder = Embedding(num_embeddings=50, embedding_dim=16)

        ce = ContextualEncoder(contextual_encoder=self.seq2seq,
                               num_layers=3,
                               return_all_layers=False)

        token_ids = torch.from_numpy(np.random.randint(0, 50, size=(5, 6)))
        token_ids[0, 3:] = 0
        embedding = token_embedder(token_ids)
        mask = (token_ids > 0).long()
        result = ce(embedding, mask)
        assert tuple(result.shape) == (5, 6, 20)
