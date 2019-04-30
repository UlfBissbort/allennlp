{
    "dataset_reader": {
        "lazy": false,
        "type": "bert_classification_test",
        "tokenizer": {
            "word_splitter": "bert-basic"
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "allennlp/tests/fixtures/bert/vocab.txt"
            }
        }
    },
    "train_data_path": "doesn't matter, we're using precanned examples",
    "validation_data_path": "also doesn't matter",
    "model": {
        "type": "basic_classifier",
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets"],
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": "doesn't matter"
                }
            }
        },
        "seq2vec_encoder": {
           "type": "bert_pooler",
           "pretrained_model": "also doesn't matter"
        }
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": 5
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 3,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": -1
    }
}
