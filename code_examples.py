from transformers import AutoTokenizer, BertConfig, BertModel
import json
from torch import tensor

from src.multiTrans import TulipPetal, BertLastPooler, Tulip


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("aatok")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '<MIS>'})

    if tokenizer.cls_token is None:
        tokenizer.add_special_tokens({'cls_token': '<CLS>'})

    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '<EOS>'})

    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '<MASK>'})

    from tokenizers.processors import TemplateProcessing
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="<CLS> $A <EOS>",
        pair="<CLS> $A <MIS> $B:1 <EOS>:1",
        special_tokens=[
            ("<EOS>", 2),
            ("<CLS>", 3),
            ("<MIS>", 4),
        ],
    )
    return tokenizer

def get_tulip(tokenizer, mhctok) -> Tulip:
    with open("configs/shallow.config.json", "r") as read_file:
        modelconfig = json.load(read_file)
    max_length = 50
    vocabsize = len(tokenizer._tokenizer.get_vocab())
    encoder_config = BertConfig(vocab_size=vocabsize,
                                max_position_embeddings=max_length,  # this shuold be some large value
                                num_attention_heads=modelconfig["num_attn_heads"],
                                num_hidden_layers=modelconfig["num_hidden_layers"],
                                hidden_size=modelconfig["hidden_size"],
                                type_vocab_size=1,
                                pad_token_id=tokenizer.pad_token_id)

    mhcvocabsize = len(mhctok._tokenizer.get_vocab())
    encoder_config.mhc_vocab_size = mhcvocabsize

    encoderA = BertModel(config=encoder_config)
    encoderB = BertModel(config=encoder_config)
    encoderE = BertModel(config=encoder_config)

    max_length = 50
    decoder_config = BertConfig(vocab_size=vocabsize,
                                max_position_embeddings=max_length,  # this shuold be some large value
                                num_attention_heads=modelconfig["num_attn_heads"],
                                num_hidden_layers=modelconfig["num_hidden_layers"],
                                hidden_size=modelconfig["hidden_size"],
                                type_vocab_size=1,
                                is_decoder=True,
                                pad_token_id=tokenizer.pad_token_id)  # Very Important

    decoder_config.add_cross_attention = True

    decoderA = TulipPetal(config=decoder_config)  # BertForMaskedLM
    decoderA.pooler = BertLastPooler(config=decoder_config)
    decoderB = TulipPetal(config=decoder_config)  # BertForMaskedLM
    decoderB.pooler = BertLastPooler(config=decoder_config)
    decoderE = TulipPetal(config=decoder_config)  # BertForMaskedLM
    decoderE.pooler = BertLastPooler(config=decoder_config)

    # Define encoder decoder model

    return Tulip(encoderA=encoderA, encoderB=encoderB, encoderE=encoderE, decoderA=decoderA, decoderB=decoderB,
                  decoderE=decoderE)


def test_sample_chain_de_novo():
    from src.multiTrans import sample_chain_de_novo, Tulip


    tokenizer = get_tokenizer()
    mhctok = AutoTokenizer.from_pretrained("mhctok/")
    model: Tulip = get_tulip(tokenizer=tokenizer, mhctok=mhctok)
    example_starting_batch = ({'input_ids': tensor([[3, 12, 19, 13, 2]]),
                               'token_type_ids': tensor([[0, 0, 0, 0, 0]]),
                               'attention_mask': tensor([[1, 1, 1, 1, 1]])},
                              {'input_ids': tensor([[3, 4, 2]]),
                               'token_type_ids': tensor([[0, 0, 0]]),
                               'attention_mask': tensor([[1, 1, 1]])},
                              {'input_ids': tensor([[3, 4, 2]]),
                               'token_type_ids': tensor([[0, 0, 0]]),
                               'attention_mask': tensor([[1, 1, 1]])},
                              tensor([0]),
                              {'input_ids': tensor([[1]]),
                               'token_type_ids': tensor([[0]]),
                               'attention_mask': tensor([[1]])})

    sample_chain_de_novo(model=model, tokenizer=tokenizer, mhctok=mhctok, starting_batch=example_starting_batch, peptide_str="ABC", num_return_sequences=1, mode="sampling", temperature=1.0)
    return 4