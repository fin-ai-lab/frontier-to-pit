import os

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """Auto-mark everything under tests/gpu/ with the `gpu` marker."""
    for item in items:
        if "tests/gpu/" in str(item.fspath).replace(os.sep, "/"):
            item.add_marker(pytest.mark.gpu)


_CORPUS = [
    "The quick brown fox jumps over the lazy dog. ",
    "Pack my box with five dozen liquor jugs!\n",
    "In 2024, prices rose by 12.5% across 1,234 categories.\n",
    "def main():\n    return [i**2 for i in range(10)]\n",
    "Strawberries, blueberries, and raspberries grow strawberry fields.\n",
    "naïve café résumé — déjà vu, ‘quotes’ and “double quotes”.\n",
    "日本語のテキストと emoji 🚀🔥 mixed with English words.\n",
    "==== ----- ##### 0000000 aaaaaaa\n",
    "supercalifragilisticexpialidocious antidisestablishmentarianism\n",
]


def _train_tiny_bpe(vocab_size: int, specials: list[str], corpus_slice: slice):
    """A tiny byte-level BPE tokenizer trained in-process (no network)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(_CORPUS[corpus_slice] * 4, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token=specials[0], eos_token=specials[1]
    )


@pytest.fixture(scope="session")
def tiny_tok_p():
    """'P-side' tokenizer: byte-level BPE, vocab<=448, its own specials."""
    return _train_tiny_bpe(448, ["<s>", "</s>", "<|special|>"], slice(None))


@pytest.fixture(scope="session")
def tiny_tok_aux():
    """'aux-side' tokenizer: byte-level BPE with different merges/vocab/specials
    (trained on a different corpus slice), guaranteeing segmentations that
    genuinely differ from tiny_tok_p's."""
    return _train_tiny_bpe(320, ["<|begin|>", "<|end|>"], slice(0, 5))


@pytest.fixture(scope="session")
def tiny_tok_sp():
    """A SentencePiece-style tokenizer (▁ word markers + <0xNN> byte-fallback
    pieces) for TokenTextTable's SP branch."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    pieces = [("<unk>", 0.0), ("<s>", 0.0), ("</s>", 0.0)]
    pieces += [(f"<0x{b:02X}>", -10.0) for b in range(256)]
    words = ["▁the", "▁cat", "▁sat", "▁straw", "berry", "▁strawberry", "th", "e", "▁", "a"]
    pieces += [(w, -2.0) for w in words]
    pieces += [(c, -5.0) for c in "abcdefghijklmnopqrstuvwxyz.,!?"]
    tok = Tokenizer(models.Unigram(pieces, unk_id=0, byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )


@pytest.fixture(scope="session")
def tiny_llama_factory():
    """Factory for tiny random LlamaForCausalLM models (CPU, fp32, seeded).

    Small enough to run full ground-truth comparisons in CI without a GPU or
    network access; different seeds give genuinely different aux models."""
    from transformers import LlamaConfig, LlamaForCausalLM

    def make(seed: int = 0):
        torch.manual_seed(seed)
        cfg = LlamaConfig(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
        )
        return LlamaForCausalLM(cfg).eval()

    return make


@pytest.fixture(scope="session")
def tiny_flexllama_factory():
    """Factory for tiny random FlexLlama models (hybrid sliding/full attention,
    the v4 aux arch) with PRECOMPUTED rope caches randomized per seed — so a
    fused pair's planes carry genuinely different rope buffers, like the real
    flexdoc cooldown checkpoints."""
    from fixtures.flexllama import FlexLlamaConfig, FlexLlamaForCausalLM

    def make(seed: int = 0):
        torch.manual_seed(seed)
        cfg = FlexLlamaConfig(
            vocab_size=512,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            global_layer_indices=[1],  # layers 0,2,3 sliding; layer 1 full
            sliding_window=16,
            rope_precomputed=True,
            rope_cache_len=256,
        )
        m = FlexLlamaForCausalLM(cfg).eval()
        for rot in (m.model.rotary_emb, m.model.rotary_emb_local):
            rot.cos_cached.uniform_(-1.0, 1.0)
            rot.sin_cached.uniform_(-1.0, 1.0)
        return m

    return make


@pytest.fixture(scope="session")
def tiny_llama(tiny_llama_factory):
    return tiny_llama_factory(0)


@pytest.fixture(scope="session")
def tiny_llama_q(tiny_llama_factory):
    """A second tiny model with genuinely different weights (the 'q' of a
    fused pair); same architecture as `tiny_llama` by construction."""
    return tiny_llama_factory(1)
