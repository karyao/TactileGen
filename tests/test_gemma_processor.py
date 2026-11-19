"""Real offline processor loading; requires the optional VLM installation."""

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from src.baseline.gemma import GemmaChartExtractor


# Skip only a wholly absent VLM stack. A partial/broken installation must fail.
VLM_ABSENT = all(importlib.util.find_spec(name) is None
                 for name in ("torch", "transformers", "sentencepiece"))


def make_processor_fixture(directory):
    """Build synthetic local assets without gated checkpoints or model weights."""
    import sentencepiece as spm
    from transformers import Gemma3ImageProcessor, Gemma3Processor, GemmaTokenizer

    directory = Path(directory)
    prefix = directory / "tokenizer"
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["hello chart", "chart values", "hello values"]),
        model_prefix=str(prefix), vocab_size=32, hard_vocab_limit=False,
        model_type="unigram", unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
        pad_piece="<pad>", minloglevel=2,
    )
    tokenizer = GemmaTokenizer(
        vocab_file=str(prefix.with_suffix(".model")),
        extra_special_tokens={"image_token": "<image_soft_token>",
                              "boi_token": "<start_of_image>",
                              "eoi_token": "<end_of_image>"},
    )
    processor = Gemma3Processor(
        image_processor=Gemma3ImageProcessor(size={"height": 28, "width": 28}),
        tokenizer=tokenizer,
    )
    # Save separately so the tokenizer's source file is not its destination.
    checkpoint = directory / "checkpoint"
    processor.save_pretrained(checkpoint)
    return checkpoint


@unittest.skipIf(VLM_ABSENT, "Install requirements-vlm.txt to test real processor loading")
class GemmaProcessorTest(unittest.TestCase):
    def test_extractor_initializes_real_slow_processor(self):
        import torch
        from transformers import AutoProcessor, Gemma3Processor, GemmaTokenizer

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = make_processor_fixture(directory)
            extractor = GemmaChartExtractor(model_id=str(checkpoint), device="cpu")
            model = Mock(config=SimpleNamespace(_commit_hash=None))
            model_class = Mock()
            model_class.from_pretrained.return_value = model

            # Only model loading is mocked: AutoProcessor and both its components
            # must initialize through the same path and kwargs as production.
            extractor._ensure_loaded(torch, AutoProcessor, model_class, {})
            processor = extractor._processor
            self.assertIsInstance(processor, Gemma3Processor)
            self.assertIsInstance(processor.tokenizer, GemmaTokenizer)
            self.assertFalse(processor.tokenizer.is_fast)
            self.assertEqual(processor.image_token_id, processor.tokenizer.image_token_id)
            tokens = processor(text="hello chart", return_tensors="pt")
            self.assertGreater(tokens["input_ids"].numel(), 0)
            self.assertEqual(processor.decode(tokens["input_ids"][0], skip_special_tokens=True),
                             "hello chart")
            model_class.from_pretrained.assert_called_once_with(
                str(checkpoint), revision=None, torch_dtype=torch.float32)
            model.to.assert_called_once_with("cpu")
            model.eval.assert_called_once_with()
