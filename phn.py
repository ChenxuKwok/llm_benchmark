import logging
import warnings
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, WavLMForCTC
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

warnings.filterwarnings("ignore", message=r".*words count mismatch.*")
logging.getLogger("phonemizer").setLevel(logging.ERROR)


class G2PProcessor:
    def __init__(self, language='en-us', with_stress=False):
        self.backend = EspeakBackend(
            language=language,
            with_stress=with_stress,
            preserve_punctuation=True,
            words_mismatch='ignore'
        )

        self.separator = Separator(
            phone=' ',
            word=',',
            syllable='|'
        )

    def batch_process(self, texts):
        if not texts:
            return []
        return self.backend.phonemize(
            texts,
            separator=self.separator,
            strip=True
        )

    def process(self, text):
        # 保持和你原来接口尽量一致：str 输入时仍然包装成 list
        if isinstance(text, str):
            text = [text]
        return self.batch_process(text)


class Wav2Vec2PRModel:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft").to(self.device)
        self.model.eval()

    def batch_process(self, audio_arrays, sr=16000, batch_size=16):
        if not audio_arrays:
            return []

        outputs = [""] * len(audio_arrays)
        valid_indices = []
        valid_audios = []

        for i, audio in enumerate(audio_arrays):
            if audio is None or len(audio) == 0:
                continue
            valid_indices.append(i)
            valid_audios.append(np.asarray(audio, dtype=np.float32))

        for start in range(0, len(valid_audios), batch_size):
            batch_audio = valid_audios[start:start + batch_size]
            batch_indices = valid_indices[start:start + batch_size]

            inputs = self.processor(
                batch_audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                logits = self.model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            decoded = self.processor.batch_decode(predicted_ids)

            for idx, text in zip(batch_indices, decoded):
                outputs[idx] = text

        return outputs

    def process(self, audio_array, sr=16000):
        return self.batch_process([audio_array], sr=sr, batch_size=1)[0]


class HuPERPRModel:
    def __init__(self, device="cuda:0"):
        self.device = device
        repo_id = "huper29/huper_recognizer"

        self.processor = Wav2Vec2Processor.from_pretrained(repo_id)
        self.model = WavLMForCTC.from_pretrained(repo_id).to(self.device)
        self.model.eval()

        self.blank_id = self.processor.tokenizer.pad_token_id

    def _decode_one(self, pred_ids):
        phone_tokens = []
        prev = None
        for token_id in pred_ids:
            if token_id != self.blank_id and token_id != prev:
                token = self.model.config.id2label.get(
                    token_id,
                    self.processor.tokenizer.convert_ids_to_tokens(token_id)
                )
                if token not in {"<PAD>", "<UNK>", "<BOS>", "<EOS>", "|"}:
                    phone_tokens.append(token)
            prev = token_id

        return " ".join(phone_tokens)

    def batch_process(self, audio_arrays, sr=16000, batch_size=16):
        if not audio_arrays:
            return []

        outputs = [""] * len(audio_arrays)
        valid_indices = []
        valid_audios = []

        for i, audio in enumerate(audio_arrays):
            if audio is None or len(audio) == 0:
                continue
            valid_indices.append(i)
            valid_audios.append(np.asarray(audio, dtype=np.float32))

        for start in range(0, len(valid_audios), batch_size):
            batch_audio = valid_audios[start:start + batch_size]
            batch_indices = valid_indices[start:start + batch_size]

            inputs = self.processor(
                batch_audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                logits = self.model(**inputs).logits

            pred_ids_batch = torch.argmax(logits, dim=-1).tolist()
            decoded = [self._decode_one(pred_ids) for pred_ids in pred_ids_batch]

            for idx, text in zip(batch_indices, decoded):
                outputs[idx] = text

        return outputs

    def process(self, audio_array, sr=16000):
        return self.batch_process([audio_array], sr=sr, batch_size=1)[0]