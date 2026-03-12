import os
import sys
import json
import logging
import warnings
import traceback
import argparse
import numpy as np
import torch
import torchaudio
import torch.multiprocessing as mp
import tqdm

# Append custom path
sys.path.append('/root/omni_pr')

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ASR and Aligners
from qwen_asr import Qwen3ASRModel
from utils.OSSAudioLoader import OSSAudioLoader, process_json_and_download
from utils.SentenceTimestampAligner import SentenceTimestampAligner

# Phone Recognition Models
from utils.phone import G2PProcessor
from utils.phone import Wav2Vec2PRModel
from utils.phone import HuPERPRModel


def configure_worker_logging():
    # Python warnings
    warnings.filterwarnings("ignore", message=r".*pad_token_id.*eos_token_id.*")
    warnings.filterwarnings("ignore", message=r".*words count mismatch.*")
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Logger-based warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("phonemizer").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass


def try_silence_generation_warning(asr_model):
    """
    有些 wrapper 会把 HF model 挂在 model / llm / backbone 里。
    能设上 pad_token_id 的话，就尽量设掉，避免生成 warning。
    """
    candidate_attrs = ["model", "llm", "backbone"]
    for attr in candidate_attrs:
        obj = getattr(asr_model, attr, None)
        if obj is None:
            continue

        generation_config = getattr(obj, "generation_config", None)
        if generation_config is not None:
            eos_id = getattr(generation_config, "eos_token_id", None)
            if eos_id is not None and getattr(generation_config, "pad_token_id", None) is None:
                generation_config.pad_token_id = eos_id

        tokenizer = getattr(obj, "tokenizer", None)
        if tokenizer is not None:
            if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id


def parse_gpu_ids(gpus_arg: str):
    if gpus_arg:
        return [int(x.strip()) for x in gpus_arg.split(",") if x.strip() != ""]
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("No CUDA devices found.")
    return list(range(n))


def count_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def build_sentence_records(
    waveform_array,
    sample_rate,
    sentence_timestamps,
    g2p_model,
    w2v2_model,
    huper_model,
    pr_batch_size,
):
    texts = []
    spans = []
    audio_segments = []

    for sentence in sentence_timestamps:
        start_time = float(sentence["start_time"])
        end_time = float(sentence["end_time"])
        text = sentence["text"]

        start_frame = max(0, int(start_time * sample_rate))
        end_frame = min(len(waveform_array), int(end_time * sample_rate))

        if end_frame <= start_frame:
            audio_segment = np.zeros(160, dtype=np.float32)
        else:
            audio_segment = waveform_array[start_frame:end_frame].astype(np.float32, copy=False)

        texts.append(text)
        spans.append((start_time, end_time, text))
        audio_segments.append(audio_segment)

    # 这里是关键：按句子 batch 做 phone inference
    phonemes_g2p = g2p_model.batch_process(texts)
    phonemes_wav2vec2 = w2v2_model.batch_process(audio_segments, sr=sample_rate, batch_size=pr_batch_size)
    phonemes_huper = huper_model.batch_process(audio_segments, sr=sample_rate, batch_size=pr_batch_size)

    records = []
    for i, (start_time, end_time, text) in enumerate(spans):
        records.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
                "phonemes_g2p": phonemes_g2p[i],
                "phonemes_wav2vec2": phonemes_wav2vec2[i],
                "phonemes_huper": phonemes_huper[i],
            }
        )
    return records


def gpu_worker(worker_rank, gpu_id, task_queue, result_queue, pr_batch_size):
    configure_worker_logging()
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu_id)

    device = f"cuda:{gpu_id}"
    print(f"[Worker {worker_rank}] Initializing models on {device} ...", flush=True)

    # Initialize ASR model
    asr_model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=32,
        max_new_tokens=1024,
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map=device,
        ),
    )
    try_silence_generation_warning(asr_model)

    # Initialize PR models
    g2p_model = G2PProcessor()
    w2v2_model = Wav2Vec2PRModel(device=device)
    huper_model = HuPERPRModel(device=device)

    loader = OSSAudioLoader()
    aligner = SentenceTimestampAligner()

    processed = 0

    while True:
        item = task_queue.get()
        if item is None:
            break

        idx, line_str = item
        wav_path = None

        try:
            json_line = json.loads(line_str)
            original_audio_uri = json_line["messages"][1]["content"][0]["audio"]

            # Download audio
            wav_path = process_json_and_download(json_line, loader)

            # ASR inference
            results = asr_model.transcribe(
                audio=wav_path,
                language=None,
                return_time_stamps=True
            )
            full_text = results[0].text
            items = results[0].time_stamps.items

            # Align timestamps
            sentence_timestamps = aligner.align(full_text, items)

            # Load audio
            waveform, sample_rate = torchaudio.load(wav_path)

            # 读完立刻删临时文件
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
                wav_path = None

            # 转单声道
            if waveform.dim() == 2:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to 16k
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                sample_rate = 16000

            waveform_array = waveform.squeeze(0).contiguous().numpy().astype(np.float32, copy=False)

            result_record = {
                "original_audio": original_audio_uri,
                "records": build_sentence_records(
                    waveform_array=waveform_array,
                    sample_rate=sample_rate,
                    sentence_timestamps=sentence_timestamps,
                    g2p_model=g2p_model,
                    w2v2_model=w2v2_model,
                    huper_model=huper_model,
                    pr_batch_size=pr_batch_size,
                ),
            }

            result_queue.put(
                {
                    "idx": idx,
                    "ok": True,
                    "line": json.dumps(result_record, ensure_ascii=False),
                }
            )

        except Exception as e:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)

            result_queue.put(
                {
                    "idx": idx,
                    "ok": False,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
            )

        processed += 1
        if processed % 20 == 0:
            print(f"[Worker {worker_rank} | GPU {gpu_id}] processed {processed} samples", flush=True)

    result_queue.put(None)
    print(f"[Worker {worker_rank}] finished.", flush=True)


def writer_worker(output_jsonl, error_jsonl, result_queue, total_tasks, num_workers):
    """
    单独 writer 进程，避免多进程写文件冲突。
    同时按 idx 重新排序，保证输出顺序和输入一致。
    """
    next_idx = 0
    finished_workers = 0
    buffer = {}

    with open(output_jsonl, "w", encoding="utf-8") as fout, open(error_jsonl, "w", encoding="utf-8") as ferr:
        with tqdm.tqdm(total=total_tasks, desc="Overall") as pbar:
            while finished_workers < num_workers:
                item = result_queue.get()

                if item is None:
                    finished_workers += 1
                    continue

                buffer[item["idx"]] = item

                while next_idx in buffer:
                    cur = buffer.pop(next_idx)
                    if cur["ok"]:
                        fout.write(cur["line"] + "\n")
                    else:
                        ferr.write(
                            json.dumps(
                                {
                                    "idx": next_idx,
                                    "error": cur["error"],
                                    "traceback": cur["traceback"],
                                },
                                ensure_ascii=False,
                            ) + "\n"
                        )
                    next_idx += 1
                    pbar.update(1)

            # 理论上这里 buffer 应该已经清空；保底再刷一次
            while next_idx in buffer:
                cur = buffer.pop(next_idx)
                if cur["ok"]:
                    fout.write(cur["line"] + "\n")
                else:
                    ferr.write(
                        json.dumps(
                            {
                                "idx": next_idx,
                                "error": cur["error"],
                                "traceback": cur["traceback"],
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    )
                next_idx += 1
                pbar.update(1)

        fout.flush()
        ferr.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, default="data/en_data.jsonl")
    parser.add_argument("--output_jsonl", type=str, default="data/en_processed.jsonl")
    parser.add_argument("--error_jsonl", type=str, default="data/en_processed.errors.jsonl")
    parser.add_argument("--gpus", type=str, default=None, help='例如 "0,1,2,3"，默认使用全部可见 GPU')
    parser.add_argument("--pr_batch_size", type=int, default=16)
    parser.add_argument("--queue_size", type=int, default=64)
    args = parser.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus)
    num_workers = len(gpu_ids)

    print(f"Using GPUs: {gpu_ids}", flush=True)

    total_tasks = count_lines(args.input_jsonl)

    task_queue = mp.Queue(maxsize=args.queue_size)
    result_queue = mp.Queue(maxsize=args.queue_size)

    # Writer process
    writer = mp.Process(
        target=writer_worker,
        args=(args.output_jsonl, args.error_jsonl, result_queue, total_tasks, num_workers),
    )
    writer.start()

    # GPU workers
    workers = []
    for worker_rank, gpu_id in enumerate(gpu_ids):
        p = mp.Process(
            target=gpu_worker,
            args=(worker_rank, gpu_id, task_queue, result_queue, args.pr_batch_size),
        )
        p.start()
        workers.append(p)

    # 主进程流式读入，不把整个 jsonl 一次性塞进内存
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for idx, line_str in enumerate(f):
            task_queue.put((idx, line_str))

    # 发送结束信号
    for _ in workers:
        task_queue.put(None)

    # 等待 worker
    for p in workers:
        p.join()

    # 等待 writer
    writer.join()

    print(f"Done. Results saved to {args.output_jsonl}", flush=True)
    print(f"Errors saved to {args.error_jsonl}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()