
# melo/eval/eval_metric_batch.py
import os
import sys
import csv
import math
import argparse
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import librosa

# 패키지 경로 보강 (직접 실행 시)
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dmtts.utils.eval_utils import (  # noqa: E402
    get_metainfo,
    get_single_testset,
    make_save_path,
    run_asr_wer,   
    run_sim,       
    run_mos,
)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x



# ----- CLI -----

parser = argparse.ArgumentParser(description="Compute WER/CER/UTMOS and save per-language CSV")
parser.add_argument("--task", type=str, default="police_prompt", help="메타가 위치한 상위 폴더명")
parser.add_argument("--language", type=str, default="TH")
parser.add_argument("-cs", "--ckpt_steps", type=int, default=2095000) # vi: 613000
parser.add_argument("--speaker", required=False, help="생성에 사용된 싱글 스피커 이름 또는 ID (폴더명 매칭용)")
parser.add_argument("--out-root", default="synthesized_speech", help="합성물 루트 (infer와 동일)")
parser.add_argument("--gpus", default="0", help="GPU id 리스트, 예: '0' 또는 '0,1'")
#ap.add_argument("--metrics", default="cer",help="계산할 지표 콤마구분: wer,cer,utmos,sim")
parser.add_argument("--eval_task", default="mos", type=str, choices=["cer", "wer", "sim", "mos"])


#ap.add_argument("--asr-ckpt-dir", default="", help="ASR ckpt 디렉토리(빈 문자열이면 자동 다운로드/캐시)")
parser.add_argument("--prompt-dir", default="", help="SIM 계산 시 참조 음성 디렉토리(num.wav가 있어야 함)")
parser.add_argument("--result-dir", default="./result", help="CSV 저장 루트 디렉토리")
parser.add_argument("--eval_gt", default=False)

args = parser.parse_args()
gpus = args.gpus
lang = args.language
eval_gt = args.eval_gt
ckpt_step = args.ckpt_steps

local = False
if local:  # use local custom checkpoint dir
    asr_ckpt_dir = "../checkpoints/Systran/faster-whisper-large-v3"
else:
    asr_ckpt_dir = ""  # auto download to cache dir

wavlm_ckpt_dir = "../checkpoints/UniSpeech/wavlm_large_finetune.pth" # path to local




eval_task = args.eval_task
data_base_path = "/home/dev_admin/KKJ/DataSet/metadata/"
metalst_path = data_base_path + f"/{args.task}/{args.language}.lst"
print(f"metalst_path : {metalst_path}")
#exit()

# TODO : one of end two args.language is 'speaker' since the first trial was only single speaker
gen_wav_dir = f"synthesized_speech/model_{args.ckpt_steps}/{args.language}/{args.language}"

print(f"gen_wav_dir: {gen_wav_dir}")

test_set = get_single_testset(metalst_path, gen_wav_dir, gpus, language=args.language, eval_gt=eval_gt)


save_metric_dir = f"result/{args.language}_{args.language}" # need to change second {} to speaker
# --------------------------- SIM ---------------------------
# In Single-Speaker Task, no need
if eval_task in ["sim"]:
    sim_list = []

    with mp.Pool(processes=len(gpus)) as pool:
        args = [(rank, sub_test_set, wavlm_ckpt_dir) for (rank, sub_test_set) in test_set]
        results = pool.map(run_sim, args)
        for sim_ in results:
            sim_list.extend(sim_)

    sim = round(sum(sim_list) / len(sim_list), 3)

    if eval_gt:
        evaluation_setting = f"Ground Truth\n"
    else:
        evaluation_setting = f"Steps : {ckpt_step}\n"

    result_text = f"Total {len(sim_list)} samples\nSIM      : {sim}%\n\n"
    print(f"\nTotal {len(sim_list)} samples")
    print(f"SIM      : {sim}")

    with open(save_metric_dir,"a") as f:
        f.write(evaluation_setting)
        f.write(result_text)

# --------------------------- MOS ---------------------------

if eval_task in ["mos"]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    utmos_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    utmos_predictor = utmos_predictor.to(device)
    
    audio_paths = [gen_wav for gen_wav, _ in test_set[0][1]]
    
    utmos_results = {}
    utmos_score = 0

    for audio_path in tqdm(audio_paths, desc="Processing"):
        wav_name = Path(audio_path).stem
        #wav_name = audio_path.stem
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        wav_tensor = torch.from_numpy(wav).to(device).unsqueeze(0)

        min_length = 16000  # 최소 1초 길이 보장 (16kHz 기준)
        if wav_tensor.shape[-1] < min_length:
            wav_tensor = torch.nn.functional.pad(wav_tensor, (0, min_length - wav_tensor.shape[-1]))




        score = utmos_predictor(wav_tensor, sr)
        utmos_results[str(wav_name)] = score.item()
        utmos_score += score.item()

    avg_score = utmos_score / len(audio_paths) if len(audio_paths) > 0 else 0
    print(f"UTMOS: {avg_score}")


    if eval_gt:
        evaluation_setting = f"Ground Truth\n"
    else:
        evaluation_setting = f"Steps : {ckpt_step}\n"

    result_text = f"UTMOS      : {avg_score}\n\n"
    print(result_text)

    with open(save_metric_dir, "a") as f:
        f.write(evaluation_setting)
        f.write(result_text)
# --------------------------- WER ---------------------------

if eval_task in ["wer"]:
    wers = []
    cers = []

    with mp.Pool(processes=min(len(gpus), mp.cpu_count())) as pool:
        args = [(rank, lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
        results = pool.map(run_asr_wer, args)

        for wers_ in results[0][0]:
            wers.extend([wers_])
        for cers_ in results[0][1]:
            cers.extend([cers_])

    wer = round(np.mean(wers) * 100, 3)
    cer = round(np.mean(cers) * 100, 3)

    if eval_gt :
        evaluation_setting = f"Ground Truth\n"
    else :
        evaluation_setting = f"Steps : {ckpt_step}\n"
    result_text = f"Total {len(wers)} samples\nWER      : {wer}%\n\n"
    result_text = result_text +f"Total {len(cers)} samples\nCER      : {cer}%\n\n"
    print(result_text)

    with open(save_metric_dir,"a") as f:
        f.write(evaluation_setting)
        f.write(result_text)