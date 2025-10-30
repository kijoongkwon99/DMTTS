import os
from typing import List, Tuple, Optional

try:
    from tqdm import tqdm
except Exception:  # tqdm이 없으면 그냥 통과
    def tqdm(x, **kwargs):
        return x
import math
import os
import random
import string
import re

import torch
import torch.nn.functional as F
import torchaudio
from importlib.resources import files
from dmtts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL
import sys

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def lang_note_if_needed(language: str) -> None:
    if language.upper() == "JP":
        print("[Note] Japanese requires UniDic. Run: python -m unidic download")


def get_metainfo(metalst: str) -> List[Tuple[str, str, str]]:

    out: List[Tuple[str, str, str]] = []
    with open(metalst, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 3:
                continue
            num, lang, text = parts
            out.append((num, lang.upper(), text))
    return out


def select_single_speaker(model, speaker_opt: Optional[str]) -> Tuple[str, int]:
    """
    단일 스피커 선택: speaker_opt가 None이면 첫 화자 선택.
    speaker_opt가 '이름' 또는 '정수 ID' 모두 허용.
    반환: (speaker_name, speaker_id)
    """
    spk2id = getattr(model.hps.data, "spk2id", None)
    if not spk2id:
        raise RuntimeError("No speakers found in model (hps.data.spk2id empty).")

    if speaker_opt is None:
        name, sid = next(iter(spk2id.items()))
        return name, sid

    if speaker_opt in spk2id:
        return speaker_opt, spk2id[speaker_opt]

    try:
        sid = int(speaker_opt)
        name = next(n for n, v in spk2id.items() if v == sid)
        return name, sid
    except Exception:
        raise ValueError(f"Unknown speaker '{speaker_opt}'. Available: {list(spk2id.keys())}")


def make_save_path(out_root: str, step: str, language: str, speaker_name: str, num: str) -> str:
    """
    synthesized_speech/model_{step}/{language}/{speaker}/{num}.wav
    """
    path = os.path.join(
        os.path.abspath(out_root),
        f"model_{step}",
        language.upper(),
        speaker_name,
        f"{num}.wav",
    )
    ensure_dir(os.path.dirname(path))
    return path



def get_single_testset(metalst, gen_wav_dir, gpus, language, eval_gt=False, eval_num=100):
    if eval_gt:
        metalst = f"/home/dev_admin/KKJ/TTS-model/DMTTS/data/V1/{language}/metadata.list"
    f = open(metalst)
    lines= f.readlines()
    f.close()

    if eval_gt:
        random.seed(42)  # 재현성 원하면 유지 / 제거해도 됨
        lines = random.sample(lines, eval_num)

    test_set_ = []

    for line in tqdm(lines):
        if eval_gt:
            path, _, langauge, prompt_text, _, _ = line.strip().split("|")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Ground Truth wav not found: {path}")
            test_set_.append((path, prompt_text))
        else:
            num, language, prompt_text = line.strip().split("|")

            gen_utt = num + f".wav"
            gen_text = prompt_text


            if not os.path.exists(os.path.join(gen_wav_dir, gen_utt)):
                raise FileNotFoundError(f"Generated wav not found: {gen_wav_dir}/{gen_utt}")
            gen_wav = os.path.join(gen_wav_dir, gen_utt)


            test_set_.append((gen_wav, gen_text))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set        


def load_asr_model(lang, ckpt_dir=""):
    if lang == "zh":
        from funasr import AutoModel
        return AutoModel(model=os.path.join(ckpt_dir, "paraformer-zh"), disable_update=True)

    # en/ko 동일 처리: faster-whisper
    else:
        print("LOADING faster_whisper large-v3 ASR MODEL\n")
        from faster_whisper import WhisperModel

        model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    return model

# WER Evaluation, the way Seed-TTS does

def caculate_cer(truth, hypo):
    from jiwer import compute_measures
    measures = compute_measures(truth, hypo)
    total_chars = len(truth.replace(" ", "")) # exclude space
    if total_chars == 0:
        return 0.0

    cer = (measures["substitutions"] + measures["deletions"] + measures["insertions"]) / total_chars
    return cer

def create_asr_result(truth: str, hypo: str, dataset_name: str) -> str:
    os.makedirs("ASR_RESULT", exist_ok=True)

    # Define the results file path
    result_file = os.path.join("ASR_RESULT", f"{dataset_name}.txt")

    # If this is the first time writing, add a header
    write_header = not os.path.exists(result_file)

    # Open in append mode and write the entry
    with open(result_file, "a", encoding="utf-8") as fw:
        clean_truth = truth.replace("\n", " ").strip()
        clean_hypo  = hypo.replace("\n", " ").strip()
        fw.write(f"truth  : {clean_truth}\n")
        fw.write(f"hypo. : {clean_hypo}\n\n")

    #return result_file

def run_asr_wer(args):
    rank, lang, test_set, ckpt_dir = args
     
    lang = lang.lower()
    if lang == "zh":
        import zhconv
        rank = int(rank)
        torch.cuda.set_device(rank)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    """
    else:
        raise NotImplementedError(
            "lang support only 'zh' (funasr paraformer-zh), 'en' & 'ko' (faster-whisper-large-v3), for now."
        )
    """
    asr_model = load_asr_model(lang, ckpt_dir=ckpt_dir)
    print("fininsh loading asr_model")
    from zhon.hanzi import punctuation

    punctuation_all = punctuation + string.punctuation
    wers = []
    cers = []

    from jiwer import compute_measures

    for gen_wav, truth in tqdm(test_set):
        if lang == "zh":
            res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
            hypo = res[0]["text"]
            hypo = zhconv.convert(hypo, "zh-cn")
        # this part causing errors
        else:

            segments, _ = asr_model.transcribe(gen_wav, temperature=[0.0], vad_filter=True, condition_on_previous_text=False, beam_size=5) #temperature? 온도? -> 이게 들어가면 변형 생성..? -> 영어로 측정

            # specifically this part
            #hypo = segments["text"]
            hypo = " ".join(seg.text for seg in segments).strip()
            #print(f"hypo = {hypo}")
            #hypo = asr_model(gen_wav, return_timestamps=False)["text"]                       

        for x in punctuation_all:
            truth = truth.replace(x, "")
            hypo = hypo.replace(x, "")

        truth = truth.replace("  ", " ")
        hypo = hypo.replace("  ", " ")
        if lang == "zh":
            truth = " ".join([x for x in truth])
            hypo = " ".join([x for x in hypo])
        elif lang == "en":
            truth = truth.lower()
            hypo = hypo.lower()
        else:  
            truth = " ".join(truth.split())  
            hypo = " ".join(hypo.split())
        measures = compute_measures(truth, hypo)

        cer = caculate_cer(truth, hypo)
        wer = measures["wer"]

        print(f"truth : {truth}")
        print(f"hypo  : {hypo}")
        #create_asr_result(truth, hypo, dataset_name)
        wers.append(wer)
        cers.append(cer)

    return wers, cers

# SIM Evaluation
def run_sim(args):
    rank, test_set, ckpt_dir = args
    device = f"cuda:{rank}"
    print(f"device : {device}")
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt_dir, weights_only=True, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)

    use_gpu = True if torch.cuda.is_available() else False
    if use_gpu:
        model = model.cuda(device)
    model.eval()

    sim_list = []
    for wav1, wav2, truth in tqdm(test_set):
        print(truth)
        print(f"wav1 : {wav1}")
        print(f"wav2 : {wav2}")

        wav1, sr1 = torchaudio.load(wav1)
        wav2, sr2 = torchaudio.load(wav2)


        resample1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
        resample2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
        wav1 = resample1(wav1)
        wav2 = resample2(wav2)

        # 여기에서 길이 체크 및 패딩 필요!
        min_length = 16000  # 최소 1초

        if wav1.shape[-1] < min_length:
            pad_size = min_length - wav1.shape[-1]
            wav1 = F.pad(wav1, (0, pad_size))

        if wav2.shape[-1] < min_length:
            pad_size = min_length - wav2.shape[-1]
            wav2 = F.pad(wav2, (0, pad_size))



        if use_gpu:
            wav1 = wav1.cuda(device)
            wav2 = wav2.cuda(device)
        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)

        sim = F.cosine_similarity(emb1, emb2)[0].item()
        sim_list.append(sim)

    return sim_list

# MOS Evaluation
def run_mos(args):
    rank, sub_test_set = args
    mos_scores = []
    
    for sample in sub_test_set:
        audio_path = sample['gen_wav']  # Assuming 'gen_wav' holds the path to the generated audio
        audio, sr = torchaudio.load(audio_path)
        audio = audio.cuda()

        # Resampling to 16kHz
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).cuda()
        audio_16k = resampler(audio)

        # 여기에서 길이 체크 및 패딩 필요!
        min_length = 16000  # 최소 1초

        # 오디오 최소 길이 확인 후 패딩 추가 (최소 1초 보장)
        min_length = 16000  # 최소 1초 (16kHz 기준)
        if audio_16k.shape[-1] < min_length:
            audio_16k = torch.nn.functional.pad(audio_16k, (0, min_length - audio_16k.shape[-1]))



        # Predict MOS
        mos = utmos_predictor(audio_16k, 16000).item()
        mos_scores.append(mos)
    return mos_scores