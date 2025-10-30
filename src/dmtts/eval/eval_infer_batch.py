
import os
import sys
import argparse
from collections import defaultdict

# 패키지 경로 보강 (직접 실행 시)
if __name__ == "__main__" and __package__ is None:
    # ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    # sys.path.insert(0, ROOT)
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dmtts.app.api import TTS  # noqa: E402
#from melo.infer.infer_cli import _resolve_ckpt_config  
from dmtts.infer.infer_cli import _resolve_ckpt_config # noqa: E402
from dmtts.utils.eval_utils import (  # noqa: E402
    #get_melo_metainfo,
    get_metainfo,
    select_single_speaker,
    make_save_path,
    lang_note_if_needed,
)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def parse_args():
    p = argparse.ArgumentParser(description="DMTTS batch inference from meta list (num|language|text)")
    p.add_argument("--task", type=str, default="police_prompt")
    #p.add_argument("--metalst", required=True, help="메타리스트 경로 (num|language|text)")
    p.add_argument("--language", type=str, default="TH")
    p.add_argument("--speaker", required=False, help="싱글 스피커 이름 또는 정수 ID (미지정 시 첫 화자)")
    p.add_argument("--out-root", default="synthesized_speech", help="output directory root")
    p.add_argument("--speed", type=float, default=1.0, help="발화 속도 (0.1~10.0)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="device")

    # 가중치 지정 (step != 0 인 경우에만 의미 있음)
    p.add_argument("-cs", "--ckpt_steps", type=int, default=2095000) # VI: 613000
    p.add_argument("-v", "--version_of_model", type=int, default=1, help="Model version number (int)")

    # 기타
    p.add_argument("--resume", action="store_true", help="이미 존재하는 파일은 건너뜀")
    return p.parse_args()


def main():
    args = parse_args()

    use_pretrained = (args.ckpt_steps == 0)

    # 메타 로드

    data_base_path = "/home/dev_admin/KKJ/DataSet/metadata/"
    metalst_path = data_base_path + f"/{args.task}/{args.language}.lst"
    metainfo = get_metainfo(metalst_path)
    if not metainfo:
        raise ValueError("No valid lines in metalst (expected 'num|language|text').")

    # 언어별 모델/스피커 캐시
    model_cache = {}
    spk_cache = {}

    # 통계
    stat = defaultdict(int)

    # step 태그(폴더명). 원하면 "pretrained"로 바꾸고 싶다면 아래처럼:
    # step_tag = "pretrained" if use_pretrained else str(args.step)
    step_tag = str(args.ckpt_steps)

    for num, lang, text in tqdm(metainfo, desc="Synthesizing"):
        lang_note_if_needed(lang)

        # 모델 캐시
        if lang not in model_cache:

            # 사용자 지정 가중치 우선
            #ckpt_path, config_path = _resolve_ckpt_config(lang, args.weights_root, args.ckpt, args.config)
            ckpt_path, config_path = _resolve_ckpt_config(args.language, args.version_of_model, args.ckpt_steps)

            if ckpt_path:
                model_cache[lang] = TTS(language=lang, device=args.device, config_path=config_path, ckpt_path=ckpt_path)
            else:
                # 지정이 없으면 해당 언어의 기본(HF) 사용
                model_cache[lang] = TTS(language=lang, device=args.device)

        model = model_cache[lang]

        # 싱글 스피커 선택/캐시
        if lang not in spk_cache:
            spk_name, spk_id = select_single_speaker(model, args.speaker)
            spk_cache[lang] = (spk_name, spk_id)
        else:
            spk_name, spk_id = spk_cache[lang]

        # 저장 경로: synthesized_speech/model_{step}/{language}/{speaker}/{num}.wav
        save_path = make_save_path(args.out_root, step_tag, lang, spk_name, num)

        # resume
        if args.resume and os.path.exists(save_path):
            stat["skip"] += 1
            continue

        try:
            model.tts_to_file(text, spk_id, save_path, speed=args.speed)
            stat["ok"] += 1
        except Exception as e:
            stat["err"] += 1
            print(f"[ERR] {lang}:{spk_name}:{num} -> {type(e).__name__}: {e}")

    # 요약
    total = stat["ok"] + stat["skip"] + stat["err"]
    print("\n=== Summary ===")
    print(f"Total : {total}")
    print(f"OK    : {stat['ok']}")
    print(f"SKIP  : {stat['skip']}")
    print(f"ERROR : {stat['err']}")


if __name__ == "__main__":
    main()
