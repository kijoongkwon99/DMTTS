# melo/infer_cli.py
import os
import sys
import argparse
from importlib.resources import files
from dmtts.utils.infer_utils import (
    _read_text, 
    _ensure_dir, 
    _resolve_ckpt_config,
    _resolve_speakers
)

# 스크립트 모드로 직접 실행해도 import 되도록 경로 보강
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dmtts.app.api import TTS  # noqa: E402

def parse_args():
    p = argparse.ArgumentParser(description="DMTTS CLI inference")
    p.add_argument("-v", "--version_of_model", type=int, default=1, help="Model version number (int)")
    p.add_argument("-lc", "--language_scope", choices=["mono", "multi"], default="mono")
    p.add_argument("-sc", "--speaker_scope", choices=["single", "multi"], default="single")
    p.add_argument("-l", "--language", type=str, choices=["JP", "KR", "EN", "VI", "ZH", "TH"], default="KR")
    p.add_argument("-cs", "--ckpt_steps", type=int)

    p.add_argument("--speed", type=float, default=1.0, help="Speech speed (0.1 ~ 10.0)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")

    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--list_speakers", type=bool, default=True)
    p.add_argument("--filename", default="{speaker}/output.wav",
                   help="Relative path template under output-dir. Supports {speaker}.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("-t", "--text", help="Input text")
    grp.add_argument("-f", "--text-file", help="Read text from file")
    grp.add_argument("--stdin", action="store_true", help="Read text from STDIN")

    return p.parse_args()


def main():
    args = parse_args()

    # 일본어 사용 시 유니딕 안내
    if args.language.upper() == "JP":
        print("Note: For Japanese, install UniDic and run: python -m unidic download")

    # 경로 결정 & 모델 로드
    #ckpt_path, config_path = _resolve_ckpt_config(args.language, args.weights_root, args.ckpt, args.config)
    
    
    #ckpt_path, config_path = _resolve_ckpt_config(args.language, args.version_of_model, args.ckpt_steps)
    ckpt_path, config_path = None, None
    if ckpt_path:
        model = TTS(language=args.language, device=args.device, config_path=config_path, ckpt_path=ckpt_path)
    else:
        # HF에서 자동 다운로드 사용
        model = TTS(language=args.language, device=args.device)

    if args.list_speakers:
        for name, sid in model.hps.data.spk2id.items():
            print(f"- {name}: {sid}")
            s_name, s_sid = name, sid


    text = _read_text(args)
    if not text:
        raise ValueError("No text provided. Use --text / --text-file / --stdin (or default not available).")

    targets = _resolve_speakers(model, s_name)
    out_root = os.path.abspath(args.output_dir)
    _ensure_dir(out_root)

    for spk_name, spk_id in targets:
        rel = args.filename.format(speaker=spk_name)
        save_path = os.path.join(out_root, rel)
        _ensure_dir(os.path.dirname(save_path))
        model.tts_to_file(text, spk_id, save_path, speed=args.speed) ## inference
        print(f"[OK] {spk_name:>12s} -> {save_path}")

    print("Done.")

if __name__ == "__main__":
    main()
