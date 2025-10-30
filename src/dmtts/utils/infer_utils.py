import os
import sys
from dmtts.app.api import TTS 
from importlib.resources import files


DEFAULT_TEXT = {
    "EN": "The field of text to speech has seen rapid development recently.",
    "ZH": "text-to-speech 领域近年来发展迅速。",
    "JP": "テキスト読み上げの分野は最近急速な発展を遂げています。",
    "KR": "최근, 텍스트 음성 합성 분야가 급속도로 발전하고 있습니다.",
    "VI": "xâm hại tình dục trẻ em là vấn đề của toàn cầu",
    "TH": "ในช่วงหลังมานี้ เทคโนโลยีสังเคราะห์เสียงพูดได้พัฒนาอย่างรวดเร็ว"
}

def _read_text(args) -> str:
    if args.text:
        return args.text
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    if args.stdin:
        return sys.stdin.read().strip()
    # 미지정 시 언어별 기본 문장 사용
    return DEFAULT_TEXT.get(args.language.upper(), "")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _resolve_ckpt_config(language: str, version_of_model: int, ckpt_steps: int):
    # base path (rel_path)
    rel_path = os.path.abspath(os.path.join(str(files("dmtts")), "../../ckpts"))
    #rel_path = (Path(files("dmtts")) / "../../ckpts").resolve()

    
    lang = language.upper()
    model_dir = os.path.join(rel_path, "V"+str(version_of_model), lang)

    # checkpoint & config path
    ckpt_path = os.path.join(model_dir, f"G_{ckpt_steps}.pth")
    config_path = os.path.join(model_dir, "config.json")

    # 존재 여부 확인
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found: {config_path}")

    return ckpt_path, config_path




def _resolve_ckpt_config_orig(language: str, weights_root: str | None, ckpt: str | None, config: str | None):

    if ckpt:
        ckpt = os.path.abspath(ckpt)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        if config is None:
            config = os.path.join(os.path.dirname(ckpt), "config.json")
        config = os.path.abspath(config)
        if not os.path.isfile(config):
            raise FileNotFoundError(f"config.json not found next to ckpt: {config}")
        return ckpt, config

    if weights_root:
        weights_root = os.path.abspath(weights_root)
        lang = language.upper()
        # 1) <root>/<LANG> 구조
        cand_dir = os.path.join(weights_root, lang)
        cand_ckpt = os.path.join(cand_dir, "checkpoint.pth")
        cand_cfg  = os.path.join(cand_dir, "config.json")
        if os.path.isfile(cand_ckpt) and os.path.isfile(cand_cfg):
            return cand_ckpt, cand_cfg
        # 2) <root> 직하
        cand_ckpt = os.path.join(weights_root, "checkpoint.pth")
        cand_cfg  = os.path.join(weights_root, "config.json")
        if os.path.isfile(cand_ckpt) and os.path.isfile(cand_cfg):
            return cand_ckpt, cand_cfg
        raise FileNotFoundError(
            f"Could not find weights under {weights_root}. "
            f"Tried {os.path.join(weights_root, lang)} and {weights_root}."
        )

    # None -> HF 사용
    return None, None

def _resolve_speakers(model: TTS, speaker_opt):
    """speaker_opt: None(모든 화자), 'name', 'id' 또는 'a,b,c' 콤마구분"""
    spk2id = model.hps.data.spk2id  # dict[name] = id
    if speaker_opt is None:
        return list(spk2id.items())

    items = [s.strip() for s in speaker_opt.split(",")] if isinstance(speaker_opt, str) and "," in speaker_opt else [speaker_opt]
    picked = []
    for it in items:
        if it in spk2id:
            picked.append((it, spk2id[it]))
            continue
        try:
            sid = int(it)
            name = next(n for n, v in spk2id.items() if v == sid)
            picked.append((name, sid))
            continue
        except Exception:
            pass
        raise ValueError(f"Unknown speaker '{it}'. Available: {list(spk2id.keys())}")
    return picked