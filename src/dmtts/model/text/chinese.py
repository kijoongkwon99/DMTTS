import os
import re

import cn2an
from pypinyin import lazy_pinyin, Style

#from melo.text.symbols import punctuation
from dmtts.model.text.tone_sandhi import ToneSandhi

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

import jieba.posseg as psg
punctuation = [" ", "!", "?", "…", ",", ".", "'", "-"]

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}

tone_modifier = ToneSandhi()


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
    )
    return replaced_text


def g2p(text):
    """중국어 문장을 받아서 phones, tones만 반환"""
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones, tones = _g2p(sentences)
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    return phones, tones


def _get_initials_finals(word):
    initials = []
    finals = []
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


def _g2p(segments):
    phones_list = []
    tones_list = []
    for seg in segments:
        seg = re.sub("[a-zA-Z]+", "", seg)  # 영어 제거
        seg_cut = psg.lcut(seg)
        initials = []
        finals = []
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
        for word, pos in seg_cut:
            sub_initials, sub_finals = _get_initials_finals(word)
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            initials.append(sub_initials)
            finals.append(sub_finals)

        initials = sum(initials, [])
        finals = sum(finals, [])

        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            if c == v:  # 구두점 같은 케이스
                phone = [c]
                tone = "0"
            else:
                v_without_tone = v[:-1]
                tone = v[-1]
                pinyin = c + v_without_tone

                # pinyin 보정
                if c:
                    v_rep_map = {"uei": "ui", "iou": "iu", "uen": "un"}
                    if v_without_tone in v_rep_map:
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    pinyin_rep_map = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}
                    if pinyin in pinyin_rep_map:
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {"v": "yu", "e": "e", "i": "y", "u": "w"}
                        if pinyin[0] in single_rep_map:
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                assert pinyin in pinyin_to_symbol_map, (pinyin, seg, raw_pinyin)
                phone = pinyin_to_symbol_map[pinyin].split(" ")

            phones_list += phone
            tones_list += [int(tone)] * len(phone)

    return phones_list, tones_list


def text_normalize(text):
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    text = replace_punctuation(text)
    return text


if __name__ == "__main__":
    text = "我可以为您连接口译员。请稍等。"
    text = "请提供完整地址（包括门牌号/公寓号）。"
    text = text_normalize(text)
    print("normalized:", text)
    phones, tones = g2p(text)
    print(f"phones({len(phones)}):", phones)
    print(f"tones({len(tones)}) :", tones)
