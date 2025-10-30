from importlib import import_module
from functools import lru_cache
import dmtts.model.text.symbols as symbols
from dmtts.model.text.symbols import cleaned_text_to_sequence
import copy

@lru_cache(maxsize=None)
def get_language_module(language_code: str):
    """symbols.LANG 안의 module 경로 문자열로 실제 모듈을 lazy import."""
    #code = _canonical(language_code)
    info = symbols.LANG.get(language_code)
    if not info:
        raise ValueError(f"Unknown language code: {language_code}")
    modpath = info.get("module")
    if not modpath:
        # 모듈 없는 언어는 지원 안 함
        return None
    return import_module(modpath)

def clean_text(text: str, language: str):
    language_module= get_language_module(language)
    if language_module is None:
        raise ValueError(f"Unsupported language (no module): {language}")
    norm_text = language_module.text_normalize(text)
    phones, tones = language_module.g2p(norm_text)
    return norm_text, phones, tones, 


def text_to_sequence(text, language):
    _, phones, tones = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    input_text = "Yoga Được viết vào video" 
    input_text = "请提供完整地址（包括门牌号/公寓号）。"
    language= "ZH"

    #input_text = "水をマレーシアから買わなくてはならないのです。"
    #language= "JP"
    #phones, tones, lang_ids= text_to_sequence(input_text, language)
    norm_text, phones, tones = clean_text(input_text, language)
    print(f"norm_text   :{norm_text}\n")
    print(f"phones      :{phones}\n")
    print(f"tones       :{tones}\n")

    phone, tone, language = cleaned_text_to_sequence(phones, tones,language, lang_list=["ZH"])
    #print(f"lang_ids:{lang_ids}")\
    """
    symbols = ['_', 'ɯə', 'k', 'iə', 't', 'a', 'w', '.']
    new_symbols = []
    for ph in phones:
        #print(f"ph   :{ph}")
        if ph not in symbols and ph not in new_symbols:
            new_symbols.append(ph)
            print(f"new_symbols :{new_symbols}")
    """