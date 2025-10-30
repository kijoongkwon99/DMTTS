import copy

punctuation = [" ", "!", "?", "…", ",", ".", "'", "-"]
#punctuation = ["!", "?", "…", ",", ".", "'", "-"] 
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"


#=======Symbols========#
LANG = {
    "KR": {
        "symbols": [
            'ᄀ', 'ᅳ', 'ᄂ', 'ᆫ', 'ᅫ', 'ᄎ', 'ᅡ', 'ᅥ', 'ᄏ', 
            'ᄅ', 'ᅧ', 'ᅩ', 'ᄋ', 'ᅢ', 'ᄊ', 'ᆮ', 'ᄁ', 'ᄐ', 
            'ᄄ', 'ᅴ', 'ᄉ', 'ᆼ', 'ᅵ', 'ᅱ', 'ᄒ', 'ᄍ', 'ᄆ', 
            'ᅮ', 'ᅭ', 'ᄃ', 'ᆯ', 'ᄌ', 'ᅪ', 'ᄇ', 'ᅦ', 'ᅯ', 
            'ᆷ', 'ᆸ', 'ᆨ', 'ᅬ', 'ᅲ', 'ᄑ', 'ᅣ', 'ᅨ', 'ᄈ', 
            'ᅤ', 'ᅰ',

        ],
        "num_tones": 1,
        "module": "dmtts.model.text.korean"
    },
    "VI": {
        "symbols": [
            'e', 'ɪ', 'ʧ', 'iə', 'n', 'ă', 'j', 'v', 'i', 'ʈ', 
            'k', 'uə', 'b', 'a', 'tʰ', 'ɤ', 'ɯə', 'ŋ', 'd', 't', 
            'ɔ', 'ŋ͡m', 'f', 'ʐ', 'ɤ̆', 'u', 'ɲ', 'w', 'h', 'l', 's', 
            'x', 'm', 'kw', 'o', 'p', 'ɛ', 'iɛ', 'ɯ', 'c', 'z', 'tʰw', 
            'ʷɤ̆', 'ʂ', 'tw', 'ʷiə', 'tʃ', 'lw', 'ɛu', 'ʂw', 'ʷi', 'xw', 
            'ʷa', 'ŋw', 'eo', 'k͡p', 'cw', 'sw', 'hw', 'ɣ', 'zw', 'ʷă', 
            'ʷɛ', 'ʤ', 'bw', 'dw', 'ʷe', 'r', 'ʈw', 'ə', 'ɲw', 'nw', 'ʊ', 
            'g', 'æ', 'ˌ', 'ô', '*', 'vw', 'ʃ', 'ɑ', 'ɣw', 'ʷiu', 'fw', 'jw'
            
        ],
        "num_tones": 6,
        "module": "dmtts.model.text.vietnamese"
    },
    "EN": {
        "symbols": [
            'b', 'k', 'p', 'r', 'ih', 'n', 't', 'ng', 'dh', 'ah', 'ow', 
            'l', 'iy', 's', 'eh', 'w', 'ch', 'aa', 'ae', 'z', 
            'k', 'er', 'd', 'f', 'm', 'ao', 'sh', 'V', 'ay', 
            'uh', 'g', 'ey', 'uw', 'th', 'jh', 'aw', 'hh', 
            'y', 'zh', 'oy', '..'],
        "num_tones": 4,
        "module": "dmtts.model.text.english"
    },
    "ZH": {
        "symbols": [

            'k', 'a', 'EE', 'er', 'p', 'u', 'ei', 'w', 'ai', 
            's', 'un', 'an', 'h', 'ua', 't', 'i', 'j', 'ia', 
            'y', 'v', 'c', 'En', 'b', 'ie', 'z', 'ong', 'ao', 
            'o', 'm', 'g', 'l', 'uo', 'AA', 'd', 'iao', 'ch', 
            'van', 'zh', 'en', 'eng', 'x', 'ing', 'q', 'ui', 
            'ou', 'sh', 'ang', 've', 'ir', 'f', 'vn', 'in', 
            'e', 'n', 'uan', 'E', 'r', 'ian', 'i0', 'uang', 
            'iang', 'iong', 'iu', 'OO', 'uai'

        ],
        "num_tones": 6,
        "module": "dmtts.model.text.chinese"
    },
    "JP": {
        "symbols": [
            'm', 'i', 'z', 'u', 'o', 'a', 'r', 'e', 'sh', 'k', 'w', 
            'n', 't', 'h', 'd', 's', 'y', 'b', 'N', 'ry', 'j', 'g', 
            'ts', 'q', 'ny', 'p', 'f', 'gy', 'ky', 'ch', 'hy', 'my', 
            'o:', 'by', 'py', 'a:', '禕', 'dy', '剝', 'i:', 'u:', 'zy', 
            'ty'
        ],
        "num_tones": 1,
        "module": "dmtts.model.text.japanese"
    },
    "TH": {
        "symbols" : [
            'k', 'r', 'a', 'd', 'oː', 't̚', 'j', 'ɤʔ', 'tʰ', 'm', 
            'h', 'aj', 't͡ɕ', 'e', 'p̚', 'kʰ', 'aw', 'p', 'ua̯', 'ɔː', 'ŋ', 'ɯa̯', 
            'n', 's', 'b', 'i', 'l', 'ɛʔ', 'ia̯', 'uː', 't͡ɕʰ', 'eː', 'iː', 'ʔ', 
            'aː', 'ɛː', 'ua̯j', 'aːw', 'o', 'ɯ', 'k̚', 'w', 'aʔ', 'ɯː', 'u', 'ew', 
            'pʰ', 'iw', 'aːj', 'f', 'c', 'ɔ', 'ː', 'uj', 'ɔːj', 't', 'ɤːj', 'ɛːw', 
            'ɔj', 'ɔʔ', 'lɯː', 'ɤː', 'rɯ', 'a̯', 'ɤ', 'ia̯w', 'lɯ', 'am', 'ɨ', 'rɯː', 
            'ɛ', 'oʔ', 'ua', 'oːj', 'cʰ', 'eːw', 'à', 'a̯j', 'ɛw', 'ì', 'eʔ',

            'ɗ', 'æ', ':', '%', '“', '”', '&', ';',
        ],
        "num_tones": 6,
        "module": "dmtts.model.text.thai"
    }
}

def prefix_language(lang_dict):
    """LANG 딕셔너리에 언어 prefix를 붙인 symbol로 새 dict 반환"""
    prefixed = copy.deepcopy(lang_dict)
    for lang, info in prefixed.items():
        info["symbols"] = [f"{lang}_{s}" for s in info["symbols"]]
    return prefixed



def _validate_langs(lang_list):
    unknown = [x for x in lang_list if x not in LANG]
    if unknown:
        raise ValueError(f"Unknown language codes: {unknown}")

def get_symbol(lang_list=None, sort_symbols=True, add_prefix_language=False):
    """
    return: (symbols_list, num_symbols)
            symbols_list = [pad] + merged_normal_symbols + pu_symbols
    """
    chosen = list(LANG.keys()) if not lang_list else lang_list
    _validate_langs(chosen)

    base_langs = prefix_language(LANG) if add_prefix_language else LANG


    # 병합
    merged = []
    for lg in chosen:
        #merged.extend(LANG[lg]["symbols"])
        merged.extend(base_langs[lg]["symbols"])

    # 중복 제거 + 정렬 옵션
    if sort_symbols:
        normal_symbols = sorted(set(merged))
    else:
        normal_symbols = list(dict.fromkeys(merged))  # 순서 보존하며 dedupe

    symbols = [pad] + normal_symbols + pu_symbols
    assert len(set(symbols)) == len(symbols), "Duplicate symbol detected."
    return symbols, len(symbols)

def symbol_to_id(sym=None):
    if sym is None:
        sym, _ = get_symbol()
    return {s: i for i, s in enumerate(sym)}

def get_language_id(lang_list=None):
    chosen = list(LANG.keys()) if not lang_list else lang_list
    _validate_langs(chosen)
    lang2id= {lg: idx for idx, lg in enumerate(chosen)}
    return lang2id, len(chosen)


def get_tone_id(lang_list=None):
    chosen = list(LANG.keys()) if not lang_list else lang_list
    _validate_langs(chosen)

    tone_start_map = {}
    cursor = 0
    for lg in chosen:
        tone_start_map[lg] = cursor
        cursor += int(LANG[lg]["num_tones"])
    return tone_start_map, cursor

def get_prefix_cleaned_text(cleaned_text, language):
    """
    cleaned_text 리스트에 대해 언어 prefix를 붙이되,
    pad, SP, UNK, punctuation 등은 제외한다.
    """
    # prefix 제외 대상
    exclude_tokens = set(punctuation + ["SP", "UNK", pad])

    prefixed = []
    for s in cleaned_text:
        if s in exclude_tokens:
            prefixed.append(s)
        else:
            prefixed.append(f"{language}_{s}")
    return prefixed



def cleaned_text_to_sequence(cleaned_text, tones, language, lang_list=None, add_prefix_language=False):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    if add_prefix_language:
        cleaned_text= get_prefix_cleaned_text(cleaned_text, language)
        #cleaned_text = [f"{language}_{s}" for s in cleaned_text]
        #print(f"cleaned text: {cleaned_text}")

    symbols, _ = get_symbol(lang_list=lang_list, add_prefix_language=add_prefix_language)
    #print(f"symbols:\n{symbols}")
    symbol_to_id_map = symbol_to_id(symbols)
    phones = [symbol_to_id_map[symbol] for symbol in cleaned_text]
    
    tone_map, total_tones = get_tone_id(lang_list)
    tone_start = tone_map[language]
    #print(f"tones : {tones}")
    tones = [i + tone_start for i in tones]

    lang_id_map, _ = get_language_id(lang_list)
    lang_id = lang_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids


if __name__ == "__main__":
    print("symbols.py")
    langs = ["KR", "EN", "JP","ZH", "VI", "TH"]
    lang = 'TH'
    
    symbols, _ = get_symbol(langs, sort_symbols=True)
    print(symbols)
    print(symbol_to_id(symbols))
    print("language_id:", get_language_id(langs))
    tone_map, total_tones = get_tone_id(langs)
    print("tone_map:", tone_map, "total_tones:", total_tones)


    phones = "_ r uː s ɯ k̚ p ua̯ t̚ kʰ aː ŋ n aj h uː _"
    tones = "0 3 3 1 1 1 1 1 1 5 5 5 2 2 4 4 0"
    phones = phones.split()         # ['_', 'ɯə', 'j', 'm', ...]
    tones  = [int(t) for t in tones.split()]  # [0, 1, 1, 5, 5, ...]
    language= "TH"

    print("#### None")
    a, b, c = cleaned_text_to_sequence(phones, tones, language, lang_list=None, add_prefix_language=False)
    print(a)
    print(b)
    print(c)
    
    print("#### [TH]")
    a, b, c = cleaned_text_to_sequence(phones, tones, language, lang_list=["TH"])
    print(a)
    print(b)
    print(c)
    