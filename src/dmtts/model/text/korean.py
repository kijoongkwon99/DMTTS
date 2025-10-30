import re
import unicodedata

from transformers import AutoTokenizer
from dmtts.model.text.kr_normalizer import N2gk, N2gkPlus



#from num2words import num2words
#from melo.text.ko_dictionary import english_dictionary, etc_dictionary
from anyascii import anyascii
from jamo import hangul_to_jamo

def normalize(text, use_n2gk_plus=True):
    text = text.strip()
    if use_n2gk_plus:
        n2gk_plus = N2gkPlus()
        return n2gk_plus(text)

    return text


g2p_kr = None
def korean_text_to_phonemes(text, character: str = "hangeul") -> str:
    """

    The input and output values look the same, but they are different in Unicode.

    example :

        input = '하늘' (Unicode : \ud558\ub298), (하 + 늘)
        output = '하늘' (Unicode :\u1112\u1161\u1102\u1173\u11af), (ᄒ + ᅡ + ᄂ + ᅳ + ᆯ)

    """
    global g2p_kr  # pylint: disable=global-statement
    if g2p_kr is None:
        from g2pkk import G2p

        g2p_kr = G2p()

    if character == "english":
        from anyascii import anyascii
        #print(f"raw_text        :{text}")
        text = normalize(text)
        #print(f"normalized_text :{text}")
        text = g2p_kr(text)
        #print(f"g2p_kr_text     :{text}")
        text = anyascii(text)
        #print(f"anyascii_text   :{text}")
        return text

    #print(f"text:   {text}")
    text = normalize(text)
    #print(f"norm:   {text}")
    text = g2p_kr(text)
    #print(f"g2p :   {text}")
    text = list(hangul_to_jamo(text))  # '하늘' --> ['ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆯ']
    return "".join(text)

def text_normalize(text):
    text = normalize(text)
    return text


def distribute_phone(n_phone, n_word):
    phones_per_word = [0] * n_word
    for task in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def g2p(norm_text, add_space=False, delimit_word=True):

    phs = []
    if delimit_word:
        words = norm_text.strip().split(" ")
        for idx, word in enumerate(words):
            if not word:
                continue
            phonemes = korean_text_to_phonemes(word)
            phs += list(phonemes)
            # 마지막 단어가 아닐 때만 공백 추가
            if add_space:
                if idx < len(words) - 1:
                    phs.append("SP")
    else:
        phonemes = korean_text_to_phonemes(norm_text)
        phonemes = re.sub(r'\s+', '', phonemes)
        phs += list(phonemes)
    print(f"phonemes:   {''.join(phs)}")  
    phones = ["_"] + phs + ["_"]
    tones = [0 for i in phones]
    return phones, tones 


if __name__ == "__main__":
    text = "그 책 다 읽은 후에 빌려 줄래?"
    text = "내 말이 맞다는 걸 증명해봐"
    text = "다음 주 목요일에 나하고 테니스 칠래?"
    text = "끊었다가 다시 걸게."
    text = "그 책 다 읽은 후에 빌려 줄래?"
    text = "용돈을 아껴 써라."
    text = "박지성은 오늘날 최고의 아시아 선수 중 한 명이다."


    phonemes = korean_text_to_phonemes(text)
    
    print(f"phonemes:   {phonemes}")
    #phones = ["_"] + list(phonemes) + ["_"]
    #tones = [0 for i in phones]

    #print(f"phones  :{phones}")
    #print(f"tones   :{tones}")

    g2p_phones, g2p_tones = g2p(text)

    print(f"g2p_phones({len(g2p_phones)})  :{g2p_phones}")
    print(f"g2p_tones({len(g2p_tones)})   :{g2p_tones}")
    print(f"len symmetry: {len(g2p_phones), len(g2p_tones)}")

        