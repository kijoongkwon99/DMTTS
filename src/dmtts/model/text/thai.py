# https://github.com/myshell-ai/MeloTTS/pull/117

import re
import unicodedata
from num2words import num2words
from pythainlp.tokenize import word_tokenize
from pythainlp.transliterate import romanize
from pythainlp.util import normalize as thai_normalize
from pythainlp.util import thai_to_eng, eng_to_thai
from collections import defaultdict
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

etc_dictionary = {
    "1+1": "วันพลัสวัน",
    "2+1": "ทูพลัสวัน",
    # TODO: Add more Thai-specific abbreviations or special cases
}

english_dictionary = {
    "IT": "ไอที",
    "IQ": "ไอคิว",
    "PC": "พีซี",
    "CCTV": "ซีซีทีวี",
    "SNS": "เอสเอ็นเอส",
    "AI": "เอไอ",
    "CEO": "ซีอีโอ",
    "A": "เอ",
    "B": "บี",
    "C": "ซี",
    "D": "ดี",
    "E": "อี",
    "F": "เอฟ",
    "G": "จี",
    "H": "เอช",
    "I": "ไอ",
    "J": "เจ",
    "K": "เค",
    "L": "แอล",
    "M": "เอ็ม",
    "N": "เอ็น",
    "O": "โอ",
    "P": "พี",
    "Q": "คิว",
    "R": "อาร์",
    "S": "เอส",
    "T": "ที",
    "U": "ยู",
    "V": "วี",
    "W": "ดับเบิลยู",
    "X": "เอ็กซ์",
    "Y": "วาย",
    "Z": "แซด",
}

def normalize_with_dictionary(text, dic):
    if any(key in text for key in dic.keys()):
        pattern = re.compile("|".join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    return text


def normalize(text):
    text = text.strip()
    text = thai_normalize(text)
    text = normalize_with_dictionary(text, etc_dictionary)
    text = re.sub(r"\d+", lambda x: num2words(int(x.group()), lang="th"), text)
    text = normalize_english(text)
    text = text.lower()
    return text


def normalize_english(text):

    def fn(m):
        word = m.group()
        if word.upper() in english_dictionary:
            return english_dictionary[word.upper()]
        return "".join(english_dictionary.get(char.upper(), char) for char in word)

    text = re.sub(r"([A-Za-z]+)", fn, text)
    return text


# Load the Thai G2P dictionary
thai_g2p_dict = defaultdict(list)

dict_path = os.path.join(os.path.dirname(__file__), "wiktionary-23-7-2022-clean.tsv")
with open(dict_path, encoding="utf-8") as f:
    for line in f:
        word, phonemes = line.strip().split("\t")
        thai_g2p_dict[word].append(phonemes.split())
        thai_g2p_dict["ะ"] = ["a"]


def map_word_to_phonemes(word):
    logger.debug(f"Looking up word: {word}")

    # First, try to find the whole word in the dictionary
    phonemes_list = thai_g2p_dict.get(word)
    if phonemes_list:
        logger.debug(f"Found whole word {word} in dictionary")
        return " ".join(phonemes_list[0])

    # If not found, try to split the word
    subwords = word_tokenize(word, engine="newmm")

    if len(subwords) > 1:
        logger.debug(f"Word {word} split into subwords: {subwords}")
        # If the word can be split, recursively process each subword
        return " . ".join(map_word_to_phonemes(subword) for subword in subwords)
    else:
        logger.debug(f"Word {word} cannot be split, processing character by character")
        return map_partial_word(word)


def map_partial_word(word):
    if not word:
        return ""

    logger.debug(f"Mapping partial word: {word}")

    # Handle Thanthakhat (์) character
    if len(word) > 1 and word[1] == '์':
        logger.debug(f"Found Thanthakhat, skipping {word[:2]}")
        return map_partial_word(word[2:])

    # Handle vowels and tone marks
    if word[0] in thai_vowels or word[0] in thai_tone_marks:
        phoneme = thai_g2p_dict.get(word[0], [word[0]])[0]
        return phoneme + " " + map_partial_word(word[1:])

    # Try to find the longest matching prefix
    for i in range(len(word), 0, -1):
        prefix = word[:i]
        phonemes_list = thai_g2p_dict.get(prefix)
        if phonemes_list:
            logger.debug(f"Found matching prefix: {prefix}")
            return " ".join(phonemes_list[0]) + " " + map_partial_word(word[i:])

    # If no match found, return the first character and continue with the rest
    logger.debug(f"No match found for {word[0]}, continuing with rest")
    return word[0] + " " + map_partial_word(word[1:])


# Comprehensive mapping of Thai characters to their phonetic representations
thai_char_to_phoneme = {
    # Consonants
    'ก': 'k',
    'ข': 'kʰ',
    'ฃ': 'kʰ',
    'ค': 'kʰ',
    'ฅ': 'kʰ',
    'ฆ': 'kʰ',
    'ง': 'ŋ',
    'จ': 't͡ɕ',
    'ฉ': 't͡ɕʰ',
    'ช': 't͡ɕʰ',
    'ซ': 's',
    'ฌ': 't͡ɕʰ',
    'ญ': 'j',
    'ฎ': 'd',
    'ฏ': 't',
    'ฐ': 'tʰ',
    'ฑ': 'tʰ',
    'ฒ': 'tʰ',
    'ณ': 'n',
    'ด': 'd',
    'ต': 't',
    'ถ': 'tʰ',
    'ท': 'tʰ',
    'ธ': 'tʰ',
    'น': 'n',
    'บ': 'b',
    'ป': 'p',
    'ผ': 'pʰ',
    'ฝ': 'f',
    'พ': 'pʰ',
    'ฟ': 'f',
    'ภ': 'pʰ',
    'ม': 'm',
    'ย': 'j',
    'ร': 'r',
    'ล': 'l',
    'ว': 'w',
    'ศ': 's',
    'ษ': 's',
    'ส': 's',
    'ห': 'h',
    'ฬ': 'l',
    'อ': 'ʔ',
    'ฮ': 'h',

    # Vowels
    'ะ': 'a',
    'ั': 'a',
    'า': 'aː',
    'ำ': 'am',
    'ิ': 'i',
    'ี': 'iː',
    'ึ': 'ɯ',
    'ื': 'ɯː',
    'ุ': 'u',
    'ู': 'uː',
    'เ': 'eː',
    'แ': 'ɛː',
    'โ': 'oː',
    'ใ': 'aj',
    'ไ': 'aj',
    '็': '',  # Short vowel marker
    'ๆ': '',  # Repetition marker

    # Tone marks
    '่': '˨˩',  # Low tone
    '้': '˦˥',  # Rising tone
    '๊': '˥˩',  # Falling tone
    '๋': '˧',  # High tone

    # Special characters
    '์': '',  # Thanthakhat (cancels sound of preceding consonant)
}


def map_remaining_thai_chars(phones):
    mapped_phones = []
    for phone in phones:
        if phone in thai_char_to_phoneme:
            mapped_phones.append(thai_char_to_phoneme[phone])
        else:
            mapped_phones.append(phone)
    return mapped_phones


def thai_text_to_phonemes(text):
    #print("THAI_TEXT_TO_PHONEMES")
    #print(f"input_text      :{text}")
    #text = normalize(text)
    #print(f"normalized_text :{text}")
    words = word_tokenize(text, engine="newmm")
    #print(f"words: {words}")
    logger.debug(f"word_tokenize output: {words}")
    phonemes = []
    for word in words:
        word_phonemes = map_word_to_phonemes(word)
        phonemes.extend(word_phonemes.split())

    # Map any remaining Thai characters
    mapped_phonemes = map_remaining_thai_chars(phonemes)

    return " ".join(mapped_phonemes)


# Define Thai vowels, tone marks, and special characters
thai_vowels = set("ะัาำิีึืุูเแโใไฤฦ็")
thai_tone_marks = set("่้๊๋")
thai_special_chars = set("์ๆฯ๎๏")  # Thanthakhat, Maiyamok, Paitai, and Phinthu

# Update the thai_g2p_dict with proper mappings for vowels and tone marks
thai_g2p_dict.update({
    'โ': ['o'],
    'ใ': ['aj'],
    'ไ': ['aj'],
    'แ': ['ɛː'],
    'เ': ['eː'],
    'ฤ': ['rɯ'],
    'ฦ': ['lɯ'],
    '็': ['ː'],  # Mai Taikhu (used to shorten a vowel)
    '่': ['˨˩'],  # Low tone
    '้': ['˦˥'],  # Rising tone
    '๊': ['˥˩'],  # Falling tone
    '๋': ['˧'],  # High tone
    '์': [''],  # Thanthakhat (cancels the sound of the syllable)
    'ๆ': [''],  # Maiyamok (repetition mark)
    'ฯ': [''],  # Paitai (abbreviation mark)
    '๎': [''],  # Phinthu (used to indicate a silent consonant)
    '๏': [''],  # Angkhankhu (used to mark the end of a paragraph or section)
})


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


tone_map = {
    "˧": 2,  # Mid tone
    "˨˩": 1,  # Low tone
    "˦˥": 3,  # Rising tone
    "˩˩˦": 4,  # Falling tone
    "˥˩": 5,  # High tone
}


def extract_tones_orig(phs):
    phonemes = []
    tones = []
    current_tone = 2  # Default mid tone
    for ph in phs.split():
        if ph in tone_map:
            current_tone = tone_map[ph]
        elif ph == '_':
            phonemes.append(ph)
            tones.append(0)  # Zero tone for underscore
        else:
            phonemes.append(ph)
            tones.append(current_tone)
    return phonemes, tones


def extract_tones(phs):
    """
    Thai tone extraction (apply tone markers to the PREVIOUS phonemes).
    """
    ph_list = phs.split()
    phonemes = []
    tones = []
    default_tone = 0  # mid tone

    for i, ph in enumerate(ph_list):
        # tone symbol → retroactively update previous phonemes' tone
        if ph in tone_map:
            tone_value = tone_map[ph]
            # update backwards until we hit a phoneme that already has a tone from another marker
            for j in range(len(tones) - 1, -1, -1):
                if tones[j] != default_tone:
                    break
                tones[j] = tone_value
        elif ph == "_":
            phonemes.append(ph)
            tones.append(0)
        else:
            phonemes.append(ph)
            tones.append(default_tone)

    return phonemes, tones


def g2p(norm_text, pad_start_end=True):
    """
    Thai G2P simplified version.
    Fixes tone propagation (apply tone mark to preceding phonemes).
    """
    #print(f"norm_text: {norm_text}")

    # Step 1: Phoneme conversion for grouped text
    phonemes = thai_text_to_phonemes(norm_text)
    phoneme_groups = phonemes.split(".")
    phoneme_groups = list(filter(str.strip, phoneme_groups))

    #print(f"phoneme: {phonemes}")
    #print("--------------------------------------------------------")
    #print(f"phoneme_groups: {phoneme_groups}")
    #print("--------------------------------------------------------")

    # Step 2: Extract tones per group (tone marks apply backward)
    all_phonemes, all_tones = [], []
    for p_group in phoneme_groups:
        group_phonemes, group_tones = extract_tones(p_group)
        #print(f"p_group         : {p_group}")
        #print(f"group_phonemes  : {group_phonemes}")
        #print(f"group_tones     : {group_tones}")
        #print("----------------------------------------------------")

        all_phonemes.extend(group_phonemes)
        all_tones.extend(group_tones)

    # Step 3: Optional padding
    if pad_start_end:
        all_phonemes = ["_"] + all_phonemes + ["_"]
        all_tones = [0] + all_tones + [0]

    assert len(all_phonemes) == len(all_tones), "Phoneme-tone length mismatch!"

    #print("\n✅ FINAL RESULT")
    #print(f"Phonemes ({len(all_phonemes)}): {all_phonemes[:50]}{'...' if len(all_phonemes)>50 else ''}")
    #print(f"Tones    ({len(all_tones)}): {all_tones[:50]}{'...' if len(all_tones)>50 else ''}")
    #print("----------------------------------------------------")

    return all_phonemes, all_tones

def g2p_orig(norm_text, pad_start_end=True):

    print(f"norm_text: {norm_text}")

    # Phoneme conversion for grouped text
    phonemes = thai_text_to_phonemes(norm_text)
    phoneme_groups = phonemes.split(".")
    phoneme_groups = list(filter(str.strip, phoneme_groups))
    """
    print(f"phoneme: {phonemes}")
    print("--------------------------------------------------------")

    print(f"phoneme_groups: {phoneme_groups}")
    print("--------------------------------------------------------")
    """
    for p_group in phoneme_groups:
        group_phonemes, group_tones = extract_tones(p_group)
        """
        print(f"p_group         :{p_group}")
        print(f"group_phonemes  :{group_phonemes}")
        print(f"group_tones     :{group_tones}")
        print("----------------------------------------------------")
        """



    """
    if pad_start_end:
        phs = ["_"] + phs + ["_"]
        tones = [0] + tones + [0]  # Zero tone for start/end padding


        assert len(phs) == len(tones)
    """

    #return phs, tones




if __name__ == "__main__":
    try:
        #from text.symbols import symbols
        #text = "ฉันเข้าใจคุณค่าของงานของฉันและความหมายของสิ่งที่ฟอนเทนทำเพื่อคนทั่วไปเป็นอย่างดี ฉันจะใช้ชีวิตอย่างภาคภูมิใจในงานของฉันต่อไป"
        text= "Chen Yongquan เจ้าของร้านถุงเท้าเล็กๆ ระดมทุนได้ทั้งหมด 1.129155 พันล้านหยวน"
        text= "คุณรู้หมายเลขโทรศัพท์ของเธอไหม?"
        print(f"text            :{text}")
        text = text_normalize(text)
        print(f"normalized_text :{text}")
        phones, tones = g2p(text)

        """
        new_symbols = []
        for ph in phones:
            if ph not in symbols and ph not in new_symbols:
                new_symbols.append(ph)
                print('update!, now symbols:')
                print(new_symbols)
                with open('thai_symbol.txt', 'w') as f:
                    f.write(f'{new_symbols}')
        """
    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"phones{len(phones)} : {phones}")
    print(f"tones{len(tones)}   : {tones}")