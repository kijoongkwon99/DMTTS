import json
from pathlib import Path
from collections import defaultdict
from random import shuffle
from typing import Optional

from tqdm import tqdm
import click
#from dmtts.model.text.cleaner import clean_text_bert
import os
import torch
#from text.symbols import symbols, num_languages, num_tones
from dmtts.model.text.symbols import get_symbol, get_language_id, get_tone_id
from dmtts.model.text.cleaner import clean_text

### config 생성시에 lang_list 넣는거 필요함

@click.command()
@click.option(
    "--metadata",
    default="data/example/metadata.list",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--train_language", default="ZH")
@click.option("--version_of_model", default=1)
@click.option("--cleaned-path", default=None)
@click.option("--train-path", default=None)
@click.option("--val-path", default=None)
@click.option(
    "--config_path",
    default="../../../data/config.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-per-spk", default=4)
@click.option("--max-val-total", default=8)
@click.option("--clean/--no-clean", default=True)
def main(
    metadata: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
    train_language: str,
    version_of_model: int,
):
    
    HERE = Path(__file__).resolve()
    PROJECT_ROOT = HERE.parents[3]          # 상황에 맞게 조정
    DATA_DIR = PROJECT_ROOT / "data"
    rel_path = DATA_DIR / f"V{version_of_model}" / f"{train_language}"
    print(f"rel_path: {rel_path}")
    #exit()
    rel_path.mkdir(parents=True, exist_ok=True)
    if train_path is None:
        train_path = rel_path / "train.list"
    if val_path is None:
        val_path = rel_path / "val.list"
    out_config_path = rel_path / "config.json"

    if cleaned_path is None:
        cleaned_path = str(rel_path / "metadata.list")
        #cleaned_path = metadata + ".cleaned"        


    #exit()
    if clean:
        out_file = open(cleaned_path, "w", encoding="utf-8")
        new_symbols = []
        language_list = []
        language_seen = set()
        #exit()
        for line in tqdm(open(metadata, encoding="utf-8").readlines()):
            try:
                utt, spk, language, text = line.strip().split("|")
                


                if language not in language_seen:
                    language_seen.add(language)
                    language_list.append(language)
                #norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device='cuda:0')

                norm_text, phones, tones = clean_text(text, language)
                #exit()
                
                for ph in phones:
                    #exit()
                    symbols, _ = get_symbol([str(language)],add_prefix_language=False)
                    #print(f"symobls of {language}: {symbols}")
                    #print(f"ph: {ph}")
                    #print(f"new_symbols: {new_symbols}")
                    #exit()
                    if ph not in symbols and ph not in new_symbols:
                        new_symbols.append(ph)
                        print('update!, now symbols:')
                        print(new_symbols)
                        
                        with open(f'{language}_symbol.txt', 'w') as f:
                            f.write(f'{new_symbols}')

                assert len(phones) == len(tones)
                #assert len(phones) == sum(word2ph)
                out_file.write(
                    "{}|{}|{}|{}|{}|{}\n".format(
                        utt,
                        spk,
                        language,
                        norm_text,
                        " ".join(phones),
                        " ".join([str(i) for i in tones]),
                        #" ".join([str(i) for i in word2ph]),
                    )
                )
                #bert_path = utt.replace(".wav", ".bert.pt")
                #os.makedirs(os.path.dirname(bert_path), exist_ok=True)
                #torch.save(bert.cpu(), bert_path)
            except Exception as error:
                print("err!", line, error)

        out_file.close()

        metadata = cleaned_path

    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(metadata, encoding="utf-8") as f:
        for line in f.readlines():
            utt, spk, language, text, phones, tones = line.strip().split("|")
            spk_utt_map[spk].append(line)

            if spk not in spk_id_map.keys():
                spk_id_map[spk] = current_sid
                current_sid += 1

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)


    num_languages = len(language_list)
    _, num_tones = get_tone_id(language_list)
    symbols, _ = get_symbol(language_list)


    config = json.load(open(config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map
    config["data"]["training_files"] = str(train_path)
    config["data"]["validation_files"] = str(val_path)
    config["data"]["n_speakers"] = len(spk_id_map)
    config["data"]["lang_list"] = language_list
    config["num_languages"] = num_languages
    config["num_tones"] = num_tones
    config["symbols"] = symbols
    

    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print("You have to update each symbol at the /text/symbols.py for training")

if __name__ == "__main__":
    main()