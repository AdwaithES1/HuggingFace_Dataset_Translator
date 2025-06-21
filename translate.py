import json
import torch
import os
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# Supported languages for IndicTrans2 model (English to Indic)
SUPPORTED_LANGUAGES = {
    1: ("Assamese", "asm_Beng"),
    2: ("Bengali", "ben_Beng"),
    3: ("Bodo", "brx_Deva"),
    4: ("Dogri", "dgo_Deva"),
    5: ("Gujarati", "guj_Gujr"),
    6: ("Hindi", "hin_Deva"),
    7: ("Kannada", "kan_Knda"),
    8: ("Kashmiri (Arabic)", "kas_Arab"),
    9: ("Konkani", "kok_Deva"),
    10: ("Maithili", "mai_Deva"),
    11: ("Malayalam", "mal_Mlym"),
    12: ("Manipuri", "mni_Beng"),
    13: ("Marathi", "mar_Deva"),
    14: ("Nepali", "npi_Deva"),
    15: ("Oriya", "ory_Orya"),
    16: ("Punjabi", "pan_Guru"),
    17: ("Sanskrit", "san_Deva"),
    18: ("Santali", "sat_Olck"),
    19: ("Sindhi", "snd_Arab"),
    20: ("Tamil", "tam_Taml"),
    21: ("Telugu", "tel_Telu"),
    22: ("Urdu", "urd_Arab")
}

def prompt_for_dataset():
    dataset = input("Enter the Hugging Face dataset name (e.g., fever, scifact): ").strip()
    return dataset

def prompt_for_config(dataset_name):
    try:
        configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
    except Exception as e:
        print(f"Could not fetch configs for {dataset_name}: {e}")
        return None
    if not configs:
        print(f"No configs found for dataset {dataset_name}. Using default config None.")
        return None
    if len(configs) == 1:
        print(f"Only one config found for dataset {dataset_name}: {configs[0]}. Using it.")
        return configs[0]
    print(f"Available configs for dataset '{dataset_name}':")
    for i, config in enumerate(configs):
        print(f"  {i+1}. {config}")
    while True:
        choice = input(f"Enter the number of the config to use (1-{len(configs)}): ")
        if choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= len(configs):
                return configs[choice_num - 1]
        print("Invalid choice. Please try again.")

def prompt_for_split(dataset_name, config_name):
    try:
        splits = get_dataset_split_names(dataset_name, config_name, trust_remote_code=True)
    except Exception as e:
        print(f"Could not fetch splits: {e}. Using default 'train'")
        return "train"
    
    if not splits:
        print("No splits found. Using default 'train'")
        return "train"
    
    print("Available splits:")
    for i, split in enumerate(splits):
        print(f"  {i+1}. {split}")
    
    while True:
        choice = input(f"Enter the number of the split to use (1-{len(splits)}): ")
        if choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= len(splits):
                return splits[choice_num - 1]
        print("Invalid choice. Please try again.")

def prompt_for_column(columns):
    print("Available columns in the dataset:")
    for i, col in enumerate(columns):
        print(f"  {i+1}. {col}")
    print(f"  {len(columns)+1}. ALL (translate all string/list columns)")
    while True:
        choice = input(f"Enter the number of the column to translate (or {len(columns)+1} for ALL): ")
        if choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= len(columns):
                return columns[choice_num - 1]
            elif choice_num == len(columns)+1:
                return "ALL"
        print("Invalid choice. Please try again.")

def prompt_for_target_language():
    print("\nSupported target languages:")
    for num, (lang_name, lang_code) in SUPPORTED_LANGUAGES.items():
        print(f"  {num}. {lang_name} ({lang_code})")
    
    while True:
        choice = input(f"Enter language number (1-{len(SUPPORTED_LANGUAGES)}): ")
        if choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= len(SUPPORTED_LANGUAGES):
                return SUPPORTED_LANGUAGES[choice_num][1]
        print("Invalid choice. Please try again.")

def prompt_for_batch_size():
    batch_size = input("Enter batch size (default: 32): ").strip()
    return int(batch_size) if batch_size.isdigit() else 32

def prompt_for_output_file():
    filename = input("Enter output file name (e.g., translated.json): ").strip()
    # Ensure the directory exists
    out_dir = os.path.join(os.path.dirname(__file__), "translated_datasets")
    os.makedirs(out_dir, exist_ok=True)
    # Join the directory and filename
    output_file = os.path.join(out_dir, filename)
    print(f"Output will be saved to: {output_file}")
    return output_file

class IndicTranslator:
    def __init__(self, src_lang="eng_Latn", tgt_lang="mal_Mlym", batch_size=32):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        self.model_name = "ai4bharat/indictrans2-en-indic-1B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            attn_implementation="flash_attention_2" if self.device == "cuda" else None
        ).to(self.device)
        self.processor = IndicProcessor(inference=True)

    def translate_texts(self, texts):
        batch = self.processor.preprocess_batch(texts, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        inputs = self.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=512,
                num_beams=5,
                num_return_sequences=1
            )
        decoded = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return self.processor.postprocess_batch(decoded, lang=self.tgt_lang)

def main():
    dataset_name = prompt_for_dataset()
    config = prompt_for_config(dataset_name)
    split = prompt_for_split(dataset_name, config)
    print(f"Loading dataset '{dataset_name}' (config: {config}, split: {split}) ...")
    try:
        dataset = load_dataset(
            dataset_name,
            config,
            split=split,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    columns = list(dataset.features.keys())
    col_choice = prompt_for_column(columns)
    if col_choice == "ALL":
        text_columns = [col for col in columns if dataset.features[col].dtype in ["string", "list"]]
    else:
        text_columns = [col_choice]
    
    tgt_lang = prompt_for_target_language()
    batch_size = prompt_for_batch_size()
    output_file = prompt_for_output_file()

    translator = IndicTranslator(
        src_lang="eng_Latn",
        tgt_lang=tgt_lang,
        batch_size=batch_size
    )

    with open(output_file, "w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(dataset), batch_size), desc="Translating"):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            batch_dict = batch.to_dict()
            batch_size_actual = len(batch)
            
            texts_dict = {}
            for col in text_columns:
                texts = []
                for idx in range(batch_size_actual):
                    text_val = batch_dict[col][idx]
                    if isinstance(text_val, list):
                        texts.append(" ".join(text_val))
                    else:
                        texts.append(str(text_val))
                texts_dict[col] = texts
            
            translations_dict = {}
            for col in text_columns:
                try:
                    translations = translator.translate_texts(texts_dict[col])
                except Exception as e:
                    print(f"Error translating column '{col}' in batch {i//batch_size}: {str(e)[:200]}")
                    translations = []
                    for txt in texts_dict[col]:
                        try:
                            translations.append(translator.translate_texts([txt])[0])
                        except Exception as ex:
                            print(f"Failed to translate: {txt[:30]}... Error: {str(ex)[:100]}")
                            translations.append(txt)
                translations_dict[col] = translations
            
            for idx in range(batch_size_actual):
                new_item = {col: batch_dict[col][idx] for col in batch_dict}
                for col in text_columns:
                    new_item[f"translated_{col}"] = translations_dict[col][idx]
                f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                f.flush()

    print(f"Translation complete. Output saved to {output_file}")

if __name__ == "__main__":
    main()
