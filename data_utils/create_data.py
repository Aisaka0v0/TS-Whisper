#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：TS_Whisper 
@File    ：create_data.py
@IDE     ：PyCharm 
@Author  ：Aisaka/Hao Ma @SDU
@Date    ：2023/12/18 下午3:24 
'''
import argparse
import json
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Union
import pandas as pd
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer



def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a jsonl file to be used for fine-tuning a Whisper model"
    )


    parser.add_argument(
        "--audio-dir",
        type=str,
        help=(
            "Path to directory containing audio files. This option is used only when "
            "`--with-timestamps` is set. Audio formats that can be read by ffmpeg are supported."
        ),
    )
    parser.add_argument(
        "--transcript-file",
        type=str,
        default='./audio_text.json'
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default='/home/user/202212661/libriMix/Libri2Mix/wav16k/max/metadata/mixture_test_mix_clean.csv',
        help=(
            "Path to a text file containing audio filenames and transcriptions. This option is "
            "used only when `--without-timestamps` is set. Each line must be in the format of "
            "`<audio_path>\t<transcription>`."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help="Language of the data",
    )
    parser.add_argument("--output", type=str, default="./data/test.json", help="Path to output json file")
    parser.add_argument(
        "--dump-dir", type=str, default="dump", help="Directory to dump audio files"
    )

    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=223,
        help=(
            "Maximum length of prompt in Whisper tokens. Defaults to 223, which equals to "
            "`model.dims.n_text_ctx (=448) // 2 - 1` (-1 is for the special token `sot_prev` and "
            "the other half is for the transcribed tokens)."
        ),
    )
    parser.add_argument(
        "--max-tokens-length",
        type=int,
        default=219,
        help=(
            "Maximum length of text and timestamps tokens. Utterances longer than this will be "
            "skipped. Defaults to 219, which equals to `model.dims.n_text_ctx (=448) // 2 - 5` "
            "(5 is the maximum number of special tokens used other than the `sot_prev`."
        ),
    )

    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="multilingual",
        choices=["multilingual", "english"],
        help=(
            "Type of Whisper tokenizer to use. Tokenizer is used to count the number of tokens "
            "in the transcriptions."
        ),
    )
    parser.add_argument("--normalize-unicode", action="store_true", help="Normalize unicode")
    return parser


DURATION = 30000  # 30 seconds in milliseconds
SAMPLE_RATE = 16000
DURATION_IN_SAMPLES = int(DURATION * SAMPLE_RATE / 1000)


@dataclass
class Utterance:
    """
    Representing a single segment of audio with a transcription. Corresponds to a single chunk in a
    .srt (or .vtt) file.
    """

    text: str
    start: Optional[int] = None  # in milliseconds
    end: Optional[int] = None  # in milliseconds


@dataclass
class Record:
    """
    A single training instance for Whisper.
    `text` can include timestamps in the format of <|0.00|>.
    """

    audio_path: str
    text: str  # text including timestamps
    language: str = "en"
    prompt: str = ""  # previous text including timestamps


@dataclass
class PromptNode:
    text: str  # text including timestamps
    num_tokens: int


class DataProcessor:
    def __init__(
        self,
        audio_dir: str = None,
        transcript_file: str = None,
        data_file: str = None,
        language: str = "en",
        output: str = "data.json",
        dump_dir: str = "dump",
        max_prompt_length: int = 223,
        max_tokens_length: int = 219,
        tokenizer_type: str = "multilingual",
        normalize_unicode: bool = False,
    ) -> None:
        self.audio_dir = audio_dir
        self.transcript_file = transcript_file
        self.data_file = data_file
        self.language = language
        self.output = output
        self.dump_dir = dump_dir
        self.max_prompt_length = max_prompt_length
        self.max_tokens_length = max_tokens_length
        self.tokenizer_type = tokenizer_type
        self.normalize_unicode = normalize_unicode
        self.speech_df = pd.read_csv(self.data_file)
        tf = open(self.transcript_file, "r")
        self.text_dict = json.load(tf)

        self._verify_args()

        self.tokenizer = get_tokenizer(multilingual=(self.tokenizer_type == "multilingual"))
        Path(self.dump_dir).mkdir(parents=True, exist_ok=True)

    def _verify_args(self) -> None:

        if not self.data_file:
            raise ValueError("`data_file` must be set when `with_timestamps` is False")

        if self.language not in LANGUAGES:
            if self.language in TO_LANGUAGE_CODE:
                self.language = TO_LANGUAGE_CODE[self.language]
            else:
                raise ValueError(f"Unsupported language: {self.language}")

        if self.tokenizer_type not in ["multilingual", "english"]:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

        if Path(self.output).exists():
            raise ValueError(f"Output file {self.output} already exists")

    def run(self) -> None:
        self._process_without_timestamps()

    def _process_without_timestamps(self) -> None:
        records = []
        # with open(self.data_file, encoding="utf-8") as f:
        for _, line in self.speech_df.iterrows():
            utts = line['mixture_ID'].split('_')
            for utt in utts:
                audio_path = line['mixture_path']
                text = self.text_dict[utt]

                # audio_path, text = line.strip().split("\t")
                if self.normalize_unicode:
                    text = unicodedata.normalize("NFKC", text)

                tokens = self.tokenizer.encode(text)
                if len(tokens) > self.max_tokens_length:
                    print(
                        f"Skipping {audio_path} ({text}) because it is too long "
                        f"({len(tokens)} tokens)"
                    )
                    continue

                record = Record(audio_path=audio_path, text=text, language=self.language, prompt=utt.split('-')[0])
                records.append(record)

        self.write_records(records, self.output)

    @staticmethod
    def read_records(path: Union[str, Path]) -> List[Record]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                record = Record(
                    audio_path=data["audio_path"],
                    text=data["text"],
                    language=data["language"],
                    prompt=data["prompt"],
                )
                records.append(record)
        return records

    @staticmethod
    def write_records(records: List[Record], path: Union[str, Path]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                data = {
                    "audio_path": record.audio_path,
                    "text": record.text,
                    "language": record.language,
                    "prompt": record.prompt,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")



def main():
    args = get_parser().parse_args()
    processor = DataProcessor(
        audio_dir=args.audio_dir,
        transcript_file=args.transcript_file,
        data_file=args.data_file,
        language=args.language,
        output=args.output,
        dump_dir=args.dump_dir,
        max_prompt_length=args.max_prompt_length,
        max_tokens_length=args.max_tokens_length,
        tokenizer_type=args.tokenizer_type,
        normalize_unicode=args.normalize_unicode,
    )
    processor.run()


if __name__ == "__main__":
    main()
