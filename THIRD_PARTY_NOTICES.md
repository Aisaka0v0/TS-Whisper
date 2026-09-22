TS-Whisper — Third-Party Notices
Copyright (c) 2024 Hao Ma

This product is licensed under the MIT License (see LICENSE).

This product includes software developed by third parties. The notices below
are provided in accordance with the MIT License requirement that the copyright
notice and permission notice of the upstream works be included in all copies or
substantial portions of the Software.

================================================================================
1. OpenAI Whisper  (https://github.com/openai/whisper)
================================================================================
Copyright (c) 2022 OpenAI
License: MIT License

Used in two distinct ways:

(a) As an external PyPI dependency (`pip install -U openai-whisper`, see
    README). The following modules import and call the upstream package:
      * training_pt.py, training_lora.py, evaluation_pt.py,
        evaluation_lora.py   -> `import whisper`, `whisper.load_model(..)`,
                                `whisper.DecodingOptions(..)`,
                                `whisper.available_models()`
      * data_utils/dataloader.py  -> `whisper.audio` (CHUNK_LENGTH, N_FRAMES,
                                log_mel_spectrogram, pad_or_trim, load_audio),
                                `whisper.tokenizer.Tokenizer`
      * data_utils/create_data.py -> `whisper.tokenizer` (LANGUAGES,
                                TO_LANGUAGE_CODE, get_tokenizer)
      * evaluation_pt.py         -> `whisper.model.ResidualAttentionBlock`,
                                `whisper.decoding.DecodingTask`,
                                `whisper.normalizers.EnglishTextNormalizer`

(b) As adapted source. `data_utils/create_data.py`, `data_utils/dataloader.py`
    and the training / evaluation entry points are adapted from OpenAI's
    Whisper fine-tuning recipe and from `whisper/audio.py`,
    `whisper/decoding.py`, `whisper/tokenizer.py` and `whisper/normalizers`.
    Several routines are carried over essentially unchanged, for example:
      * `AudioDataset._get_special_tokens`, `_encode_text_with_timestamps`,
        `_get_partial_segment_start`, `_get_text_tokens`, `_construct_decoder_output`
      * the `Record` / `DataProcessor` structures in `create_data.py`
      * `_get_prompt_tokens` and the timestamps prompt handling
      * the mel-spectrogram window and `pad_or_trim(.., N_FRAMES)` handling

The MIT license notice for the OpenAI Whisper work is reproduced below:

    MIT License

    Copyright (c) 2022 OpenAI

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

================================================================================
2. Microsoft LoRA (loralib)  (https://github.com/microsoft/LoRA)
================================================================================
Copyright (c) Microsoft Corporation
Used by: training_lora.py, evaluation_lora.py, and the LoRA path in
         CLAPSep-style model surgery (`import loralib as lora`).
Install: `pip install loralib` (see README).
License: MIT License

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

================================================================================
3. Other dependencies
================================================================================
Other runtime dependencies (PyTorch, numpy, pandas, tqdm) are used as
unmodified external libraries and are NOT vendored into this repository.
Each remains under its own upstream license; those licenses are not
reproduced here. Consult the individual packages for their terms (for
example, PyTorch is BSD-3-Clause).

The LibriMix recipe and the SSL_for_multitalker speaker-embedding extractor
referenced in the README are used as external data-preparation tools and are
not vendored into this repository. Consult those projects for their licenses.
