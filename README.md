# Automated Detection of Personally Identifiable Information on Speech Data
## Repository of Master's research paper

This repository contains the code for a research project on detecting personally identifiable information (PII) directly from speech audio, without relying on text transcripts. The project studies both binary (PII vs. non-PII) and multilabel setups, exploring how far one can go using only acoustic features for privacy-related classification on speech data.

​
##Overview

**Goal:** Automatically classify whether segments of speech contain PII categories (names, locations, religions, date_time and nrp) using only audio input.

**Key idea: **
- Replace the typical “ASR → text PII detection” pipeline with a model that operates directly on spectrograms or other audio features.

**Motivation: Audio-first detection can:**
- Reveal how much privacy-sensitive information is encoded in speech
- Reduce dependency on full ASR systems
- Enable earlier, on-device or streaming privacy filtering.
        ​

##Methodology

**Input representation: **audio tesnsors of different sizes(Phrase, word, 0.5 slice and frame)

**Models used for Classification:** WavLM, Hubert, Whisper
    
**Task setups:**

- Binary: “Contains any PII” vs. “No PII”.
- Multilabel: Predict multiple PII categories per segment.


