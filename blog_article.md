# From Chinese to English — and a Detour into Moroccan Arabic: Building an AI Video Language Transformation Pipeline

**Authors: Adam Maytoussi, Soulaymane Temimi**

---

## The Starting Point

This semester, our AI course asked us to pick a project and push it further — past the tutorial, past the ready-made tool, into something that required us to understand the underlying machinery.

We picked **AI Video Language Transformation**: a pipeline that takes Chinese-language GitHub tutorial videos and converts them into English — same video, new audio, voice cloned from the original speaker. The idea was to make technical content accessible across language barriers without losing the personality of the original presenter.

That was the main project. But by the end of the project, we had also taken a detour into something much less explored: fine-tuning a voice cloning model for Darija — Moroccan Arabic. What happened during that detour is, honestly, the more interesting story.

---

## The Pipeline

The main pipeline chains together several components, each handling a distinct task:

1. **Transcription** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (large-v3) converts the Chinese audio into timed text segments
2. **Translation** — [DeepL](https://www.deepl.com/) translates Chinese segments to English, with a [Gemini](https://deepmind.google/technologies/gemini/) review pass for technical terminology (GitHub-specific vocabulary like "pull request", "commit", "branch")
3. **Speaker Diarization** — [pyannote.audio](https://github.com/pyannote/pyannote-audio) identifies who is speaking and when, enabling per-speaker voice cloning
4. **Speech Synthesis** — [F5-TTS](https://github.com/SWivid/F5-TTS) and [Voxtral](https://mistral.ai/news/voxtral) generate English audio in the original speaker's voice
5. **Audio Alignment** — [pyrubberband](https://github.com/bmcfee/pyrubberband) time-stretches segments that are too long, preserving pitch; [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) normalizes loudness to broadcast standard (-14 LUFS)
6. **Video Merge** — [FFmpeg](https://ffmpeg.org/) replaces the original audio track with the new English one

The result: a video that looks identical to the original, but speaks English — in the same voice.

A [Streamlit](https://streamlit.io/) web interface ties it all together, letting you monitor each stage, review individual segments, edit translations, and inspect quality metrics.

---

## Voice Cloning: The Hardest Part

The most technically demanding piece of the pipeline is voice cloning — and it is worth dwelling on why.

Commercial TTS systems give you a library of pre-made voices. You pick one and type your text. That is convenient but fundamentally limited: the voice is fictional, and you lose the identity of the original speaker.

Zero-shot voice cloning is different. Models like F5-TTS can synthesize speech in an arbitrary speaker's voice using only a short reference audio clip. They learn not just *how to speak*, but *how a specific person speaks* — their timbre, their rhythm, their breath patterns. Feed it a clip of someone, and it will generate new sentences in that voice.

This is powerful. It is also hard to get right across multiple speakers, across videos, across different recording conditions. Our pipeline uses speaker diarization to extract clean reference audio per speaker, enabling consistent voice identity throughout the output.

---

## The Extension: Fine-Tuning for Darija

After completing the main pipeline, we asked a natural question: what would it take to apply this to a language that barely exists in any TTS model's training data?

That question pointed directly at Darija — Moroccan Arabic — for reasons that were both personal and technical.

On the personal side: Darija is the language of everyday life in Morocco. It is the language of family conversations, street markets, and voice messages between friends. The idea that AI speech systems largely do not support it — while supporting dozens of European languages with far fewer speakers — felt worth pushing against.

On the technical side: Darija is a genuine research challenge. It is a low-resource dialect with deep structural differences from Modern Standard Arabic (MSA) — different phonology, a vocabulary that borrows heavily from Amazigh, French, and Spanish, and a rapid consonant-heavy rhythm. It can be written in Arabic script or in Arabizi (a Latin-numeric hybrid where, for example, **3afak** represents **عفاك** — the 3 standing for the Arabic letter ع). Neither form is standardized. When Arabic script is used, short vowels are omitted, as is standard in Arabic writing; vowel diacritics (tashkeel) would help a TTS model but are rarely used in practice and hard to collect at scale.

---

## The State of Darija TTS

Before describing our experiment, it is worth situating it within what already exists.

Commercial services like SpeechGen.io offer native Moroccan Arabic voices — "Mouna" (female) and "Jamal" (male) — explicitly tuned to Darija. These work well for marketing videos or radio promotions. But they use predefined voices. You cannot clone an arbitrary speaker with them.

The open-source landscape is more recent and more interesting. Hugging Face spaces like [medmac01/Darija-Arabic-TTS](https://huggingface.co/spaces/medmac01/Darija-Arabic-TTS) fine-tune XTTS checkpoints for Darija. HAMMALE's speecht5-darija adapts Microsoft's SpeechT5 for Moroccan Arabic. Most recently, the "Habibi" framework (2026) adapted F5-TTS itself to cover 12 regional Arabic dialects using a curriculum training strategy — starting from MSA data and gradually shifting to dialect-specific speech.

The datasets remain the bottleneck. The premier open-source TTS corpus for Darija is DODa (AtlasIA): roughly 9 hours of speech across 7 speakers, with text in Arabic, Latin, and English. Most other large Darija corpora (MGB-5, MoulSot, Casablanca) were built for speech recognition, not synthesis, and lack the standardized orthography synthesis models need.

This data situation shapes everything that follows.

---

## What We Tried

F5-TTS's base vocabulary contains 2,545 character tokens. Arabic characters are not among them. Before any fine-tuning could begin, we had to extend the model's input layer.

The process:

1. **Vocabulary extension** — collect every unique character from the Darija dataset, append the new Arabic ones to the base vocab file
2. **Embedding resize** — the model's text embedding matrix needs a new row for each added character. We initialized new rows with random noise scaled to the standard deviation of existing embeddings, so the model starts with a reasonable prior rather than zeros
3. **Fine-tuning** — train on 20,000 samples from [DarijaTTS-clean](https://huggingface.co/datasets/KandirResearch/DarijaTTS-clean) on an H100 GPU in Google Colab

```python
old_emb = state_dict['ema_model.transformer.text_embed.text_embed.weight']
new_emb = torch.zeros(new_vocab_size, emb_dim)
new_emb[:old_size] = old_emb
std = old_emb.std().item()
new_emb[old_size:] = torch.randn(new_vocab_size - old_size, emb_dim) * std
```

The acoustic architecture — the flow-matching model, the voice cloning mechanism — was preserved entirely. Only the text input representation changed.

---

## What Happened

The fine-tuned model did not produce intelligible Darija. The output had structure — rhythm, voice, the shape of speech — but no recognizable phonemes. The model was clearly generating something, but it had not learned to map the input to actual pronunciation. This was our first fine-tuning attempt, and the results reflect that.

The failure very likely stems from a combination of 2 factors :

- **Not enough data, spread across too many voices**: Teaching F5-TTS an entirely new language typically requires 50 to 100+ hours of audio. DarijaTTS-clean contains around 15–20 hours — already short of that threshold. More importantly, that audio is spread across many different speakers. A multi-speaker dataset forces the model to simultaneously learn the language and adapt to speaker variation, which makes convergence harder. A single-voice dataset of the same size would likely have given the model a much better chance, since it could focus entirely on learning Darija phonetics without the added complexity.
- **Unvocalized text**: DarijaTTS-clean uses Arabic script without tashkeel. Tashkeel would give the model an explicit pronunciation target, but Moroccans do not write Darija with it in everyday life, making a diacritized dataset very hard to build — you cannot scrape it from real-world sources and manual annotation is costly. This alone definitely didn't caused the failure, but it is a known source of input ambiguity for any TTS model trained on Arabic-script Darija data.

---

## What We Learned

**Data quantity and speaker diversity matter as much as the model.** Teaching F5-TTS a new language requires significantly more data than we had — and ideally from a single speaker or a small consistent set, so the model can focus on learning the language rather than simultaneously adapting to many different voices. DarijaTTS-clean's 15–20 hours spread across multiple speakers was not enough for convergence.

**The text input matters as much as the model.** Getting the acoustic architecture right is necessary but not sufficient. If the text representation cannot fully specify pronunciation, the model cannot learn the mapping. For Darija, the missing piece is a G2P layer that handles how the language is actually written — unvocalized Arabic or Arabizi — rather than expecting a sanitized, diacritized input that no one produces in practice.

**A negative result is still a result.** We know exactly where DarijaTTS-clean fails for F5-TTS fine-tuning. We know what a better dataset looks like. The vocabulary extension and embedding resize procedures work correctly — those are reusable building blocks. The failure is localized and diagnosable.

---

## What Is Next

A longer-term direction could be a grapheme-to-phoneme model for Darija, which would normalize written text into a phoneme sequence before passing it to the synthesis model. This is harder than it sounds: Arabic script and Arabizi are structurally different enough that a single G2P model covering both would be noisy and complex to train — separate models per script are probably more realistic, and each one is already a non-trivial task on its own. That said, the payoff would be significant, especially for loanwords: French words like "voiture" or "problème", Amazigh terms, and proper names all get transliterated inconsistently in Arabic script, and a G2P layer would resolve them to the correct Moroccan-inflected pronunciation rather than a broken Arabic reading.

---

## Closing Thought

We did not ship a working Darija TTS system. We built the infrastructure, identified exactly why the first attempt failed, and pointed at what needs to change.

The main pipeline — Chinese-to-English video transformation — worked. The voices were cloned. The timing held. The videos were watchable.

The Darija extension was a shot at something harder. It taught us more.

---

*Adam Maytoussi and Soulaymane Temimi are students at UTSEUS. This work was completed as part of an AI course project. The full pipeline code is available on [GitHub](https://github.com/EmporioSabo/ai-video-language-transformation).*
