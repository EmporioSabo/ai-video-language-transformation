# From Chinese to English — and a Detour into Moroccan Arabic: Building an AI Video Language Transformation Pipeline

**Authors: Adam Maytoussi, Soulaymane Temimi**

---

## The Starting Point

This semester, our AI course asked us to pick a project and push it further — past the tutorial, past the ready-made tool, into something that required us to understand the underlying machinery.

We picked **AI Video Language Transformation**: a pipeline that takes Chinese-language GitHub tutorial videos and converts them into English — same video, new audio, voice cloned from the original speaker. The idea was to make technical content accessible across language barriers without losing the personality of the original presenter.

That was the main project. But by the end of the project, we had also taken a detour into something much less explored: fine-tuning a voice cloning model for Darija — Moroccan Arabic. What happened during that detour is, honestly, the more interesting story.

---

## The Pipeline

The main pipeline chains together several AI components, each handling a distinct task:

1. **Transcription** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (large-v3) converts the Chinese audio into timed text segments
2. **Speaker Diarization** — [pyannote.audio](https://github.com/pyannote/pyannote-audio) identifies who is speaking and when, enabling per-speaker voice cloning
3. **Translation** — [DeepL](https://www.deepl.com/) translates Chinese segments to English, with a [Gemini](https://deepmind.google/technologies/gemini/) review pass for technical terminology (GitHub-specific vocabulary like "pull request", "commit", "branch")
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

On the technical side: Darija is a genuine research challenge. It is a low-resource dialect with deep structural differences from Modern Standard Arabic (MSA) — different phonology, a vocabulary that borrows heavily from Amazigh, French, and Spanish, and a rapid consonant-heavy rhythm that standard Arabic models are not built to handle.

But there is an added layer of complexity that makes Darija uniquely difficult, even by the standards of low-resource languages: **there is no standardized way to write it**.

---

## The Writing Problem

Depending on who is typing — and on what platform — Darija might appear as:

- Arabic script: **عفاك** ("Please" / "excuse me")
- Latin script with numbers substituting for sounds that have no Latin equivalent: **3afak** (the number 3 represents the Arabic letter ع, a pharyngeal consonant with no equivalent in French or English)
- A mix of both, sometimes within the same sentence

This is not a niche phenomenon. Arabizi — the Latin-numeric hybrid — is the dominant register for Darija on WhatsApp, Instagram, and most social media. Arabic script is used in more formal contexts. Neither is standardized. Neither has agreed-upon spelling rules. Two people writing the same Darija sentence might produce completely different character sequences.

For a TTS model, this is a fundamental problem. Before you can train a model to speak a word, you need to know how that word is spelled — consistently, across thousands of training examples. Arabic script at least gives you a consistent character set. But even in Arabic, Darija adds another layer: **the vowel problem**.

Written Arabic omits short vowels. The consonant skeleton is written; the reader fills in the vowels from morphological knowledge and context. A native speaker can do this automatically. A TTS model cannot — it needs either explicit vowel markings (called *shakl* or *tashkeel* in Arabic) or a separate system that infers pronunciation from the written form.

These two problems — no standardized orthography, and unvocalized text even when Arabic script is used — are the core reasons why Darija TTS lags so far behind MSA.

---

## The State of Darija TTS

Before describing our experiment, it is worth situating it within what already exists.

Commercial services like SpeechGen.io and ElevenLabs have launched Moroccan Arabic voices — "Mouna", "Jamal", "Ghizlane" — that capture the cadence of Casablanca or Rabat convincingly. These work well for marketing videos or radio promotions. But they use predefined voices. You cannot clone an arbitrary speaker with them.

The open-source landscape is more recent and more interesting. Hugging Face spaces like [medmac01/Darija-Arabic-TTS](https://huggingface.co/spaces/medmac01/Darija-Arabic-TTS) fine-tune XTTS checkpoints for Darija. HAMMALE's speecht5-darija adapts Microsoft's SpeechT5 for Moroccan Arabic. Most recently, the "Habibi" framework (2026) adapted F5-TTS itself to cover 12 regional Arabic dialects using a curriculum training strategy — starting from MSA data and gradually shifting to dialect-specific speech.

The datasets remain the bottleneck. The premier open-source TTS corpus for Darija is DODa (AtlasIA): roughly 9 hours of speech across 7 speakers, with text in Arabic, Latin, and English. High quality — but no vowel diacritics. Most other large Darija corpora (MGB-5, MoulSot, Casablanca) were built for speech recognition, not synthesis, and lack the standardized orthography synthesis models need. The one corpus designed for TTS that includes proper diacritization — lahgtna-chatterbox — remains relatively small.

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

The fine-tuned model produced gibberish.

Not noise — structured audio that sounded like an attempt at speech, with rhythm and voice, but no recognizable phonemes. The model was generating *something*, but not Darija.

The culprit, we believe, is exactly the problem described above: unvocalized text. DarijaTTS-clean uses Arabic script without shakl. The model was trying to learn a mapping from ambiguous consonant skeletons to speech, with no consistent signal for how vowels should be realized. Given that Darija also systematically drops short vowels compared to MSA (for example, "he wrote" is /kteb/ in Darija versus /kataba/ in MSA), the training signal was doubly ambiguous. The model collapsed.

This is not a failure unique to us. It mirrors what the broader literature reports: every serious Darija TTS effort has had to confront the vocalization problem one way or another. The "Habibi" framework sidestepped it through curriculum learning from diacritized MSA. Others have explored grapheme-to-phoneme (G2P) models that infer pronunciation without explicit diacritics. Both are real paths — and neither is trivial.

We found the lahgtna-chatterbox corpus, which supports diacritized text, but ran out of time to run a second experiment.

---

## What We Learned

**The text input matters as much as the model.** Getting the acoustic architecture right is necessary but not sufficient. If the text representation cannot fully specify pronunciation, the model cannot learn the mapping. For Arabic dialects, that means either diacritized training data or a dedicated G2P preprocessing step.

**Darija's writing chaos is itself a research problem.** Before you can train a model, you need clean, consistently-formatted text. For a language with no agreed-upon orthography, that normalization step is non-trivial and probably deserves its own model.

**A negative result is still a result.** We know exactly where DarijaTTS-clean fails for F5-TTS fine-tuning. We know what a better dataset looks like. The vocabulary extension and embedding resize procedures work correctly — those are reusable building blocks. The failure is localized and diagnosable.

**Zero-shot voice cloning is the right goal for Darija.** Commercial Darija TTS exists, but it gives you fixed voices. A zero-shot system would let anyone clone their own voice — for accessibility tools, for education, for content creation in their own dialect. That capability, which works well for English and Chinese, is worth bringing to Darija.

---

## What Is Next

The most direct next step: repeat the experiment with a diacritized dataset. The lahgtna-chatterbox corpus was designed for this purpose and is the obvious candidate.

A longer-term direction: a grapheme-to-phoneme model for Darija — one that handles both Arabic script and Arabizi — which would normalize any Darija text into a phoneme sequence before passing it to the synthesis model. This would make the system robust to the orthographic variation that is a fact of life for anyone writing Darija.

---

## Closing Thought

We did not ship a working Darija TTS system. We built the infrastructure, identified exactly why the first attempt failed, and pointed at what needs to change.

The main pipeline — Chinese-to-English video transformation — worked. The voices were cloned. The timing held. The videos were watchable.

The Darija extension was a shot at something harder. It taught us more.

---

*Adam Maytoussi and Soulaymane Temimi are students at UTSEUS. This work was completed as part of an AI course project. The full pipeline code is available on [GitHub](https://github.com/EmporioSabo/ai-video-language-transformation).*
