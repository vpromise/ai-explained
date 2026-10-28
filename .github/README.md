# AI, Explained — from first principles to frontier research

A video course in 46 short episodes (3–5 minutes each), narrated and animated: the ideas of modern AI, the math behind them made visual, and the innovations researchers work on today.

Part of a family of narrated, animated video courses, rendered with [ManimGX](https://pypi.org/project/manimgx/) and voiced by a local narrator ([tutorial-video](https://github.com/vpromise/tutorial-video)):

- **AI, Explained** — 46 episodes on modern AI, from first principles to frontier research *(this repository)*
- **Agents, Animated** — how AI agents work, from one loop to a team · [agents-animated](https://github.com/vpromise/agents-animated)
- **Tokens per Second** — how LLM inference really works · [tokens-per-second](https://github.com/vpromise/tokens-per-second)
- **Trial and Reward** — reinforcement learning, from slot machines to reasoning models · [trial-and-reward](https://github.com/vpromise/trial-and-reward)
- **Out of the Noise** — how diffusion models make pictures and video · [out-of-the-noise](https://github.com/vpromise/out-of-the-noise)

This repository holds the **source code** of the course: one scene per episode, the series kit, reference code, research notes and the production tools. The rendered videos are not included.

## Episodes

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 1 | What is AI? | How "learning from examples" differs from writing rules; how AI, ML, deep learning and generative AI nest | 2:34 | [`ep01-what-is-ai/scene.py`](../ep01-what-is-ai/scene.py) |
| 2 | How Machines Learn | Training as guess → measure → adjust | 2:35 | [`ep02-how-machines-learn/scene.py`](../ep02-how-machines-learn/scene.py) |
| 3 | Neural Networks | What a neuron computes and why layers of them can recognise anything | 2:18 | [`ep03-neural-networks/scene.py`](../ep03-neural-networks/scene.py) |
| 4 | Data: Features, Labels & Splits | Why data quality decides model quality | 3:22 | [`ep04-data-features-labels/scene.py`](../ep04-data-features-labels/scene.py) |
| 5 | Overfitting & Generalization | Memorising vs learning | 3:23 | [`ep05-overfitting/scene.py`](../ep05-overfitting/scene.py) |
| 6 | Measuring a Model | Why accuracy can lie | 3:15 | [`ep06-measuring-a-model/scene.py`](../ep06-measuring-a-model/scene.py) |
| 7 | The Classic ML Toolkit | When a simple model is the right model | 3:18 | [`ep07-classic-ml-toolkit/scene.py`](../ep07-classic-ml-toolkit/scene.py) |
| 8 | Four Ways to Learn | The learning paradigms | 3:28 | [`ep08-four-ways-to-learn/scene.py`](../ep08-four-ways-to-learn/scene.py) |
| 9 | Vectors & Similarity | Meaning as direction in space | 3:29 | [`ep09-vectors-similarity/scene.py`](../ep09-vectors-similarity/scene.py) |
| 10 | Matrices as Transformations | A layer is a matrix | 3:26 | [`ep10-matrices-transformations/scene.py`](../ep10-matrices-transformations/scene.py) |
| 11 | Derivatives & Gradients | Slopes tell you which way is downhill | 3:28 | [`ep11-derivatives-gradients/scene.py`](../ep11-derivatives-gradients/scene.py) |
| 12 | The Chain Rule & Backpropagation | How a network learns all its weights at once | 3:49 | [`ep12-chain-rule-backprop/scene.py`](../ep12-chain-rule-backprop/scene.py) |
| 13 | Optimizers: SGD, Momentum, Adam | Why plain gradient descent is not enough | 3:47 | [`ep13-optimizers/scene.py`](../ep13-optimizers/scene.py) |
| 14 | Probability for ML | Uncertainty as a number | 3:29 | [`ep14-probability/scene.py`](../ep14-probability/scene.py) |
| 15 | Maximum Likelihood | Where loss functions come from | 4:04 | [`ep15-maximum-likelihood/scene.py`](../ep15-maximum-likelihood/scene.py) |
| 16 | Softmax & Cross-Entropy | How a classifier turns scores into probabilities and learns from them | 3:26 | [`ep16-softmax-cross-entropy/scene.py`](../ep16-softmax-cross-entropy/scene.py) |
| 17 | Information Theory | Surprise, measured | 3:44 | [`ep17-information-theory/scene.py`](../ep17-information-theory/scene.py) |
| 18 | Eigenvectors, SVD & PCA | Most data is lower-dimensional than it looks | 3:32 | [`ep18-eigen-svd-pca/scene.py`](../ep18-eigen-svd-pca/scene.py) |
| 19 | Activation Functions | Why the shape of a nonlinearity matters | 3:43 | [`ep19-activation-functions/scene.py`](../ep19-activation-functions/scene.py) |
| 20 | Convolutional Networks | How networks see | 3:51 | [`ep20-convolutional-networks/scene.py`](../ep20-convolutional-networks/scene.py) |
| 21 | Recurrent Networks & LSTMs | Sequences, and why memory is hard | 3:54 | [`ep21-rnns-lstms/scene.py`](../ep21-rnns-lstms/scene.py) |
| 22 | Embeddings | Words, images and users as points in space | 3:19 | [`ep22-embeddings/scene.py`](../ep22-embeddings/scene.py) |
| 23 | Tokenization | What a language model actually reads | 3:58 | [`ep23-tokenization/scene.py`](../ep23-tokenization/scene.py) |
| 24 | Attention & Transformers | Self-attention and the Transformer block | 3:50 | [`ep24-attention-transformers/scene.py`](../ep24-attention-transformers/scene.py) |
| 25 | Residuals & Normalization | How very deep networks train at all | 3:29 | [`ep25-residuals-normalization/scene.py`](../ep25-residuals-normalization/scene.py) |
| 26 | Regularization in Practice | Making models generalise | 3:29 | [`ep26-regularization/scene.py`](../ep26-regularization/scene.py) |
| 27 | Pretraining a Language Model | What happens in the months before a chatbot exists | 3:35 | [`ep27-pretraining/scene.py`](../ep27-pretraining/scene.py) |
| 28 | Scaling Laws | Why bigger kept getting better, predictably | 3:46 | [`ep28-scaling-laws/scene.py`](../ep28-scaling-laws/scene.py) |
| 29 | Fine-Tuning & Instruction Tuning | From text predictor to assistant | 3:53 | [`ep29-fine-tuning/scene.py`](../ep29-fine-tuning/scene.py) |
| 30 | RLHF & DPO | Teaching models what people prefer | 4:02 | [`ep30-rlhf-dpo/scene.py`](../ep30-rlhf-dpo/scene.py) |
| 31 | Prompting & In-Context Learning | Learning without changing a single weight | 3:30 | [`ep31-prompting-icl/scene.py`](../ep31-prompting-icl/scene.py) |
| 32 | Retrieval-Augmented Generation | Grounding answers in your documents | 3:56 | [`ep32-rag/scene.py`](../ep32-rag/scene.py) |
| 33 | Decoding & Sampling | How the next word is actually chosen | 3:27 | [`ep33-decoding-sampling/scene.py`](../ep33-decoding-sampling/scene.py) |
| 34 | VAEs & GANs | The first deep generative models | 4:01 | [`ep34-vaes-gans/scene.py`](../ep34-vaes-gans/scene.py) |
| 35 | Diffusion Models | Making images by removing noise | 4:09 | [`ep35-diffusion/scene.py`](../ep35-diffusion/scene.py) |
| 36 | Multimodal Models | Connecting words and images | 3:21 | [`ep36-multimodal/scene.py`](../ep36-multimodal/scene.py) |
| 37 | Efficient Attention | Why attention is memory-bound and how it's fixed | 4:17 | [`ep37-efficient-attention/scene.py`](../ep37-efficient-attention/scene.py) |
| 38 | Positional Encodings & Long Context | How models know word order, and go beyond their training length | 4:03 | [`ep38-positional-long-context/scene.py`](../ep38-positional-long-context/scene.py) |
| 39 | Mixture of Experts | Huge models that only use part of themselves | 3:58 | [`ep39-mixture-of-experts/scene.py`](../ep39-mixture-of-experts/scene.py) |
| 40 | State Space Models & Mamba | Linear-time alternatives to attention | 3:58 | [`ep40-state-space-mamba/scene.py`](../ep40-state-space-mamba/scene.py) |
| 41 | LoRA & Parameter-Efficient Fine-Tuning | Adapting giant models cheaply | 4:16 | [`ep41-lora-peft/scene.py`](../ep41-lora-peft/scene.py) |
| 42 | Quantization & Distillation | Making models small and fast | 4:03 | [`ep42-quantization-distillation/scene.py`](../ep42-quantization-distillation/scene.py) |
| 43 | Reasoning Models & Test-Time Compute | Thinking longer instead of being bigger | 3:50 | [`ep43-reasoning-test-time/scene.py`](../ep43-reasoning-test-time/scene.py) |
| 44 | Agents & Tool Use | Models that act | 3:42 | [`ep44-agents-tool-use/scene.py`](../ep44-agents-tool-use/scene.py) |
| 45 | Mechanistic Interpretability | Looking inside the network | 3:60 | [`ep45-interpretability/scene.py`](../ep45-interpretability/scene.py) |
| 46 | Alignment & Evaluation | Measuring and steering model behaviour | 3:54 | [`ep46-alignment-evaluation/scene.py`](../ep46-alignment-evaluation/scene.py) |

## Repository layout

| Path | What |
|---|---|
| `COURSE.md` | the full syllabus: key concepts, what each episode animates, the running example |
| `kit.py` | the series kit: palette, fonts, components, intro and outro cards |
| `epNN-*/scene.py` | one narrated scene per episode (`self.say("…[cue]…", animation)`: the narration drives the timing) |
| `assets/` | icons and the backdrop |
| `lexicon.json` | pronunciation overrides for the narrator |
| `briefs/` | the production briefs each episode was made from, and the QA tools (`check.sh`, `make_index.py`) |
| `LISTEN.md` | narration lines flagged for a human listen |

## Rendering an episode

Requirements: macOS on Apple silicon (the narrator runs Chatterbox on MPS and Whisper on MLX), ManimGX (`uv tool install --python 3.13 manimgx`) and the [tutorial-video](https://github.com/vpromise/tutorial-video) toolkit (set it up with its `scripts/setup_tts.sh`).

```sh
cd epNN-*/
VOICE=draft manimgx inspect scene.py      # silent draft: layout check + storyboards
# in the course folder, start the narrator (it reads lexicon.json):
#   <tutorial-video>/scripts/narrator.sh
manimgx render scene.py                    # MP4 + SRT captions (speaks new lines once, then caches them)
manimgx present scene.py --no-open         # HTML slides
```

## Not in this repository

Rendered MP4s, captions, HTML slides, the narration cache (`voice/`), storyboards and logs.
