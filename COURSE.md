# AI, Explained — from first principles to frontier research

A video course in 46 short episodes (3–5 minutes each), narrated and animated: the ideas of
modern AI, the math behind them made visual, and the innovations researchers work on today.

- **Who it's for:** curious beginners (Level 1) through engineers (Levels 2–4) to researchers
  (Level 5). Each episode stands alone, but lists what to watch first.
- **How every episode works:** one concrete running example from start to finish; a hook, the
  problem, an analogy, the mechanism step by step, the big picture, why it matters (and its
  catch), and a three-word recap. Math appears as formulas *and* as moving pictures.
- **Format:** 1080p60 MP4 with captions (.srt) and an HTML slide version; made with the
  `tutorial-video` skill; series kit in `kit.py`.

Status: ✅ produced · 🎬 in this batch · ○ planned

## Level 1 · Foundations (beginner) — sky blue

| # | Episode | The viewer will understand | Key concepts | Math | Running example | Status |
|---|---|---|---|---|---|---|
| 1 | **What is AI?** | How "learning from examples" differs from writing rules; how AI, ML, deep learning and generative AI nest | rules vs data, traditional programming vs ML, AI ⊃ ML ⊃ DL ⊃ GenAI, narrow vs general AI, milestones 1956–2022 | none | a spam filter | ✅ `ep01-what-is-ai` |
| 2 | **How Machines Learn** | Training as guess → measure → adjust | model, parameters, loss, gradient descent, learning rate, epoch | `y = wx + b`, mean squared error, `w ← w − η·∂L/∂w` | house size → price | ✅ `ep02-how-machines-learn` |
| 3 | **Neural Networks** | What a neuron computes and why layers of them can recognise anything | weighted sum, bias, activation (ReLU), layers, hierarchy of features, why non-linearity | `a = ReLU(w·x + b)` | a handwritten "7" | ✅ `ep03-neural-networks` |
| 4 | **Data: Features, Labels & Splits** | Why data quality decides model quality | features, labels, train / validation / test, leakage, bias in data | proportions | a loan-approval dataset | ✅ `ep04-data-features-labels` |
| 5 | **Overfitting & Generalization** | Memorising vs learning | overfitting, underfitting, bias–variance, regularisation, more data | a polynomial fit | noisy points, polynomial degree 1 → 15 | ✅ `ep05-overfitting` |
| 6 | **Measuring a Model** | Why accuracy can lie | confusion matrix, precision, recall, F1, ROC | ratios | a rare-disease test | ✅ `ep06-measuring-a-model` (ROC not covered) |
| 7 | **The Classic ML Toolkit** | When a simple model is the right model | linear & logistic regression, decision trees, k-NN, k-means | distances | sorting fruit | ✅ `ep07-classic-ml-toolkit` |
| 8 | **Four Ways to Learn** | The learning paradigms | supervised, unsupervised, self-supervised, reinforcement learning | rewards | a robot learning to walk | ✅ `ep08-four-ways-to-learn` |

## Level 2 · The Math of AI (intermediate) — amber

| # | Episode | The viewer will understand | Key concepts | Math | Running example | Status |
|---|---|---|---|---|---|---|
| 9 | **Vectors & Similarity** | Meaning as direction in space | vectors, dot product, norm, cosine similarity | `a·b = ‖a‖‖b‖cos θ` | "king", "queen", "apple" | ✅ `ep09-vectors-similarity` |
| 10 | **Matrices as Transformations** | A layer is a matrix | matrix–vector product, shapes, linear maps, batches | `y = Wx` | rotating and stretching a grid | ✅ `ep10-matrices-transformations` |
| 11 | **Derivatives & Gradients** | Slopes tell you which way is downhill | derivative, partial derivative, gradient, steepest ascent | `∇f = (∂f/∂x, ∂f/∂y)` | a hiker on a foggy mountain | ✅ `ep11-derivatives-gradients` |
| 12 | **The Chain Rule & Backpropagation** | How a network learns all its weights at once | computational graph, chain rule, backward pass, local gradients | `∂L/∂w = ∂L/∂y · ∂y/∂w` | a tiny two-layer network, by hand | ✅ `ep12-chain-rule-backprop` |
| 13 | **Optimizers: SGD, Momentum, Adam** | Why plain gradient descent is not enough | mini-batches, noise, momentum, adaptive learning rates | Adam's update | a ball in a narrow valley | ✅ `ep13-optimizers` |
| 14 | **Probability for ML** | Uncertainty as a number | distributions, expectation, conditional probability, Bayes' rule | `P(A\|B) = P(B\|A)P(A)/P(B)` | a medical test, revisited | ✅ `ep14-probability` |
| 15 | **Maximum Likelihood** | Where loss functions come from | likelihood, log-likelihood, MSE from Gaussian noise | `argmax_θ Σ log p(x\|θ)` | estimating a coin's bias | ✅ `ep15-maximum-likelihood` |
| 16 | **Softmax & Cross-Entropy** | How a classifier turns scores into probabilities and learns from them | logits, softmax, temperature, cross-entropy | `softmax(z)_i = e^{z_i}/Σ e^{z_j}` | cat / dog / bird scores | ✅ `ep16-softmax-cross-entropy` |
| 17 | **Information Theory** | Surprise, measured | entropy, cross-entropy, KL divergence, perplexity | `H(p) = −Σ p log p` | guessing a word | ✅ `ep17-information-theory` |
| 18 | **Eigenvectors, SVD & PCA** | Most data is lower-dimensional than it looks | eigenvectors, SVD, rank, PCA, compression | `A = UΣVᵀ` | compressing a face image | ✅ `ep18-eigen-svd-pca` |

## Level 3 · Deep Learning Core (intermediate) — violet

| # | Episode | The viewer will understand | Key concepts | Math | Running example | Status |
|---|---|---|---|---|---|---|
| 19 | **Activation Functions** | Why the shape of a nonlinearity matters | sigmoid, tanh, ReLU, GELU, vanishing gradients | derivatives of each | a gradient fading through layers | ✅ `ep19-activation-functions` |
| 20 | **Convolutional Networks** | How networks see | filters, feature maps, stride, pooling, translation invariance | 2D convolution | finding edges in a photo | ✅ `ep20-convolutional-networks` |
| 21 | **Recurrent Networks & LSTMs** | Sequences, and why memory is hard | hidden state, unrolling, vanishing gradients, gates | the LSTM cell | predicting the next note of a melody | ✅ `ep21-rnns-lstms` |
| 22 | **Embeddings** | Words, images and users as points in space | one-hot vs dense, word2vec idea, analogies, embedding tables | vector arithmetic | king − man + woman | ✅ `ep22-embeddings` |
| 23 | **Tokenization** | What a language model actually reads | characters vs words vs subwords, BPE, vocabulary size, token costs | merge counts | "unbelievable" → tokens | ✅ `ep23-tokenization` |
| 24 | **Attention & Transformers** | Self-attention and the Transformer block | Q/K/V, attention weights, multi-head, feed-forward, stacking, next-token prediction | weighted sums | "The cat was tired, so it…" | ✅ `ep24-attention-transformers` |
| 25 | **Residuals & Normalization** | How very deep networks train at all | skip connections, LayerNorm, pre-norm, gradient flow | `x + f(x)` | a 100-layer tower | ✅ `ep25-residuals-normalization` |
| 26 | **Regularization in Practice** | Making models generalise | dropout, weight decay, data augmentation, early stopping | L2 penalty | a model that memorised its training set | ✅ `ep26-regularization` |

## Level 4 · Large Language Models & Generative AI (advanced) — teal

| # | Episode | The viewer will understand | Key concepts | Math | Running example | Status |
|---|---|---|---|---|---|---|
| 27 | **Pretraining a Language Model** | What happens in the months before a chatbot exists | next-token objective, web-scale data, compute, checkpoints | cross-entropy per token | one sentence, a trillion times | ✅ `ep27-pretraining` |
| 28 | **Scaling Laws** | Why bigger kept getting better, predictably | loss vs parameters, data and compute; compute-optimal training | power laws | doubling the budget | ✅ `ep28-scaling-laws` |
| 29 | **Fine-Tuning & Instruction Tuning** | From text predictor to assistant | supervised fine-tuning, chat templates, instruction data | — | the same model, before and after | ✅ `ep29-fine-tuning` |
| 30 | **RLHF & DPO** | Teaching models what people prefer | preference data, reward models, PPO idea, DPO | the DPO loss | two answers, one preferred | ✅ `ep30-rlhf-dpo` |
| 31 | **Prompting & In-Context Learning** | Learning without changing a single weight | zero-/few-shot, chain-of-thought, prompt structure | — | a classifier built from a prompt | ✅ `ep31-prompting-icl` |
| 32 | **Retrieval-Augmented Generation** | Grounding answers in your documents | chunking, embeddings, vector search, prompt augmentation, citations | similarity | "refund window for Enterprise plans" | ✅ `ep32-rag` |
| 33 | **Decoding & Sampling** | How the next word is actually chosen | greedy, temperature, top-k, top-p, beam search | softmax with temperature | finishing "Once upon a…" | ✅ `ep33-decoding-sampling` |
| 34 | **VAEs & GANs** | The first deep generative models | latent space, encoder/decoder, reparameterisation, generator vs discriminator | ELBO, minimax | generating faces | ✅ `ep34-vaes-gans` |
| 35 | **Diffusion Models** | Making images by removing noise | forward noising, learned denoising, score, guidance | noise schedule | a photo dissolving into static and back | ✅ `ep35-diffusion` |
| 36 | **Multimodal Models** | Connecting words and images | contrastive learning (CLIP), shared embedding space, vision-language models | InfoNCE | matching captions to photos | ✅ `ep36-multimodal` |

## Level 5 · Research Frontiers (senior) — pink

| # | Episode | The viewer will understand | Key concepts | Math | Running example | Status |
|---|---|---|---|---|---|---|
| 37 | **Efficient Attention** | Why attention is memory-bound and how it's fixed | KV cache, FlashAttention (IO-awareness), multi-query & grouped-query attention | O(n²) memory | serving a long chat | ✅ `ep37-efficient-attention` |
| 38 | **Positional Encodings & Long Context** | How models know word order, and go beyond their training length | sinusoidal, learned, RoPE, context extension | rotations | a 100-page contract | ✅ `ep38-positional-long-context` |
| 39 | **Mixture of Experts** | Huge models that only use part of themselves | experts, routers, top-k routing, load balancing, sparsity | gating | routing tokens to specialists | ✅ `ep39-mixture-of-experts` |
| 40 | **State Space Models & Mamba** | Linear-time alternatives to attention | recurrence, selective state spaces, scans | `h_t = A h_{t−1} + B x_t` | streaming a very long sequence | ✅ `ep40-state-space-mamba` |
| 41 | **LoRA & Parameter-Efficient Fine-Tuning** | Adapting giant models cheaply | low-rank updates, adapters, QLoRA | `W + BA`, rank r | personalising a model on a laptop | ✅ `ep41-lora-peft` |
| 42 | **Quantization & Distillation** | Making models small and fast | int8/int4, outliers, knowledge distillation | rounding error | a model that fits on a phone | ✅ `ep42-quantization-distillation` |
| 43 | **Reasoning Models & Test-Time Compute** | Thinking longer instead of being bigger | chain-of-thought, RL with verifiable rewards, search, self-consistency | majority voting | a hard math word problem | ✅ `ep43-reasoning-test-time` |
| 44 | **Agents & Tool Use** | Models that act | function calling, the reason–act loop, planning, memory | — | booking a trip | ✅ `ep44-agents-tool-use` |
| 45 | **Mechanistic Interpretability** | Looking inside the network | features, superposition, sparse autoencoders, circuits | sparse coding | finding a "Golden Gate" feature | ✅ `ep45-interpretability` |
| 46 | **Alignment & Evaluation** | Measuring and steering model behaviour | benchmarks, contamination, red-teaming, reward hacking, scalable oversight | — | a model gaming its reward | ✅ `ep46-alignment-evaluation` |

## Paths through the course

- **New to AI:** 1 → 2 → 3 → 8 → 22 → 24 → 27 → 32 (what modern AI is and how chatbots work).
- **Engineer building with LLMs:** 3 → 16 → 22 → 23 → 24 → 29 → 31 → 32 → 33 → 41 → 44.
- **Math-first:** 2 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18.
- **Researcher fast-track:** 24 → 25 → 28 → 30 → 35 → 37 → 38 → 39 → 40 → 43 → 45 → 46.

## Production notes

- Episodes 24 and 32 were remade in series format (`ep24-attention-transformers`, `ep32-rag`);
  the original standalone films stay in `~/videos/transformer` and `~/rag-video`.
- Episodes 9–46 were made by subagents in a rolling window of five to six, one shared narrator;
  every episode was then checked by the coordinator (storyboards, numbers, `verify_audio`).
- Kit-level fixes applied to every film by re-rendering with cached audio: TransformFromCopy /
  FadeTransform shown early, parts of groups shown early (`Episode.say`), long titles fitted.
- Fast-moving topics (Level 5, parts of Level 4) should be reviewed for accuracy before each
  release; record the date of the facts in the episode's section notes.
- Folder per episode: `epNN-slug/scene.py` importing `../kit.py`; shared `assets/`,
  `lexicon.json` (pronunciations) and one narrator for the whole series.
- Episodes 4–8 were made by five parallel subagents (brief: the skill's
  `templates/episode_brief.md`), each reviewed afterwards. Example data is made up for
  teaching (the loan table, the disease counts, the robot's rewards); the fits in 5 and 7 are
  real, computed from seeded numpy; 6 and 7 assert that narrated numbers match computed ones.
- Episode 6 leaves out the ROC curve (listed in its concepts); add it to Ep 14 (probability,
  the medical test revisited) or a short follow-up.
