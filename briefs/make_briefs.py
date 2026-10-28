"""Write one subagent brief per episode (briefs/epNN.md) from the skill's template, the rows of
COURSE.md, and the per-episode notes below. Run: python3 briefs/make_briefs.py [N ...]"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERIES_DIR = HERE.parent
TEMPLATE = Path.home() / ".claude/skills/tutorial-video/templates/episode_brief.md"
NARRATOR_PYTHON = "~/rag-video/tts/.venv/bin/python"

LEVELS = {
    2: ("THE MATH OF AI", "intermediate (comfortable with school algebra; has watched Level 1)"),
    3: ("DEEP LEARNING", "intermediate (knows the Level 1–2 ideas: training, gradients, vectors, matrices)"),
    4: ("LLMs & GENERATIVE AI", "advanced (developers and engineers who build with models)"),
    5: ("RESEARCH FRONTIERS", "senior (ML engineers and researchers; precise, no hand-waving, cite papers)"),
}

FINISHED = {
    1: "ep01-what-is-ai (concepts, nested rings, a timeline)",
    2: "ep02-how-machines-learn (math-heavy: formulas, a loss valley, gradient descent animated)",
    3: "ep03-neural-networks (network diagrams, a forward pass with real numbers)",
    4: "ep04-data-features-labels (tables, splits, data pitfalls)",
    5: "ep05-overfitting (genuine computed plots from seeded numpy, live readouts, curve sweeps)",
    6: "ep06-measuring-a-model (a grid of people, a confusion matrix, ratios built from counts)",
    7: "ep07-classic-ml-toolkit (five models on one dataset, asserts tying numbers to narration)",
    8: "ep08-four-ways-to-learn (four paradigms side by side, an agent–environment loop)",
}

# number: (slug, ClassName, closest reference episodes, extra notes)
SPEC = {
    9: ("vectors-similarity", "VectorsSimilarity", [2, 7, 5],
        "Use toy 2-D (and one 3-D) vectors for king, queen, apple and say plainly that they are toys: "
        "real embeddings have hundreds or thousands of dimensions and are learned (Episode 22 covers how). "
        "Compute every dot product, norm and cosine with numpy and assert them. Show the angle between arrows "
        "changing as cosine moves from 1 to 0 to −1; contrast the dot product (grows with length) with cosine "
        "(direction only). Mention that search engines and RAG rank by cosine similarity (callback: Episode 32)."),
    10: ("matrices-transformations", "MatricesTransformations", [2, 3, 5],
        "Animate a grid under 2×2 matrices (rotation, stretch, shear) with ApplyMatrix-style transforms; the key "
        "insight: the columns of W are where the basis vectors land. Show y = Wx with real numbers worked through, "
        "shapes (m×n times n gives m), composing two transformations = multiplying matrices (order matters), and "
        "a batch as stacking inputs into a matrix. Close the loop: a neural-network layer is Wx + b followed by an "
        "activation (callback: Episode 3)."),
    11: ("derivatives-gradients", "DerivativesGradients", [2, 5, 3],
        "The hiker in fog feels the slope underfoot. Derivative = slope of the tangent, as the limit of rise over "
        "run (animate the secant shrinking). Partial derivatives: slope along one axis with the other held fixed. "
        "Use a concrete surface f(x, y) (e.g. a tilted bowl) with analytic gradients computed in code; draw contour "
        "lines and gradient arrows perpendicular to them; the gradient points steepest uphill, so descent follows "
        "−∇f (callback: Episode 2's ball in the valley). Keep 3-D minimal: contours plus a side view work well."),
    12: ("chain-rule-backprop", "ChainRuleBackprop", [2, 3, 7],
        "Work a tiny network by hand with real numbers (one input, a hidden ReLU unit or two, one output, squared "
        "loss): forward pass values on the graph, then the backward pass multiplying local gradients along each "
        "path, and summing where paths meet. Compute every number in code and assert it; show the weight update "
        "for one step. Say that frameworks (PyTorch, JAX) do this automatically (automatic differentiation), and "
        "the backward pass costs about the same order as the forward pass (roughly twice). 'ReLU' is in the "
        "lexicon and passes QC."),
    13: ("optimizers", "Optimizers", [2, 5, 7],
        "A narrow, elongated valley (an ill-conditioned quadratic): compute the real trajectories of plain "
        "gradient descent (zig-zag), momentum, and Adam with numpy from the same start, and animate them. "
        "Mini-batches: gradients from a sample are noisy estimates (show noisy arrows around the true one). "
        "Adam (Kingma & Ba, 2014): m ← β₁m + (1−β₁)g, v ← β₂v + (1−β₂)g², bias-corrected, step −η·m̂/(√v̂ + ε); "
        "defaults β₁ = 0.9, β₂ = 0.999. Mention AdamW (decoupled weight decay) as today's common default for "
        "large models. Pronunciation: 'Adam' is transcribed 'Atom' by Whisper; the QC now treats them as one, so "
        "say 'Adam' freely; 'SGD' passes."),
    14: ("probability", "Probability", [6, 2, 5],
        "This is Episode 6's rare-disease test, revisited: reuse its exact numbers for continuity — 1,000 people, "
        "10 sick; the test flags 8 of them (2 missed) and 40 of the 990 healthy (950 cleared). So P(sick) = 1%, "
        "P(positive | sick) = 80%, P(positive | healthy) = 40/990 ≈ 4%, and P(sick | positive) = 8/48 ≈ 17%. Build "
        "Bayes' rule from the counts first, then as the formula. Also: a distribution (a bar chart over outcomes, "
        "then a bell curve as a smooth limit), expectation as a probability-weighted average, and that a "
        "classifier's output is a probability (callback: Episode 16 will show how). Compute and assert all numbers."),
    15: ("maximum-likelihood", "MaximumLikelihood", [2, 5, 14],
        "Estimate a coin's bias: e.g. 7 heads in 10 flips. Plot the likelihood L(θ) = θ⁷(1−θ)³ (computed) and show "
        "its peak at θ = 0.7; then the log-likelihood (same peak, sums instead of products, numerically stable). "
        "Then the big reveal: assuming Gaussian noise, maximising likelihood is exactly minimising squared error "
        "(Episode 2's loss), and for yes/no labels it gives cross-entropy (Episode 16). Keep the derivation honest "
        "but light: show the log turning a product into a sum, and the Gaussian exponent becoming (y − ŷ)²."),
    16: ("softmax-cross-entropy", "SoftmaxCrossEntropy", [2, 3, 7],
        "Cat / dog / bird logits, e.g. [2.0, 1.0, 0.1] → softmax ≈ [0.66, 0.24, 0.10] (compute exactly and assert). "
        "Show exponentiate-then-normalise as bars. Temperature: divide logits by T; low T sharpens, high T flattens "
        "(animate). Cross-entropy for the true class: −log p(correct); show the −log curve (confident and wrong is "
        "punished hard). The neat result: the gradient with respect to the logits is p − y (predicted minus "
        "one-hot), which nudges the right score up and the others down. Callback: temperature returns in Episode 33."),
    17: ("information-theory", "InformationTheory", [2, 5, 6],
        "Guessing a word: surprise = −log₂ p, in bits. A fair coin is 1 bit; a 1-in-8 outcome is 3 bits. Entropy "
        "H(p) = −Σ p log p is the average surprise; show it for a fair vs a loaded coin (the H(p) curve peaking at "
        "0.5). Cross-entropy H(p, q): average surprise when you believe q but the world is p; KL(p‖q) = H(p, q) − "
        "H(p) ≥ 0, and it is not symmetric (show both directions numerically). Perplexity = 2^H (bits) = the "
        "effective number of equally likely choices; a language model's perplexity is exp of its average "
        "cross-entropy per token in nats. Compute all numbers in code."),
    18: ("eigen-svd-pca", "EigenSVDPCA", [2, 5, 7],
        "No real photographs: generate a synthetic grayscale 'face' image in code (shapes and smooth gradients, "
        "e.g. 64×64) and compute its SVD with numpy; show rank-1, 5, 20 reconstructions with their real errors and "
        "storage counts. Eigenvectors: directions a matrix only stretches (Av = λv), shown on a transformed grid "
        "(callback: Episode 10). SVD A = UΣVᵀ as rotate–stretch–rotate; singular values sorted, most energy in the "
        "first few. PCA: centre the data, the top directions of variance (the top singular vectors / eigenvectors "
        "of the covariance); show a 2-D point cloud with its principal axes computed in code."),
    19: ("activation-functions", "ActivationFunctions", [3, 2, 5],
        "Plot sigmoid, tanh, ReLU, GELU and their derivatives (computed). The vanishing-gradient demo: sigmoid's "
        "slope is at most 0.25, so through 10 layers the gradient can shrink by 0.25¹⁰ ≈ 10⁻⁶ (show a bar fading "
        "layer by layer, computed); ReLU's slope is 1 for positive inputs; 'dead' ReLUs output zero forever if "
        "pushed negative. GELU = x·Φ(x), a smooth ReLU, used in BERT and GPT-style Transformers. Pronunciation: "
        "'ReLU' (lexicon) and 'GELU' pass QC; you may also say 'Gaussian error linear unit' once."),
    20: ("convolutional-networks", "ConvolutionalNetworks", [3, 5, 7],
        "No real photographs: build a synthetic grayscale image in code (shapes, a house, a letter) and run real "
        "3×3 filters over it with numpy (a vertical-edge Sobel kernel, a horizontal one); show the kernel sliding, "
        "one multiply-and-sum worked with numbers, and the resulting feature map. Stride and padding briefly; 2×2 "
        "max pooling; stacking layers → edges → shapes → objects (callback: Episode 3's features). Weight sharing: "
        "a 3×3 filter has 9 weights whatever the image size (compare to a dense layer's count). Be precise: "
        "convolution is translation-equivariant; pooling adds some invariance. Milestone: AlexNet won ImageNet in 2012."),
    21: ("rnns-lstms", "RNNsLSTMs", [3, 8, 2],
        "Next note of a melody: draw notes on a small staff or as a piano-roll. RNN: h_t = tanh(W·h_{t−1} + U·x_t + b), "
        "the same weights at every step; unroll it over time. Training through time multiplies many factors, so "
        "gradients vanish or explode (show a computed product shrinking). LSTM (Hochreiter & Schmidhuber, 1997): a "
        "cell state with an additive path, and forget, input and output gates (sigmoids between 0 and 1) deciding "
        "what to erase, write and show. Close: Transformers (Episode 24) replaced RNNs for most language tasks by "
        "reading all positions at once; state space models (Episode 40) revive recurrence."),
    22: ("embeddings", "Embeddings", [3, 7, 5],
        "One-hot vectors (huge, all words equally far apart) vs dense embeddings (a few hundred numbers; similar "
        "words close). An embedding table is a matrix; looking up a word = taking its row (equivalently one-hot × "
        "matrix). word2vec (Mikolov et al., 2013): learn vectors by predicting a word's neighbours. king − man + "
        "woman lands near queen: show with toy 2-D vectors and say it is approximately true in real models (and "
        "the input words are excluded from the search). Embeddings exist for images, users, products too. "
        "Callbacks: Episode 9 (cosine similarity), Episode 23 (what the tokens are)."),
    23: ("tokenization", "Tokenization", [7, 4, 5],
        "'unbelievable' → tokens. Contrast characters (long sequences), words (huge vocabulary, unknown words) and "
        "subwords. Byte-pair encoding: run a real toy BPE in code on a tiny corpus and animate the merges by "
        "frequency; say the exact split of a word depends on each model's tokenizer. Vocabulary sizes: tens of "
        "thousands to a couple of hundred thousand (GPT-2 used 50,257). Rough rule for English: a token is about "
        "three-quarters of a word. Why it matters: models are priced and limited in tokens; other languages and "
        "code can cost more tokens; odd failures (spelling, counting letters) come from tokens. 'BPE' passes QC."),
    24: ("attention-transformers", "AttentionTransformers", [3, 7, 2],
        "PORT, NOT FROM SCRATCH: a standalone film on this topic exists at ~/videos/transformer/ (scene.py, "
        "TransformerExplainer.mp4, 3:26). Read its scene.py and storyboards, then remake it as a series episode "
        "with kit.py: the Episode base class (intro/outro cards come from the kit, so drop its own), kit "
        "components and headers in the Level 3 violet accent, the series' recap style. Keep its running example "
        "('The cat was tired, so it…'), its narration where it works, and its best visuals; tighten to ~3.5–4 min; "
        "add callbacks (Episodes 16 softmax, 22 embeddings, 23 tokens; Episode 37 for efficient attention). Facts: "
        "Q/K/V, scaled dot-product attention softmax(QKᵀ/√d)V, multi-head, feed-forward, residuals + LayerNorm "
        "(Episode 25), stacking, next-token prediction; Vaswani et al., 2017. NEVER modify anything in "
        "~/videos/transformer/; copy any icon you need into the series assets as ep24-*.svg."),
    25: ("residuals-normalization", "ResidualsNormalization", [3, 5, 2],
        "A 100-layer tower: simulate in numpy how a signal / gradient magnitude evolves through 100 plain layers vs "
        "100 residual layers (x + f(x)) and plot both (computed, seeded). ResNet (He et al., 2015) trained networks "
        "of 152 layers. The skip connection is a gradient highway: the derivative of x + f(x) includes 1. "
        "LayerNorm: per token, subtract the mean and divide by the standard deviation across features, then a "
        "learned scale and shift (show numbers). Pre-norm (normalise before each sub-layer, used by most modern "
        "LLMs) vs post-norm (the original 2017 Transformer); RMSNorm is a common lighter variant."),
    26: ("regularization", "Regularization", [5, 7, 4],
        "A model that memorised its training set (callback: Episode 5). Show train vs validation loss curves "
        "from a small real numpy experiment (seeded) and how each remedy changes them: dropout (Srivastava et al., "
        "2014; randomly zero units during training, use all at test time with scaling), weight decay / L2 penalty "
        "(loss + λΣw²; in AdamW it is decoupled), data augmentation (flips, crops, noise on a synthetic image), "
        "early stopping (stop at the validation minimum). Be careful to call any curve you did not compute "
        "'illustrative' in the section notes."),
    27: ("pretraining", "Pretraining", [2, 8, 5],
        "One sentence, a trillion times: the next-token objective with cross-entropy per token (callbacks: 16, "
        "23). Show a sentence predicting each next token with a probability bar and its loss. Scale, kept "
        "conservative: training sets of trillions of tokens (Llama 3 reported about 15 trillion), thousands of "
        "GPUs for weeks to months; checkpoints saved along the way; a falling loss curve (illustrative, say so). "
        "Data pipeline: crawl, filter, deduplicate, mix. What comes out is a text predictor, not yet an assistant "
        "(Episode 29). Avoid exact cost figures unless widely reported, and hedge them."),
    28: ("scaling-laws", "ScalingLaws", [5, 2, 7],
        "Power laws as straight lines on log–log axes: plot L(N) = (N_c/N)^α style curves (computed from a stated "
        "illustrative formula). Kaplan et al. (2020): loss falls smoothly as a power law in parameters, data and "
        "compute. Chinchilla (Hoffmann et al., 2022): for a fixed compute budget, scale parameters and tokens "
        "together, about 20 tokens per parameter; training compute ≈ 6·N·D FLOPs. 'Doubling the budget' as the "
        "running example: compute how N and D should grow. Today many models are trained far past that ratio, "
        "because smaller models are cheaper to serve. Hedge: scaling laws are empirical fits, not laws of nature."),
    29: ("fine-tuning", "FineTuning", [8, 4, 2],
        "The same model before and after: a pretrained model continues text (it might answer a question with more "
        "questions); after supervised fine-tuning on (instruction, response) pairs it answers. Show a chat "
        "template with roles (system, user, assistant) and that the loss is usually computed only on the "
        "assistant's tokens. Data: thousands to millions of curated examples; quality beats quantity. InstructGPT "
        "(Ouyang et al., 2022) as the classic recipe: SFT, then preference tuning (Episode 30). Mention full "
        "fine-tuning vs cheap adapters (Episode 41)."),
    30: ("rlhf-dpo", "RLHFDPO", [8, 2, 6],
        "Two answers, one preferred. Preference data from people; a reward model trained with the Bradley–Terry "
        "model P(A ≻ B) = σ(r_A − r_B); then PPO maximises reward with a KL penalty that keeps the policy near the "
        "SFT model (show the trade-off). DPO (Rafailov et al., 2023): skip the reward model; loss = −log σ(β[log "
        "π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x)]) — show it and explain each part (raise the preferred "
        "answer's probability relative to the reference, lower the rejected one's). Compute a numeric example. "
        "Mention reward hacking (Episode 46). 'RLHF', 'DPO', 'PPO' pass QC."),
    31: ("prompting-icl", "PromptingICL", [8, 4, 7],
        "A sentiment classifier built from a prompt, with zero weights changed. Zero-shot vs few-shot (GPT-3, Brown "
        "et al., 2020) shown as prompts with examples; chain-of-thought (Wei et al., 2022): asking for steps helps "
        "on multi-step problems. Prompt structure: role, task, context, examples, output format. Show prompts as "
        "code-styled cards. Honest limits: sensitivity to wording and example order, and that the context window "
        "is finite. Callback: Episode 32 (RAG puts retrieved documents in the prompt) and Episode 43 (reasoning)."),
    32: ("rag", "RAG", [4, 7, 8],
        "PORT, NOT FROM SCRATCH: a standalone film on this topic exists at ~/rag-video/ (rag.py, RAGExplainer.mp4, "
        "3:47, assets/). Read rag.py and its storyboards, then remake it as a series episode with kit.py: the "
        "Episode base class (the kit adds intro/outro, so drop its own), kit components, headers in the Level 4 "
        "teal accent, the series' recap. Keep its running example ('refund window for Enterprise plans'), its "
        "narration where it works, its best visuals; ~3.5–4 min; add callbacks (Episodes 9 cosine similarity, 22 "
        "embeddings, 31 prompting). Pronunciation: the series lexicon reads 'RAG' as the word 'rag' (tested in the "
        "original film), so write 'RAG' freely and run verify_audio with --terms AI,RAG. NEVER modify "
        "anything in ~/rag-video/; copy icons you need into the series assets as ep32-*.svg."),
    33: ("decoding-sampling", "DecodingSampling", [5, 7, 6],
        "Finishing 'Once upon a…' with a toy next-token distribution (state it is a toy): greedy (always the top "
        "token; can loop), temperature (callback: Episode 16; animate the bars), top-k (keep the k best, "
        "renormalise), top-p / nucleus (Holtzman et al., 2019; keep the smallest set whose probability reaches p), "
        "beam search (keep several partial sentences; common in translation, less so in chat). Compute every "
        "distribution in code, and sample with a seed to show different continuations."),
    34: ("vaes-gans", "VAEsGANs", [5, 8, 3],
        "Generating faces: draw simple synthetic cartoon faces in code (no photos), parameterised by a few numbers, "
        "so a 2-D latent space can be shown as a grid of faces. VAE (Kingma & Welling, 2013): encoder outputs μ and "
        "σ, sample z = μ + σ·ε (the reparameterisation trick makes it differentiable), decoder reconstructs; "
        "ELBO = reconstruction − KL to the prior. GAN (Goodfellow et al., 2014): generator vs discriminator, the "
        "minimax game; mode collapse; GANs give sharp images, VAEs blurrier ones; both largely overtaken for "
        "images by diffusion (Episode 35). 'ELBO' is said 'elbow' and QC accepts it; 'VAEs', 'GANs' pass."),
    35: ("diffusion", "Diffusion", [5, 2, 7],
        "A picture dissolving into static and back. Use a synthetic image (no photos). Forward process computed for "
        "real: x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε with a stated noise schedule. For the reverse, do NOT fake it by playing "
        "the forward frames backwards and calling it denoising: either label that clearly as an illustration, or "
        "(better) run a real reverse diffusion on 2-D points from a known mixture of Gaussians, whose score is "
        "analytic, and show the points flowing into the shape. Concepts: the network predicts the noise (DDPM, Ho "
        "et al., 2020), the score (gradient of log density), many small steps, classifier-free guidance for "
        "following a prompt. Latent diffusion (Stable Diffusion) runs in a compressed space."),
    36: ("multimodal", "Multimodal", [7, 6, 8],
        "Matching captions to pictures: use drawn icons/illustrations, not photos. CLIP (Radford et al., 2021): an "
        "image encoder and a text encoder map into one shared space; train on a batch of N pairs so the N matching "
        "pairs (the diagonal of an N×N similarity matrix) score high and the rest low — InfoNCE / contrastive "
        "loss with a temperature; compute a small example matrix and its loss. Zero-shot classification by "
        "comparing an image to captions like 'a photo of a dog'. Vision-language models: image patches become "
        "tokens fed into a language model. 'CLIP' passes QC."),
    37: ("efficient-attention", "EfficientAttention", [5, 7, 2],
        "Serving a long chat. Attention builds an n×n score matrix: compute and memory grow with n². KV cache: keep "
        "past keys and values so each new token costs O(n), but the cache grows linearly — compute its size for a "
        "stated illustrative configuration (2 × layers × kv_heads × head_dim × tokens × bytes) and assert it. "
        "FlashAttention (Dao et al., 2022): exact attention, IO-aware — tiles that fit in fast on-chip SRAM, an "
        "online softmax, the n×n matrix never written to slow HBM; show tiling. Multi-query (Shazeer, 2019) and "
        "grouped-query attention (Ainslie et al., 2023) share K/V heads to shrink the cache (compute the saving). "
        "'KV cache', 'GPU', 'FlashAttention' pass QC. Cite papers on screen in small text."),
    38: ("positional-long-context", "PositionalLongContext", [5, 2, 7],
        "A 100-page contract. Attention alone is order-blind (shuffle the words, same result). Sinusoidal "
        "encodings (Vaswani et al., 2017) and learned absolute positions; RoPE (Su et al., 2021): rotate each pair "
        "of query/key dimensions by an angle proportional to position, so q·k depends only on the relative offset "
        "— show the rotation on 2-D arrows, and compute that the dot product is unchanged under a shared shift. "
        "Going beyond the training length: position interpolation (Chen et al., 2023), NTK-aware scaling and YaRN "
        "(Peng et al., 2023). Also: long context ≠ good use of it ('lost in the middle', Liu et al., 2023). 'RoPE' passes QC."),
    39: ("mixture-of-experts", "MixtureOfExperts", [7, 8, 5],
        "Routing tokens to specialists: a router (a small linear layer + softmax) scores experts per token and "
        "picks the top-k (e.g. 2 of 8); only those experts run. Total vs active parameters — Mixtral 8x7B (Jiang et "
        "al., 2024): about 47B total, about 13B active per token. Switch Transformer (Fedus et al., 2021) used "
        "top-1. Load balancing: an auxiliary loss so tokens don't all crowd one expert (show an unbalanced vs "
        "balanced histogram). Trade-offs: memory holds all experts; communication across devices. Experts rarely "
        "map to human topics neatly — say so. 'MoE' passes QC (or say 'mixture of experts')."),
    40: ("state-space-mamba", "StateSpaceMamba", [5, 7, 2],
        "Streaming a very long sequence. Linear recurrence h_t = A·h_{t−1} + B·x_t, y_t = C·h_t: constant memory per "
        "step, linear time in length (vs attention's n²; callback: Episode 37). Training in parallel with a scan "
        "(or as a convolution for fixed A, B, C — S4, Gu et al., 2021). Mamba (Gu & Dao, 2023): selective — B, C and "
        "the step size Δ depend on the input, so the model chooses what to remember; a hardware-aware scan. "
        "Compute a small recurrence in numpy and animate the state. Honest trade-off: fixed-size memory struggles "
        "with exact recall/copying; hybrids interleave attention layers. 'Mamba', 'SSM' pass QC."),
    41: ("lora-peft", "LoRAPEFT", [7, 2, 5],
        "Personalising a model on a laptop. LoRA (Hu et al., 2021): freeze W, learn ΔW = B·A with rank r ≪ d; B is "
        "d×r, A is r×k, B starts at zero so training starts from the original model; merge after training (no "
        "inference cost). Compute the parameter count for d = k = 4096, r = 8: 65,536 vs 16,777,216 (0.39%) and "
        "assert it. Adapters (Houlsby et al., 2019) as the earlier idea. QLoRA (Dettmers et al., 2023): a 4-bit "
        "(NF4) frozen base + LoRA, fine-tuning a 65B model on one 48 GB GPU. Show low rank visually (a thin "
        "matrix times a wide one). 'LoRA' passes; 'QLoRA' may take extra takes — say 'Q-LoRA' if it struggles."),
    42: ("quantization-distillation", "QuantizationDistillation", [7, 5, 6],
        "A model that fits on a phone. Memory: parameters × bytes — 7B at 16-bit ≈ 14 GB, at 4-bit ≈ 3.5 GB (plus "
        "overhead); compute in code. Quantization: scale, round, clamp; compute real rounding errors with numpy on a "
        "seeded weight vector for int8 and int4, absmax vs per-group scales. Outliers: a few huge activations ruin "
        "a shared scale (LLM.int8(), Dettmers et al., 2022, keeps them in 16-bit); GPTQ and AWQ as post-training "
        "methods. Distillation (Hinton et al., 2015): a small student learns the teacher's soft probabilities at a "
        "temperature (callback: Episode 16). Say 'int eight' / 'eight-bit' in narration; 'int8' on screen."),
    43: ("reasoning-test-time", "ReasoningTestTime", [6, 7, 5],
        "A hard math word problem. Chain-of-thought (callback: Episode 31). Self-consistency (Wang et al., 2022): "
        "sample several solutions, majority-vote the answers — compute exactly with the binomial distribution how "
        "accuracy rises with more samples when each is right with, say, 60% (assert numbers), and note it fails if "
        "errors agree. Best-of-N with a verifier; search over steps. Reasoning models are trained with "
        "reinforcement learning on verifiable rewards (math answers that can be checked, code that must pass tests) "
        "to think longer before answering — e.g. OpenAI o1 (2024), DeepSeek-R1 (2025). More thinking tokens help, "
        "with diminishing returns and higher cost. Keep claims conservative; this field moves fast — date facts "
        "(as of 2026) in the section notes."),
    44: ("agents-tool-use", "AgentsToolUse", [8, 4, 7],
        "Booking a trip. Function calling: the model outputs a structured call (JSON with a tool name and "
        "arguments), the runtime executes it, the result goes back into the context; show this as cards. The "
        "reason–act–observe loop (ReAct, Yao et al., 2022); planning (break the goal into steps); memory (the "
        "context window, plus notes or a database). Tools here: search flights, check calendar, book. Risks: errors "
        "compound over many steps, tool outputs can contain malicious instructions (prompt injection), so agents "
        "need permissions, sandboxing and a human confirmation before spending money — show the agent asking "
        "before it pays."),
    45: ("interpretability", "Interpretability", [3, 9, 7],
        "Looking inside: neurons are often polysemantic; features as directions in activation space (callback: "
        "Episode 9). Superposition (Elhage et al., 2022, toy models): more features than dimensions, packed nearly "
        "orthogonally — show a 2-D toy with 5 feature directions computed in code. Sparse autoencoders / "
        "dictionary learning (Anthropic: 'Towards Monosemanticity', 2023; 'Scaling Monosemanticity', 2024, on Claude "
        "3 Sonnet) find interpretable features; one responded to the Golden Gate Bridge, and amplifying it made the "
        "model talk about the bridge constantly ('Golden Gate Claude'). Circuits: e.g. induction heads (Olsson et "
        "al., 2022). Keep claims faithful to the papers; draw no real logos or photos."),
    46: ("alignment-evaluation", "AlignmentEvaluation", [6, 4, 8],
        "A model gaming its reward: the classic CoastRunners example (OpenAI, 2016) — a boat-racing agent learned to "
        "circle hitting targets for points instead of finishing the race; draw it simply. Specification gaming / "
        "reward hacking (callback: Episode 30). Evaluation: benchmarks (multiple-choice suites like MMLU), "
        "saturation, contamination (test questions leaking into training data), held-out and private tests, human "
        "evaluation, red-teaming. Scalable oversight: using AI to help people judge AI (debate, critiques), "
        "constitutional-style principles. THIS IS THE FINAL EPISODE: override `series_outro()` in your scene (in "
        "your own file) to close the course instead of the kit's bare 'Thanks for watching!': a short card and "
        "line like 'That's the end of AI, Explained: from a spam filter to the frontier of research. Thanks for "
        "watching!', optionally the 5 level badges in their accents. Keep it under 15 seconds."),
}

COMMON = """
SERIES NOTES (apply to every episode):
- Episodes 1–8 are finished; later episodes are being made in parallel by other agents, so you can refer to any episode by number ("as we saw in episode two") using COURSE.md's titles, but never rely on another episode's files.
- Advanced levels may be denser, but every formula on screen is explained term by term and shown moving; a section never shows more than one new formula at a time. 3–4.5 minutes.
- Pictures: no real photographs, logos or copyrighted images; draw everything in code or as flat SVG icons. Data you invent for teaching is fine but say "for example" in narration and record it as illustrative in the section notes.
- Papers: where a result has a well-known source, show "Author et al., Year" in small MUTED text near it (font_size ≥ 20). Prefer established, well-cited facts; if unsure of a number, leave it out or hedge it. You may use WebSearch/WebFetch to check a fact, and put sources in the section notes.
- QC facts measured before you started: these terms pass the voice check as written — AI, ML, SGD, Adam, SVD, PCA, LSTM(s), RNN, CNN, RLHF, DPO, PPO, SFT, VAE(s), GAN(s), ELBO, LoRA, KV cache, GPU, CLIP, BPE, MoE, SSM, Mamba, FlashAttention, GELU, ReLU, RoPE. UK and US spellings, "a hundred"/"100", decimals ("8.5" ≡ "eight point five" ≡ "eight and a half"), "weights"/"waits", and digits vs number words are all treated as equal by the checker. A whole-film check often drops the final "Next time… Thanks for watching!" line; transcribe the last 15 s alone to confirm it.
- ManimGX: TransformFromCopy, FadeTransform and FadeTransformPieces show their target BEFORE their cue; kit.py patches them to FadeIn(target, target_position=source), so it is safe through the kit, but never Transform/ReplacementTransform FROM an object that is not yet on screen (the same early-show bug). Probe a doubtful animation with inspect -t before and after its cue.
- ManimGX: never emphasise an object (Indicate, Wiggle, ScaleInPlace…) on a LATER cue of the same say() line while its intro (FadeIn/Write/Create/LaggedStart) may still be running: it freezes half-drawn (dim or partly written). Put both in one AnimationGroup, or use Circumscribe(x), which leaves x alone. Check storyboards for faint or partly drawn objects.
- Fonts: the kit's code font (Fira Code) draws ligatures — "<|" and "|>" become triangles, and "->", "=>", "!=", "==" merge into symbols. For special tokens, chat templates, JSON or code where the exact characters matter, set that text in "Menlo" (installed) instead.
- ManimGX: animating a PART of a group introduced in the same say() line (or transforming a GROUP whose children are introduced in it) used to show that object from the line's start; Episode.say in kit.py now prevents this automatically. Still probe a doubtful line with inspect -t just after it starts.
- Run verify_audio from INSIDE your episode folder (it needs voice/ beside the MP4), with --terms AI plus every ALL-CAPS acronym you say.
"""


def course_rows():
    rows = {}
    for line in (SERIES_DIR / "COURSE.md").read_text().splitlines():
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if len(cells) == 7 and cells[0].isdigit():
            n = int(cells[0])
            rows[n] = dict(title=cells[1].strip("*"), goal=cells[2], concepts=cells[3], math=cells[4], example=cells[5])
    return rows


def level_of(n):
    return 2 if n <= 18 else 3 if n <= 26 else 4 if n <= 36 else 5


def brief(n, rows, template):
    slug, cls, refs, notes = SPEC[n]
    row, level = rows[n], level_of(n)
    level_name, audience = LEVELS[level]
    math = row["math"].replace("`", "")
    references = "; ".join(f"{SERIES_DIR}/{FINISHED[r].split(' ')[0]}/scene.py — {FINISHED[r].split(' ', 1)[1]}" for r in refs if r in FINISHED)
    closest = FINISHED[refs[0]].split(" ")[0]
    fill = {
        "{SERIES}": "AI, Explained",
        "{N}": str(n),
        "{NN}": f"{n:02d}",
        "{TITLE}": row["title"],
        "{L}": str(level),
        "{LEVEL_NAME}": level_name,
        "{AUDIENCE}": audience,
        "{SERIES_DIR}": str(SERIES_DIR),
        "{slug}": slug,
        "{ClassName}": cls,
        "{GOAL}": row["goal"],
        "{CONCEPTS}": row["concepts"],
        "{EXAMPLE}": row["example"],
        "{MATH}": math if math not in ("—", "none") else "no formulas required",
        "{ACCURACY_NOTES}": "EPISODE NOTES: " + notes,
        "{REFERENCE_EPISODES}": references,
        "{CLOSEST}": closest,
        "{NARRATOR_PYTHON}": NARRATOR_PYTHON,
    }
    text = template
    for key, value in fill.items():
        text = text.replace(key, value)
    leftover = re.findall(r"\{(?:[A-Z_]+|slug|ClassName)\}", text)
    assert not leftover, (n, leftover)
    return text.rstrip() + "\n" + COMMON


def main():
    rows = course_rows()
    template = TEMPLATE.read_text()
    numbers = [int(a) for a in sys.argv[1:]] or sorted(SPEC)
    for n in numbers:
        path = HERE / f"ep{n:02d}.md"
        path.write_text(brief(n, rows, template))
        print(path.name, len(path.read_text().split()), "words")


if __name__ == "__main__":
    main()
