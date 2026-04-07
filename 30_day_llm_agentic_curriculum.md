# 30-Day Intensive Curriculum: Deep Learning → Transformers → LLMs → RAG → Agentic AI

**Student profile fit:** Basic Python + basic ML, 1.5 hours/day, math-heavy + implementation-heavy.  
**Stack:** PyTorch, Hugging Face, LangChain.  
**Constraint satisfied:** No CNNs, no CV.

---

## Day 1 — Linear Algebra for Neural Text Models
### 1) Layman Explanation
A neural model is a smart calculator that turns text numbers into better text numbers. Think of each layer as a filtering station that reweighs signals. This matters because every LLM operation is matrix math.

### 2) Mathematical Foundation
Formula: \(y = XW + b\)
- \(X\in\mathbb{R}^{n\times d}\): batch of token features
- \(W\in\mathbb{R}^{d\times h}\): weights
- \(b\in\mathbb{R}^{h}\): bias
- \(y\in\mathbb{R}^{n\times h}\)
Derivation: weighted sums per output dimension. Gradient (MSE):
\(L=\frac{1}{n}\|y-\hat y\|^2\),  \(\partial L/\partial W = \frac{2}{n}X^T(\hat y-y)\).
Probability view: linear logits later become probabilities via softmax.
Numerical: if \(X=[1,2],W=[[2],[3]],b=[1]\Rightarrow y=1*2+2*3+1=9\).

### 3) Intuition Behind Math
Matrix multiply is parallel feature scoring. Each column of \(W\) is a detector.

### 4) Architecture Flow
Token IDs → embedding lookup → linear transform \(XW+b\) → scores.

### 5) Practical Example (Real Dataset)
Dataset: **AG News** (Hugging Face `ag_news`). 4-class news topic classification.

### 6) Full Working Code
```python
import torch, torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("ag_news")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def encode(batch):
    return tok(batch["text"], truncation=True, padding="max_length", max_length=64)

ds = ds.map(encode, batched=True)

def collate(ex):
    x = torch.tensor([e["input_ids"] for e in ex])
    y = torch.tensor([e["label"] for e in ex])
    return x, y

class MeanPoolClassifier(nn.Module):
    def __init__(self, vocab=30522, d=128, c=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.fc = nn.Linear(d, c)
    def forward(self, x):
        h = self.emb(x).mean(1)
        return self.fc(h)

m = MeanPoolClassifier(); opt = torch.optim.Adam(m.parameters(), 1e-3); ce = nn.CrossEntropyLoss()
loader = torch.utils.data.DataLoader(ds["train"].select(range(2000)), batch_size=32, shuffle=True, collate_fn=collate)
for _ in range(2):
    for x,y in loader:
        opt.zero_grad(); loss = ce(m(x), y); loss.backward(); opt.step()
print("train-loss", float(loss))
```

### 7) Code Walkthrough
Embedding gives matrix \(X\); mean pool approximates sentence vector; linear layer implements \(XW+b\); CE trains class probabilities.

### 8) Exercises
Conceptual: Why does matrix multiplication scale well on GPUs? Why embeddings before linear layers?  
Math: derive \(\partial L/\partial b\); compute output shape for batch=32, d=128, c=4.  
Coding: add dropout; compare mean vs max pooling.

### 9) Summary Notes
- Core: \(y=XW+b\)
- Gradient: \(X^T\delta\)
- LLMs are mostly repeated linear algebra.

---

## Day 2 — Activation Functions and Nonlinearity
### 1) Layman Explanation
Without activation, many layers collapse into one linear layer. Activation is like adding bends to a straight pipe.

### 2) Mathematical Foundation
ReLU: \(f(z)=\max(0,z)\), derivative \(f'(z)=1(z>0)\).  
GELU (used in Transformers): \(\text{GELU}(x)=x\Phi(x)\approx0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))\).
Probability view: GELU gates values proportionally to how likely they are under Gaussian CDF.
Matrix: \(H=f(XW_1+b_1)\), \(Y=HW_2+b_2\).
Numerical: ReLU([-2,3])=[0,3].

### 3) Intuition Behind Math
Activation creates curved decision boundaries.

### 4) Architecture Flow
Embeddings → linear → GELU → linear → logits.

### 5) Practical Example
Dataset: **IMDb** sentiment (`imdb`). Binary classification.

### 6) Full Working Code
```python
import torch, torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

ds = load_dataset("imdb")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def enc(b): return tok(b["text"], truncation=True, padding="max_length", max_length=128)
ds = ds.map(enc, batched=True)

class MLPText(nn.Module):
    def __init__(self, vocab=30522, d=128):
        super().__init__(); self.e=nn.Embedding(vocab,d); self.f1=nn.Linear(d,256); self.f2=nn.Linear(256,2); self.act=nn.GELU()
    def forward(self,x):
        h=self.e(x).mean(1)
        return self.f2(self.act(self.f1(h)))
```
Training/eval loop: same as Day 1 with CE + accuracy.

### 7) Code Walkthrough
`act(self.f1(h))` is nonlinearity that prevents linear collapse.

### 8) Exercises
Conceptual: why dead ReLU? why GELU for transformers?  
Math: derivative of tanh approximation; chain rule through two linear layers.  
Coding: swap GELU/ReLU and compare validation accuracy.

### 9) Summary Notes
- Nonlinearity = expressiveness.
- GELU is smoother than ReLU.

---

## Day 3 — Loss Functions and Probability (Softmax + Cross-Entropy)
### 1) Layman Explanation
Model outputs raw scores (logits). Softmax turns them into probabilities that sum to 1.

### 2) Mathematical Foundation
Softmax: \(p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}\).  
Cross entropy: \(L=-\sum_i y_i\log p_i\). For class \(k\): \(L=-\log p_k\).  
Gradient: \(\partial L/\partial z_i=p_i-y_i\).
Probability interpretation: maximizes likelihood of correct class.
Matrix batch form: \(P=\text{softmax}(Z)\), \(L=-\frac1n\sum \log P_{n,k}\).
Numerical: logits [2,1] => p≈[0.731,0.269], if true class 0 => loss 0.313.

### 3) Intuition Behind Math
CE heavily penalizes confident wrong predictions.

### 4) Architecture Flow
Features → logits \(z\) → softmax probabilities → CE loss.

### 5) Practical Example
Dataset: **DBPedia 14** topic classification.

### 6) Full Working Code
```python
# Reuse Day1 model; focus on explicit softmax for inspection
logits = m(x)
probs = torch.softmax(logits, dim=-1)
loss = torch.nn.functional.cross_entropy(logits, y)
```

### 7) Code Walkthrough
`cross_entropy` combines log-softmax + NLL stably.

### 8) Exercises
Conceptual: logits vs probabilities?  
Math: prove softmax sums to 1; derive CE gradient.  
Coding: implement softmax manually with max-shift stabilization.

### 9) Summary Notes
- Key identity: gradient = \(p-y\).

---

## Day 4 — Backpropagation and Chain Rule
### 1) Layman Explanation
Backprop tells each weight how responsible it was for error.

### 2) Mathematical Foundation
For 2-layer MLP: \(a_1=XW_1+b_1\), \(h=f(a_1)\), \(z=hW_2+b_2\).  
\(\delta_2=\partial L/\partial z\), \(\delta_1=(\delta_2W_2^T)\odot f'(a_1)\).  
\(\partial L/\partial W_2=h^T\delta_2\), \(\partial L/\partial W_1=X^T\delta_1\).

### 3) Intuition Behind Math
Error is redistributed backward through each operation.

### 4) Architecture Flow
Forward pass caches activations → backward pass computes gradients → optimizer updates.

### 5) Practical Example
Dataset: **SST-2** (`glue`, `sst2`).

### 6) Full Working Code
```python
loss.backward()
for n,p in m.named_parameters():
    if p.grad is not None:
        print(n, p.grad.norm().item())
```

### 7) Code Walkthrough
Inspecting gradient norms validates chain rule flow.

### 8) Exercises
Conceptual: vanishing gradients? exploding gradients?  
Math: derive \(\delta_1\); compute one manual update step.  
Coding: gradient clipping at 1.0.

### 9) Summary Notes
- Backprop = repeated chain rule.

---

## Day 5 — Optimization: SGD, Momentum, Adam
### 1) Layman Explanation
Optimizer is the steering wheel minimizing error.

### 2) Mathematical Foundation
SGD: \(\theta_{t+1}=\theta_t-\eta g_t\).  
Momentum: \(v_t=\beta v_{t-1}+(1-\beta)g_t\), \(\theta\leftarrow\theta-\eta v_t\).  
Adam: \(m_t, v_t\) moments + bias correction.

### 3) Intuition Behind Math
Momentum smooths noisy gradients; Adam adapts per-parameter step sizes.

### 4) Architecture Flow
Compute grad → optimizer state update → parameter update.

### 5) Practical Example
Dataset: **Yelp Polarity** (`yelp_polarity`).

### 6) Full Working Code
```python
opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
```

### 7) Code Walkthrough
AdamW decouples weight decay from adaptive gradient scaling.

### 8) Exercises
Conceptual: why learning-rate warmup?  
Math: Adam bias-correction derivation.  
Coding: compare SGD vs AdamW curves.

### 9) Summary Notes
- Default LLM finetuning: AdamW + warmup.

---

## Day 6 — Embeddings and Vector Spaces
### 1) Layman Explanation
Words become coordinates in space where meaning is geometric distance.

### 2) Mathematical Foundation
Embedding lookup: \(E\in\mathbb{R}^{|V|\times d}\), token index \(i\Rightarrow e_i=E[i]\).  
Cosine: \(\cos(u,v)=\frac{u\cdot v}{\|u\|\|v\|}\).
Gradient from CE updates only rows used in batch.

### 3) Intuition Behind Math
Similar contexts push vectors together.

### 4) Architecture Flow
Token ID → embedding vector → contextual layers.

### 5) Practical Example
Dataset: **WikiText-2** language modeling subset.

### 6) Full Working Code
```python
emb = nn.Embedding(30000, 256)
ids = torch.tensor([10, 200, 10])
vecs = emb(ids)  # repeated token gets same base vector
```

### 7) Code Walkthrough
Embedding is parameter table, not arithmetic on IDs.

### 8) Exercises
Conceptual: static vs contextual embeddings.  
Math: prove cosine scale invariance.  
Coding: nearest-neighbor search in embedding table.

### 9) Summary Notes
- Embeddings are learned geometry.

---

## Day 7 — Week 1 Integration: From Math to Small Text Classifier
### 1) Layman Explanation
You now combine linear algebra, loss, backprop, optimizer, embeddings.

### 2) Mathematical Foundation
End-to-end objective: \(\min_\theta \mathbb{E}_{(x,y)}[-\log p_\theta(y|x)]\).

### 3) Intuition Behind Math
Every component contributes to probability calibration.

### 4) Architecture Flow
IDs → Embedding → MLP → Softmax → CE → Backprop.

### 5) Practical Example
Dataset: **AG News** full training subset 20k.

### 6) Full Working Code
Use Days 1–6 combined; include train/val split and accuracy metric.

### 7) Code Walkthrough
Track loss, accuracy, gradient norms.

### 8) Exercises
Conceptual: overfitting indicators?  
Math: derive full gradient graph.  
Coding: add early stopping.

### 9) Summary Notes
- You can now train stable text MLP baselines.

### Weekly Assignment (Week 1)
Build 2-layer text classifier with residual connection on AG News. Derive gradients for residual branch and report effect on convergence.

### Weekly Project (Week 1)
**Problem:** Multi-class news categorization.  
**Dataset:** AG News.  
**Math:** CE + AdamW + L2 regularization.  
**Architecture:** Embedding → FFN → FFN + residual → classifier.  
**Steps:** preprocess, train, validate, ablation (ReLU vs GELU).  
**Metrics:** accuracy, macro-F1, confusion matrix.

---

## Day 8 — Attention Intuition and Dot-Product Scoring
### 1) Layman Explanation
Attention lets each token look at other relevant tokens.

### 2) Mathematical Foundation
Score: \(s_{ij}=q_i\cdot k_j\). Weights: \(\alpha_{ij}=\text{softmax}_j(s_{ij}/\sqrt{d_k})\). Output: \(o_i=\sum_j\alpha_{ij}v_j\).
Gradient through softmax and weighted sum.
Matrix: \(\text{Attn}(Q,K,V)=\text{softmax}(QK^T/\sqrt{d_k})V\).

### 3) Intuition Behind Math
Dot product measures compatibility between query and context key.

### 4) Architecture Flow
Input embeddings → Q,K,V projections → attention mixing → contextual output.

### 5) Practical Example
Dataset: **WikiText-2** next-token prediction.

### 6) Full Working Code
```python
def attention(Q,K,V):
    w = (Q @ K.transpose(-1,-2)) / (Q.size(-1)**0.5)
    a = torch.softmax(w, dim=-1)
    return a @ V
```

### 7) Code Walkthrough
`Q@K^T` builds pairwise token relevance matrix.

### 8) Exercises
Conceptual: why scaling by \(\sqrt{d_k}\)?  
Math: derive \(\partial o_i/\partial q_i\).  
Coding: add causal mask.

### 9) Summary Notes
- Attention = differentiable retrieval inside sequence.

---

## Day 9 — Self-Attention with Causal Masking
### 1) Layman Explanation
In generation, token t must not peek at future tokens.

### 2) Mathematical Foundation
Masked logits: \(M_{ij}=-\infty\) if \(j>i\); \(A=\text{softmax}((QK^T+M)/\sqrt{d})\).

### 3) Intuition Behind Math
Mask imposes autoregressive information flow.

### 4) Architecture Flow
Token embeddings + positions → masked self-attn → logits.

### 5) Practical Example
Dataset: **TinyStories** subset (HF).

### 6) Full Working Code
```python
T = x.size(1)
mask = torch.triu(torch.ones(T,T), diagonal=1).bool()
attn_scores.masked_fill_(mask, float('-inf'))
```

### 7) Code Walkthrough
Upper triangle is forbidden future context.

### 8) Exercises
Conceptual: bidirectional vs causal models.  
Math: prove triangular mask causality.  
Coding: validate no future leakage with unit test.

### 9) Summary Notes
- Causal masking defines decoder-only LLMs.

---

## Day 10 — Multi-Head Attention
### 1) Layman Explanation
Multiple heads = multiple perspectives (syntax, agreement, entities).

### 2) Mathematical Foundation
\(\text{MHA}(X)=\text{Concat}(head_1,...,head_h)W^O\),  
\(head_i=\text{Attn}(XW_i^Q, XW_i^K, XW_i^V)\).

### 3) Intuition Behind Math
Different subspaces learn different relations.

### 4) Architecture Flow
Shared input → split into heads → parallel attention → merge.

### 5) Practical Example
Dataset: WikiText-2.

### 6) Full Working Code
```python
mha = torch.nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
out,_ = mha(x,x,x, attn_mask=mask)
```

### 7) Code Walkthrough
PyTorch handles projection matrices and head reshaping.

### 8) Exercises
Conceptual: more heads always better?  
Math: parameter count formula.  
Coding: compare 4 vs 8 heads at fixed embedding.

### 9) Summary Notes
- Head diversity boosts expressiveness.

---

## Day 11 — Positional Encoding
### 1) Layman Explanation
Attention alone ignores word order; positions inject sequence order.

### 2) Mathematical Foundation
Sinusoidal: \(PE_{pos,2i}=\sin(pos/10000^{2i/d})\), \(PE_{pos,2i+1}=\cos(pos/10000^{2i/d})\).

### 3) Intuition Behind Math
Different frequencies encode relative distance patterns.

### 4) Architecture Flow
Token embedding + positional embedding → transformer block.

### 5) Practical Example
Dataset: WikiText-2.

### 6) Full Working Code
```python
pos = torch.arange(T).unsqueeze(1)
i = torch.arange(d//2).unsqueeze(0)
angles = pos / (10000 ** (2*i/d))
pe = torch.zeros(T,d); pe[:,0::2]=torch.sin(angles); pe[:,1::2]=torch.cos(angles)
```

### 7) Code Walkthrough
Add PE to token embeddings before attention.

### 8) Exercises
Conceptual: learned vs sinusoidal position.  
Math: show periodicity.  
Coding: replace with learned `nn.Embedding(max_len,d)`.

### 9) Summary Notes
- Order is mandatory for language semantics.

---

## Day 12 — Transformer Feed-Forward + Residual + LayerNorm
### 1) Layman Explanation
After attention mixes tokens, FFN refines each token independently.

### 2) Mathematical Foundation
\(\text{FFN}(x)=W_2\sigma(W_1x+b_1)+b_2\).  
Residual+Norm: \(y=\text{LN}(x+\text{Sublayer}(x))\).

### 3) Intuition Behind Math
Residual preserves signal; LayerNorm stabilizes scale.

### 4) Architecture Flow
x → attention → add+norm → FFN → add+norm.

### 5) Practical Example
Dataset: TinyStories subset.

### 6) Full Working Code
```python
class Block(nn.Module):
    def __init__(self,d,nhead):
        super().__init__(); self.mha=nn.MultiheadAttention(d,nhead,batch_first=True)
        self.ln1=nn.LayerNorm(d); self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d)); self.ln2=nn.LayerNorm(d)
    def forward(self,x,mask=None):
        a,_=self.mha(x,x,x,attn_mask=mask); x=self.ln1(x+a); x=self.ln2(x+self.ff(x)); return x
```

### 7) Code Walkthrough
Two residual pathways + two normalizations per block.

### 8) Exercises
Conceptual: pre-norm vs post-norm.  
Math: Jacobian effect of residual.  
Coding: implement dropout in attention + FFN.

### 9) Summary Notes
- Transformer block = Attn + FFN + residual norms.

---

## Day 13 — Full Decoder-Only Transformer from Scratch
### 1) Layman Explanation
Stack blocks to predict next token repeatedly.

### 2) Mathematical Foundation
Objective: \(\sum_t -\log p(x_t|x_{<t})\). matrix logits \(Z=HW_{vocab}^T\).

### 3) Intuition Behind Math
Model learns conditional distributions over vocabulary at each position.

### 4) Architecture Flow
IDs + pos → N transformer blocks → LM head → next-token probs.

### 5) Practical Example
Dataset: TinyStories/WikiText-2.

### 6) Full Working Code
```python
# modules from Day 12 + token/position embeddings + lm_head
# train loop with shift:
logits = model(x[:, :-1])
loss = nn.functional.cross_entropy(logits.reshape(-1,vocab), x[:,1:].reshape(-1))
```

### 7) Code Walkthrough
Shifted inputs/targets implement autoregressive teacher forcing.

### 8) Exercises
Conceptual: why tie embeddings?  
Math: perplexity relation \(\exp(CE)\).  
Coding: gradient accumulation for memory limits.

### 9) Summary Notes
- You now own full transformer mechanics.

---

## Day 14 — Week 2 Integration + HF Trainer
### 1) Layman Explanation
Move from hand-built loops to production-style training APIs.

### 2) Mathematical Foundation
Same CE objective, better engineering abstractions.

### 3) Intuition Behind Math
Frameworks automate plumbing, not learning theory.

### 4) Architecture Flow
Tokenizer → data collator → model → Trainer → metrics.

### 5) Practical Example
Dataset: **WikiText-2**.

### 6) Full Working Code
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
# load gpt2, tokenize dataset, DataCollatorForLanguageModeling(mlm=False)
# train with Trainer
```

### 7) Code Walkthrough
Trainer wraps optimization, logging, evaluation.

### 8) Exercises
Conceptual: from-scratch vs pretrained tradeoffs.  
Math: effective batch size with grad accumulation.  
Coding: add perplexity metric callback.

### 9) Summary Notes
- You can train/finetune transformer LMs efficiently.

### Weekly Assignment (Week 2)
Implement a decoder-only transformer with configurable heads/layers; derive complexity \(O(T^2d)\) and run scaling experiment on WikiText-2.

### Weekly Project (Week 2)
**Problem:** Small language model pretraining.  
**Dataset:** WikiText-2.  
**Math:** autoregressive likelihood, CE/perplexity.  
**Architecture:** token+pos embeddings, 4 transformer blocks, tied LM head.  
**Steps:** tokenize, chunk, train, sample text.  
**Metrics:** validation perplexity, tokens/sec.

---

## Day 15 — Tokenization and Subword Probability
### 1) Layman Explanation
Tokenization decides the atomic units your model predicts.

### 2) Mathematical Foundation
Sequence probability: \(P(x)=\prod_t P(x_t|x_{<t})\). Tokenization changes factorization and length.

### 3) Intuition Behind Math
Better subwords reduce OOV and improve compression.

### 4) Architecture Flow
Raw text → tokenizer IDs → LM.

### 5) Practical Example
Dataset: **OpenWebText-like subset** from HF `stas/openwebtext-10k`.

### 6) Full Working Code
```python
tok = AutoTokenizer.from_pretrained("gpt2")
ids = tok("unbelievable")
print(ids.tokens())
```

### 7) Code Walkthrough
Inspect segmentation and token counts.

### 8) Exercises
Conceptual: BPE vs WordPiece.  
Math: show length-probability tradeoff.  
Coding: compare perplexity with two tokenizers.

### 9) Summary Notes
- Tokenization is part of model design.

---

## Day 16 — Finetuning an LLM for Classification (Instruction Style)
### 1) Layman Explanation
Reuse pretrained language knowledge for downstream tasks.

### 2) Mathematical Foundation
Minimize CE on label tokens or classification head outputs.

### 3) Intuition Behind Math
Pretraining gives prior; finetuning specializes likelihood.

### 4) Architecture Flow
Prompt template → model → label logits.

### 5) Practical Example
Dataset: **Financial PhraseBank** (`takala/financial_phrasebank`).

### 6) Full Working Code
```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
# standard Trainer finetune
```

### 7) Code Walkthrough
Only head or full model can be updated.

### 8) Exercises
Conceptual: catastrophic forgetting.  
Math: class-weighted CE derivation.  
Coding: freeze base model and compare.

### 9) Summary Notes
- Finetuning = targeted distribution shift adaptation.

---

## Day 17 — Embedding Similarity and Semantic Search
### 1) Layman Explanation
Represent passages/questions as vectors and match by proximity.

### 2) Mathematical Foundation
Cosine similarity and inner product ranking. ANN approximates top-k neighbors.

### 3) Intuition Behind Math
Semantic meaning corresponds to direction in embedding space.

### 4) Architecture Flow
Text → encoder embeddings → vector index → retrieval.

### 5) Practical Example
Dataset: **MS MARCO Passage** (subset via HF).

### 6) Full Working Code
```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
D = enc.encode(passages, normalize_embeddings=True)
index = faiss.IndexFlatIP(D.shape[1]); index.add(D.astype('float32'))
q = enc.encode([query], normalize_embeddings=True).astype('float32')
scores, ids = index.search(q, 5)
```

### 7) Code Walkthrough
Normalization turns dot-product into cosine ranking.

### 8) Exercises
Conceptual: bi-encoder vs cross-encoder.  
Math: prove normalized dot = cosine.  
Coding: evaluate Recall@5.

### 9) Summary Notes
- Retrieval quality starts with embedding geometry.

---

## Day 18 — RAG Fundamentals
### 1) Layman Explanation
RAG lets model “look up” facts before answering.

### 2) Mathematical Foundation
Approx: \(P(a|q)\approx\sum_{d\in topk} P(a|q,d)P(d|q)\).

### 3) Intuition Behind Math
Marginalize answer probability over retrieved evidence.

### 4) Architecture Flow
Question → retrieve documents → augment prompt → generate answer.

### 5) Practical Example
Dataset: **Natural Questions** passages subset.

### 6) Full Working Code
```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
```
(Construct retriever + HF pipeline LLM + RetrievalQA chain.)

### 7) Code Walkthrough
Retriever provides context; generator conditions on it.

### 8) Exercises
Conceptual: hallucination reduction mechanism.  
Math: derive top-k truncation approximation error intuition.  
Coding: compare k=3 vs k=10.

### 9) Summary Notes
- RAG shifts knowledge from weights to external memory.

---

## Day 19 — Chunking, Indexing, and Retrieval Evaluation
### 1) Layman Explanation
How you split documents strongly affects what gets found.

### 2) Mathematical Foundation
Chunk overlap tradeoff: recall vs redundancy. Metrics: Recall@k, MRR, nDCG.

### 3) Intuition Behind Math
Too-small chunks lose context; too-large chunks dilute relevance.

### 4) Architecture Flow
Documents → chunk strategy → embeddings → ANN index → ranked chunks.

### 5) Practical Example
Dataset: **SQuAD v2 contexts**.

### 6) Full Working Code
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(long_doc)
```

### 7) Code Walkthrough
Tune chunk size and overlap; measure retrieval metrics.

### 8) Exercises
Conceptual: precision vs recall in retrieval.  
Math: compute MRR for sample ranked list.  
Coding: implement evaluation script.

### 9) Summary Notes
- Retrieval pipeline needs explicit offline metrics.

---

## Day 20 — Reranking and Hybrid Search
### 1) Layman Explanation
First retrieve fast candidates, then rerank deeply.

### 2) Mathematical Foundation
Hybrid score: \(s=\lambda s_{dense}+(1-\lambda)s_{bm25}\).

### 3) Intuition Behind Math
Keyword and semantic signals complement each other.

### 4) Architecture Flow
Dense + BM25 retrieval → union candidates → cross-encoder rerank.

### 5) Practical Example
Dataset: **BEIR (FiQA subset)**.

### 6) Full Working Code
```python
# dense top-50 + bm25 top-50 -> merge -> cross-encoder rerank top-10
```

### 7) Code Walkthrough
Cross-encoder directly models query-document interaction.

### 8) Exercises
Conceptual: latency cost of reranker.  
Math: optimize \(\lambda\) by grid search.  
Coding: add reciprocal-rank fusion.

### 9) Summary Notes
- Production RAG usually uses multi-stage retrieval.

---

## Day 21 — Week 3 Integration: End-to-End RAG App
### 1) Layman Explanation
Build a question-answering system grounded in retrieved docs.

### 2) Mathematical Foundation
Pipeline objective combines retrieval relevance + generation likelihood.

### 3) Intuition Behind Math
Good evidence narrows output entropy.

### 4) Architecture Flow
Query → retriever → reranker → generator → cited answer.

### 5) Practical Example
Dataset: **HotpotQA** supporting docs subset.

### 6) Full Working Code
LangChain pipeline with FAISS retriever + HF causal LM, output includes source docs.

### 7) Code Walkthrough
Instrument retrieval scores and answer faithfulness checks.

### 8) Exercises
Conceptual: faithfulness vs fluency.  
Math: compute answer confidence proxy from token logprobs.  
Coding: add citation enforcement in prompt.

### 9) Summary Notes
- RAG is probabilistic evidence conditioning.

### Weekly Assignment (Week 3)
Implement hybrid RAG with reranking on BEIR subset. Derive and tune weighted retrieval objective; report Recall@k and exact match on QA subset.

### Weekly Project (Week 3)
**Problem:** Domain QA assistant.  
**Dataset:** SQuAD v2 contexts + question set.  
**Math:** cosine retrieval, hybrid fusion, CE-based answer generation.  
**Architecture:** embedder + FAISS + BM25 + reranker + generator.  
**Steps:** index, evaluate retrieval, integrate generator, measure QA quality.  
**Metrics:** Recall@k, MRR, EM/F1, hallucination rate.

---

## Day 22 — Agentic AI Basics: Perception, Planning, Action
### 1) Layman Explanation
Agent = model that can decide actions (tool calls), not just chat.

### 2) Mathematical Foundation
Policy view: choose action \(a_t\sim\pi(a|s_t)\), transition to new state with tool output.

### 3) Intuition Behind Math
State-action loops turn language model into problem solver.

### 4) Architecture Flow
User goal → planner → tool execution → observation → revised plan.

### 5) Practical Example
Dataset: **GSM8K** for reasoning tasks with calculator tool.

### 6) Full Working Code
```python
from langchain.agents import initialize_agent, Tool
# tool: calculator, wiki lookup; agent type: structured chat zero-shot
```

### 7) Code Walkthrough
LLM picks tool based on textual affordances.

### 8) Exercises
Conceptual: agent vs workflow DAG.  
Math: expected utility of action sequences.  
Coding: add max-steps guardrail.

### 9) Summary Notes
- Agentic loop: think → act → observe → update.

---

## Day 23 — Tool Selection and Function Calling
### 1) Layman Explanation
Agent must pick the right tool with proper arguments.

### 2) Mathematical Foundation
Tool choice as classification over tool set; argument generation as conditional sequence modeling.

### 3) Intuition Behind Math
Constrained output schemas reduce ambiguity.

### 4) Architecture Flow
Task + tool specs → tool logits → selected tool + JSON args → execution.

### 5) Practical Example
Dataset: **API-Bank** style tool-use samples (HF equivalents).

### 6) Full Working Code
```python
# LangChain tool schema with pydantic args; call model with structured output parser
```

### 7) Code Walkthrough
Validation catches malformed arguments before tool execution.

### 8) Exercises
Conceptual: why schema constraints improve reliability.  
Math: calibration for tool probability scores.  
Coding: add fallback tool when confidence < threshold.

### 9) Summary Notes
- Tooling = structured decision + generation.

---

## Day 24 — Planning Algorithms for Agents
### 1) Layman Explanation
Complex tasks require subgoals.

### 2) Mathematical Foundation
Search objective: minimize cost \(J=\sum_t c(s_t,a_t)\); heuristic planning approximates optimal policy.

### 3) Intuition Behind Math
Explicit plans reduce myopic errors.

### 4) Architecture Flow
Goal → decompose into steps → execute each step with verification.

### 5) Practical Example
Dataset: **ALFWorld textual tasks** logs.

### 6) Full Working Code
```python
# Plan-and-execute chain: planner LLM -> executor agent -> replanner
```

### 7) Code Walkthrough
Planner output stored as structured list; executor updates progress.

### 8) Exercises
Conceptual: ReAct vs Plan-and-Execute.  
Math: dynamic programming intuition for subproblems.  
Coding: implement replanning trigger on failure.

### 9) Summary Notes
- Planning improves long-horizon reliability.

---

## Day 25 — Memory Systems (Short-Term + Long-Term)
### 1) Layman Explanation
Memory lets agent maintain context across turns/tasks.

### 2) Mathematical Foundation
State update: \(m_t=f(m_{t-1}, o_t)\). Long-term recall uses vector similarity retrieval.

### 3) Intuition Behind Math
Memory compresses history into actionable state.

### 4) Architecture Flow
Observation → summary memory + vector memory write → retrieval on demand.

### 5) Practical Example
Dataset: **MultiWOZ** dialogue tasks.

### 6) Full Working Code
```python
# LangChain ConversationBufferMemory + vectorstore-backed memory retriever
```

### 7) Code Walkthrough
Short-term keeps recent turns; long-term stores semantic facts.

### 8) Exercises
Conceptual: memory drift risks.  
Math: forgetting-factor recurrence.  
Coding: add memory TTL and eviction policy.

### 9) Summary Notes
- Good memory = coherence + personalization.

---

## Day 26 — Multi-Agent Systems
### 1) Layman Explanation
Multiple specialized agents collaborate (planner, researcher, critic).

### 2) Mathematical Foundation
Team objective decomposition: \(\max \sum_i r_i\) subject to communication constraints.

### 3) Intuition Behind Math
Specialization can improve quality but adds coordination overhead.

### 4) Architecture Flow
Orchestrator routes tasks → specialist agents → aggregation.

### 5) Practical Example
Dataset: **FEVER** claim verification with researcher+verifier agents.

### 6) Full Working Code
```python
# supervisor chain dispatches tasks to role-based agents with shared scratchpad
```

### 7) Code Walkthrough
Each agent has narrower prompt and tools.

### 8) Exercises
Conceptual: failure modes of agent communication.  
Math: latency model for parallel/serial agents.  
Coding: add voting/consensus finalizer.

### 9) Summary Notes
- Multi-agent helps complex workflows.

---

## Day 27 — Verification, Self-Critique, and Guardrails
### 1) Layman Explanation
Agent should verify itself before final answer.

### 2) Mathematical Foundation
Confidence gating: finalize if \(p(correct|trace)>\tau\), else critique loop.

### 3) Intuition Behind Math
Verification reduces hallucinations and tool misuse.

### 4) Architecture Flow
Draft → critic checks evidence/tool logs → revise/finalize.

### 5) Practical Example
Dataset: **TruthfulQA** subset.

### 6) Full Working Code
```python
# two-pass chain: answerer then verifier; reject if citation missing or contradiction detected
```

### 7) Code Walkthrough
Verifier uses explicit checklist rubric.

### 8) Exercises
Conceptual: precision-recall tradeoff in guardrails.  
Math: threshold tuning via ROC.  
Coding: implement automatic refusal on low confidence.

### 9) Summary Notes
- Reliability comes from verification loops.

---

## Day 28 — Agent Evaluation and Observability
### 1) Layman Explanation
You cannot improve what you do not measure.

### 2) Mathematical Foundation
Metrics: task success rate, step efficiency, tool accuracy, groundedness score.

### 3) Intuition Behind Math
Break outcomes into retrieval, reasoning, and action components.

### 4) Architecture Flow
Run benchmark set → trace logging → error taxonomy.

### 5) Practical Example
Dataset: **Bamboogle / HotpotQA** for tool-augmented QA evaluation.

### 6) Full Working Code
```python
# logging middleware captures tool calls, latencies, and final outcome labels
```

### 7) Code Walkthrough
Store traces for offline failure analysis.

### 8) Exercises
Conceptual: online vs offline eval.  
Math: confidence intervals for success rate.  
Coding: create dashboard-ready CSV logs.

### 9) Summary Notes
- Observability is core for production agents.

---

## Day 29 — Deployment Blueprint (Local + API)
### 1) Layman Explanation
Package your RAG+agent system as a reliable service.

### 2) Mathematical Foundation
Capacity planning: throughput \(\approx 1/(latency)\) per worker; queueing effects at high load.

### 3) Intuition Behind Math
System design constraints shape model choices.

### 4) Architecture Flow
Client → API → orchestrator → retriever/tools/model → response + logs.

### 5) Practical Example
Dataset: Use Week 3 indexed corpus for serving QA.

### 6) Full Working Code
```python
# FastAPI endpoint calling LangChain runnable with retriever + tools
```

### 7) Code Walkthrough
Add timeout/retry and caching layer for stable latency.

### 8) Exercises
Conceptual: cold-start vs warm pools.  
Math: estimate max QPS from p95 latency.  
Coding: add Redis cache for retrieval results.

### 9) Summary Notes
- Deployment is engineering + ML tradeoffs.

---

## Day 30 — Final Integration and Capstone Build Day
### 1) Layman Explanation
Assemble all components into a full agentic system.

### 2) Mathematical Foundation
Joint view:
\(P(y|q)\approx\sum_d P(d|q)\sum_{a_{1:T}}P(a_{1:T}|q,d)P(y|q,d,a_{1:T})\).

### 3) Intuition Behind Math
Answer quality depends on evidence retrieval and action sequence quality.

### 4) Architecture Flow
User query → retrieval → planner → tool calls + memory → verified final response.

### 5) Practical Example
Dataset: **HotpotQA + Wikipedia passages**, plus tool APIs (calculator, date/time, web search sandbox).

### 6) Full Working Code
Integrate retriever, planner, executor, memory, verifier in LangChain RunnableGraph.

### 7) Code Walkthrough
Trace each stage with logs and confidence gates.

### 8) Exercises
Conceptual: where does uncertainty originate most?  
Math: ablation contribution estimate per module.  
Coding: disable one module at a time and compare metrics.

### 9) Summary Notes
- You now have end-to-end LLM engineering + agentic AI competency.

### Weekly Assignment (Week 4)
Build an agent with at least 3 tools, persistent vector memory, and plan-replan loop. Provide mathematical analysis of tool selection accuracy and end-task success probability.

### Weekly Project (Week 4)
**Problem:** Research assistant agent for multi-hop questions.  
**Dataset:** HotpotQA + indexed Wikipedia passages.  
**Math:** retrieval probability, policy over tool actions, confidence thresholding.  
**Architecture:** planner, retriever, tool executor, memory, verifier.  
**Steps:** build modules, integrate traces, run benchmark.  
**Metrics:** exact match/F1, tool precision, average steps, groundedness.

---

# Capstone Project — Full Agentic AI System
## Problem Statement
Build a production-grade **multi-hop QA agent** that uses RAG, tools, memory, and iterative planning.

## Dataset
- HotpotQA for multi-hop questions
- Wikipedia passage dump (subset) for retrieval corpus
- Optional FEVER for fact-check extension

## Mathematical Approach
1. Retrieval: cosine/hybrid score \(s(d|q)\) with top-k truncation.  
2. Planning: action policy \(\pi(a_t|s_t)\).  
3. Generation: token likelihood \(\prod_t P(y_t|context)\).  
4. Verification: confidence gate \(\mathbb{1}[p>\tau]\).  
5. Overall objective: maximize task success under latency constraints.

## Architecture
- Retriever: HF embeddings + FAISS (+ optional BM25)
- Planner: LLM structured output
- Executor: tool-calling agent
- Memory: buffer + vector store
- Verifier: critic LLM + citation checks

## Implementation Steps
1. Prepare corpus and chunking.
2. Build retrieval + evaluation.
3. Implement tool registry.
4. Add planning and replanning.
5. Add memory write/read policies.
6. Add verification and refusal policies.
7. Evaluate and ablate.

## Evaluation Metrics
- QA: EM, F1
- Retrieval: Recall@k, MRR
- Agent: success rate, avg steps, tool precision
- Reliability: groundedness/hallucination rate
- Efficiency: p95 latency

---

# Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers datasets accelerate evaluate sentence-transformers faiss-cpu langchain langchain-community rank-bm25 fastapi uvicorn pydantic
```

## requirements.txt
```txt
torch
transformers
datasets
accelerate
evaluate
sentence-transformers
faiss-cpu
langchain
langchain-community
rank-bm25
fastapi
uvicorn
pydantic
```

---

# GitHub Folder Structure
```txt
llm-agentic-30days/
  data/
  notebooks/
  src/
    week1_foundations/
    week2_transformers/
    week3_rag/
    week4_agents/
    capstone/
  eval/
  configs/
  requirements.txt
  README.md
```

---

# Formula Cheat Sheet
1. Linear layer: \(y=XW+b\)  
2. ReLU: \(\max(0,x)\)  
3. GELU (approx): \(0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))\)  
4. Softmax: \(p_i=e^{z_i}/\sum_j e^{z_j}\)  
5. Cross-entropy: \(-\sum_i y_i\log p_i\)  
6. Backprop chain rule: \(\delta_l=(W_{l+1}^T\delta_{l+1})\odot f'(a_l)\)  
7. Attention: \(\text{softmax}(QK^T/\sqrt{d_k})V\)  
8. Positional encoding sinusoid equations  
9. Perplexity: \(\exp(CE)\)  
10. Cosine similarity: \(u\cdot v/(\|u\|\|v\|)\)  
11. RAG marginalization: \(\sum_d P(a|q,d)P(d|q)\)  
12. Agent policy: \(a_t\sim\pi(a|s_t)\)

---

# Learning Outcomes
By Day 30, you will be able to:
- Derive and implement core deep learning math used in LLMs.
- Build a transformer LM in PyTorch from first principles.
- Finetune HF models and evaluate perplexity/task metrics.
- Build robust RAG systems with retrieval diagnostics.
- Design agentic systems with tool use, planning, memory, and verification.
- Evaluate quality, reliability, and latency for deployment.
