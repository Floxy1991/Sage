"""Re-implementation of SAGE (Yang et al., ESWA 2026) on T5-base.

The authors released no code. Every choice the paper leaves open is marked
ASSUMPTION and listed in README.md. Pipeline (paper section in brackets):

  train-selfret  One T5 learns, multi-task: context -> "[STRAT] s [RESP] r"
                 (Eq. 1) and a self-evaluation task "judge" -> yes/no (P_reject).
  build-index    RQ-VAE (4 levels x 512 codes, Sinkhorn balancing) turns every
                 training response into a ResID [3.2, B.2.1].
  knowledge      COMET (xIntent/xNeed/xEffect/xWant) and HEAL expert responses
                 for each example; optional, files are cached [3.4].
  train-fusion   T5 generator whose encoder output is fused with COMET and
                 HEAL by cross-attention (Eq. 6) [3.5].
  infer          Strategy top-B_s -> ResID retrieval of R candidates per
                 strategy -> trie-constrained beam (Eq. 4) -> psychological
                 re-ranking (Eq. 5) -> fused generation. Writes predictions.jsonl
                 that metrics.py scores.
  eval-retrieval Tables 4/8: rank the test response pool by self-retrieval.

Ablation switches on `infer` match Table 6: --no-strategy-token (use a model
trained with --no-strategy-token), --no-constraint, --no-resid, --no-comet,
--no-heal, --retrieval-only.
"""
import argparse
import json
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoTokenizer, T5ForConditionalGeneration,
                          get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup)
from transformers.modeling_outputs import BaseModelOutput

from prepare_data import STRATEGIES

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPECIAL = ["[STRAT]", "[RESP]", "[CAND]", "[CTX]"]
COMET_RELS = ["xIntent", "xNeed", "xEffect", "xWant"]


# ---------------------------------------------------------------- utilities
def load(path):
    return [json.loads(l) for l in open(path)]


def seed_all(s):
    random.seed(s)
    torch.manual_seed(s)


def ctx_text(ex, turns=6):
    # ASSUMPTION: last 6 utterances of context, speaker-tagged.
    return " ".join(f"{'Seeker' if t['speaker'] == 'usr' else 'Supporter'}: {t['text']}" for t in ex["context"][-turns:])


def target_text(strategy, response, use_strategy=True):
    return f"[STRAT] {strategy} [RESP] {response}" if use_strategy else f"[RESP] {response}"


def get_tok(path):
    tok = AutoTokenizer.from_pretrained(path)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL})
    return tok


def batches(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


# ------------------------------------------------- stage 1: self-retrieval LM
def selfret_pairs(train, use_strategy, neg_ratio=1.0):
    pairs = [("context: " + ctx_text(e), target_text(e["strategy"], e["response"], use_strategy)) for e in train]
    # Self-evaluation task: real next response -> yes, response from another dialogue -> no.
    # ASSUMPTION: this is how P_reject (Sec. 3.4) is learned; the paper does not say.
    for e in train:
        pairs.append((f"judge: {ctx_text(e)} response: {e['response']}", "yes"))
        if random.random() < neg_ratio:
            o = random.choice(train)
            while o["dialog_idx"] == e["dialog_idx"]:
                o = random.choice(train)
            pairs.append((f"judge: {ctx_text(e)} response: {o['response']}", "no"))
    random.shuffle(pairs)
    return pairs


def train_seq2seq(model, tok, pairs, out, epochs, lr, bs, warmup, sched="linear", grad_accum=1, fp16=True):
    model.to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(pairs) / bs) // grad_accum
    mk = get_cosine_schedule_with_warmup if sched == "cosine" else get_linear_schedule_with_warmup
    sch = mk(opt, int(warmup * steps) if warmup < 1 else int(warmup), steps)
    use_amp = fp16 and DEV == "cuda"
    for ep in range(epochs):
        random.shuffle(pairs)
        tot = 0.0
        for i, b in enumerate(batches(pairs, bs)):
            enc = tok([x for x, _ in b], max_length=256, truncation=True, padding=True, return_tensors="pt").to(DEV)
            lab = tok([y for _, y in b], max_length=128, truncation=True, padding=True, return_tensors="pt").input_ids.to(DEV)
            lab[lab == tok.pad_token_id] = -100
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=use_amp):
                loss = model(**enc, labels=lab).loss / grad_accum
            loss.backward()
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(), sch.step(), opt.zero_grad()
            tot += loss.item() * grad_accum
        print(f"epoch {ep + 1}/{epochs} loss {tot / (i + 1):.4f}", flush=True)
        model.save_pretrained(out)
        tok.save_pretrained(out)


def cmd_train_selfret(a):
    seed_all(a.seed)
    tok = get_tok(a.base)
    model = T5ForConditionalGeneration.from_pretrained(a.base)
    model.resize_token_embeddings(len(tok))
    train = load(f"data/{a.split}/train.jsonl")
    pairs = selfret_pairs(train, not a.no_strategy_token)
    # Table B.11 "SFT" column: lr 2e-5, bs 16, 3 epochs, warmup 0.03, cosine. ASSUMPTION: T5 needs a
    # higher lr than an LLM; default 1e-4 (the "Retrieval" column), override with --lr.
    train_seq2seq(model, tok, pairs, a.out, a.epochs, a.lr, a.bs, 0.03, "cosine")


# ------------------------------------------------------ stage 2: RQ-VAE ResIDs
@torch.no_grad()
def embed(model, tok, texts, bs=64):
    model.to(DEV).eval()
    out = []
    for b in batches(texts, bs):
        enc = tok(b, max_length=128, truncation=True, padding=True, return_tensors="pt").to(DEV)
        h = model.encoder(**enc).last_hidden_state
        m = enc.attention_mask.unsqueeze(-1).float()
        out.append(F.normalize((h * m).sum(1) / m.sum(1), dim=-1).float().cpu())
    return torch.cat(out)


def sinkhorn(cost, eps=0.1, iters=10):
    q = torch.exp(-(cost - cost.min()) / (eps * cost.std().clamp_min(1e-6)))
    q /= q.sum()
    n, k = q.shape
    for _ in range(iters):
        q /= q.sum(0, keepdim=True) * k
        q /= q.sum(1, keepdim=True) * n
    return q.argmax(1)


class RQVAE(nn.Module):
    """4-level residual quantiser (B.2.1). ASSUMPTION: an MLP encoder/decoder
    over sentence vectors; the paper's 6-layer Transformer has no stated input."""

    def __init__(self, d_in=768, d_code=256, levels=4, size=512):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 512), nn.GELU(), nn.Linear(512, d_code))
        self.dec = nn.Sequential(nn.Linear(d_code, 512), nn.GELU(), nn.Linear(512, d_in))
        self.books = nn.ParameterList([nn.Parameter(torch.randn(size, d_code) * 0.1) for _ in range(levels)])

    def quantise(self, z, balance=False):
        codes, zq, r, vq, commit = [], 0, z, 0, 0
        for book in self.books:
            cost = torch.cdist(r, book) ** 2
            idx = sinkhorn(cost.detach()) if balance else cost.argmin(1)
            e = book[idx]
            vq = vq + F.mse_loss(e, r.detach())
            commit = commit + F.mse_loss(r, e.detach())
            codes.append(idx)
            zq = zq + e
            r = r - e.detach()
        return torch.stack(codes, 1), zq, vq, commit

    def forward(self, x):
        z = self.enc(x)
        codes, zq, vq, commit = self.quantise(z, balance=self.training)
        zq = z + (zq - z).detach()  # straight-through
        return codes, F.mse_loss(self.dec(zq), x) + 0.25 * vq + 0.25 * commit  # B.2.1 lambdas

    @torch.no_grad()
    def codes(self, x):
        return self.quantise(self.enc(x))[0]


def cmd_build_index(a):
    seed_all(a.seed)
    tok = get_tok(a.model)
    model = T5ForConditionalGeneration.from_pretrained(a.model)
    train = load(f"data/{a.split}/train.jsonl")
    R = embed(model, tok, [e["response"] for e in train])
    rq = RQVAE(d_in=R.shape[1]).to(DEV)
    opt = torch.optim.AdamW(rq.parameters(), lr=1e-4, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(R) / 256)
    sch = get_cosine_schedule_with_warmup(opt, min(2000, steps // 10), steps)
    # Initialise codebooks from data (k-means++-lite) to avoid dead codes.
    with torch.no_grad():
        z = rq.enc(R.to(DEV))
        for book in rq.books:
            book.copy_(z[torch.randint(len(z), (book.shape[0],))] + 1e-3 * torch.randn_like(book))
            z = z - book[torch.cdist(z, book).argmin(1)]
    for ep in range(a.epochs):
        rq.train()
        perm = torch.randperm(len(R))
        for b in batches(perm, 256):
            _, loss = rq(R[b].to(DEV))
            opt.zero_grad(), loss.backward()
            torch.nn.utils.clip_grad_norm_(rq.parameters(), 1.0)
            opt.step(), sch.step()
        if (ep + 1) % 10 == 0:
            print(f"rqvae epoch {ep + 1} loss {loss.item():.4f}", flush=True)
    rq.eval()
    codes = rq.codes(R.to(DEV)).cpu()
    used = [len(set(codes[:, m].tolist())) for m in range(codes.shape[1])]
    uniq = len(set(map(tuple, codes.tolist())))
    print(f"codes used per level {used}; unique ResIDs {uniq}/{len(codes)}")
    os.makedirs(a.out, exist_ok=True)
    torch.save({"rq": rq.state_dict(), "codes": codes, "emb": R, "d_in": R.shape[1]}, os.path.join(a.out, "index.pt"))


# --------------------------------------------------- stage 3: COMET and HEAL
def cmd_knowledge(a):
    """Cache COMET inferences and HEAL top-K expert responses per example.
    --comet: a COMET-ATOMIC-2020 BART checkpoint dir (ASSUMPTION on version).
    --heal:  JSONL of {"text": expert response} from the HEAL graph."""
    from transformers import AutoModelForSeq2SeqLM
    sent_tok = get_tok(a.model)
    sent = T5ForConditionalGeneration.from_pretrained(a.model)
    heal = [json.loads(l)["text"] for l in open(a.heal)] if a.heal else []
    heal_emb = embed(sent, sent_tok, heal) if heal else None
    if a.comet:
        ctok = AutoTokenizer.from_pretrained(a.comet)
        cmodel = AutoModelForSeq2SeqLM.from_pretrained(a.comet).to(DEV).eval()
    for split in ("train", "valid", "test"):
        rows = load(f"data/{a.split}/{split}.jsonl")
        out = []
        for b in batches(rows, 32):
            comet = [[] for _ in b]
            if a.comet:
                for rel in COMET_RELS:
                    enc = ctok([f"{e['query']} {rel} [GEN]" for e in b], return_tensors="pt", padding=True,
                               truncation=True, max_length=64).to(DEV)
                    with torch.no_grad():
                        gen = cmodel.generate(**enc, num_beams=1, max_new_tokens=16)
                    for i, t in enumerate(ctok.batch_decode(gen, skip_special_tokens=True)):
                        comet[i].append(f"{rel} {t.strip()}")
            heal_k = [[] for _ in b]
            if heal:
                q = embed(sent, sent_tok, [e["query"] for e in b])
                top = (q @ heal_emb.T).topk(a.k, dim=1).indices
                heal_k = [[heal[j] for j in row] for row in top.tolist()]
            for e, c, h in zip(b, comet, heal_k):
                out.append({"id": e["id"], "comet": c, "heal": h})
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"{split}.jsonl"), "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(split, len(out), flush=True)


def knowledge_map(kdir, split):
    p = os.path.join(kdir, f"{split}.jsonl") if kdir else None
    return {r["id"]: r for r in load(p)} if p and os.path.exists(p) else {}


# ------------------------------------------------ stage 4: fusion generator
class FusionT5(nn.Module):
    """Eq. 6: H_out = l1 H_ctx + l2 CA(H_ctx, H_C) + l3 CA(H_ctx, H_H), l softmax-normalised."""

    def __init__(self, t5):
        super().__init__()
        self.t5 = t5
        d = t5.config.d_model
        self.ca_c = nn.MultiheadAttention(d, 8, batch_first=True)
        self.ca_h = nn.MultiheadAttention(d, 8, batch_first=True)
        self.lam = nn.Parameter(torch.tensor([2.0, 0.0, 0.0]))

    def fused(self, ctx, comet, heal):
        enc = self.t5.encoder
        H = enc(**ctx).last_hidden_state
        HC = enc(**comet).last_hidden_state
        HH = enc(**heal).last_hidden_state
        l1, l2, l3 = F.softmax(self.lam, 0)
        c = self.ca_c(H, HC, HC, key_padding_mask=comet["attention_mask"] == 0)[0]
        h = self.ca_h(H, HH, HH, key_padding_mask=heal["attention_mask"] == 0)[0]
        return BaseModelOutput(last_hidden_state=l1 * H + l2 * c + l3 * h)

    def forward(self, ctx, comet, heal, labels):
        return self.t5(encoder_outputs=self.fused(ctx, comet, heal), attention_mask=ctx["attention_mask"], labels=labels)

    def generate(self, ctx, comet, heal, **kw):
        return self.t5.generate(encoder_outputs=self.fused(ctx, comet, heal), attention_mask=ctx["attention_mask"], **kw)


def fusion_inputs(tok, exs, cands, strategies, kmap, use_comet=True, use_heal=True):
    ctx = [f"[STRAT] {s} [CAND] {c} [CTX] {ctx_text(e)}" for e, c, s in zip(exs, cands, strategies)]
    comet = [" ; ".join(kmap.get(e["id"], {}).get("comet", [])) if use_comet else "" for e in exs]
    heal = [" ; ".join(kmap.get(e["id"], {}).get("heal", [])) if use_heal else "" for e in exs]
    f = lambda xs, n: tok([x or "none" for x in xs], max_length=n, truncation=True, padding=True, return_tensors="pt").to(DEV)
    return f(ctx, 384), f(comet, 128), f(heal, 256)


def load_index(path):
    ix = torch.load(os.path.join(path, "index.pt"))
    rq = RQVAE(d_in=ix["d_in"]).to(DEV)
    rq.load_state_dict(ix["rq"])
    return rq.eval(), ix["codes"], ix["emb"]


def resid_topk(qcode, qemb, codes, emb, mask, r, use_resid=True):
    """Hamming distance on ResIDs (Sec. 3.2); ASSUMPTION: ties broken by cosine."""
    cos = emb @ qemb
    score = -(codes != qcode).sum(1).float() * 10.0 + cos if use_resid else cos
    score[~mask] = -1e9
    return score.topk(min(r, int(mask.sum()))).indices.tolist()


def cmd_train_fusion(a):
    seed_all(a.seed)
    tok = get_tok(a.selfret)
    sr = T5ForConditionalGeneration.from_pretrained(a.selfret)
    train = load(f"data/{a.split}/train.jsonl")
    rq, codes, emb = load_index(a.index)
    # Candidate for training = best ResID neighbour with the gold strategy from another dialogue.
    q = embed(sr, tok, ["context: " + ctx_text(e) for e in train])
    qcodes = rq.codes(q.to(DEV)).cpu()
    strat = [e["strategy"] for e in train]
    dlg = torch.tensor([e["dialog_idx"] for e in train])
    cands = []
    for i, e in enumerate(train):
        mask = torch.tensor([s == e["strategy"] for s in strat]) & (dlg != e["dialog_idx"])
        cands.append(train[resid_topk(qcodes[i], q[i], codes, emb, mask, 1)[0]]["response"])
    kmap = knowledge_map(a.knowledge, "train")
    t5 = T5ForConditionalGeneration.from_pretrained(a.base)
    t5.resize_token_embeddings(len(tok))
    model = FusionT5(t5).to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    idx = list(range(len(train)))
    steps = a.epochs * math.ceil(len(idx) / a.bs)
    sch = get_linear_schedule_with_warmup(opt, 100, steps)
    for ep in range(a.epochs):
        random.shuffle(idx)
        tot = 0
        for n, b in enumerate(batches(idx, a.bs)):
            exs = [train[i] for i in b]
            ctx, com, hel = fusion_inputs(tok, exs, [cands[i] for i in b], [e["strategy"] for e in exs], kmap)
            lab = tok([e["response"] for e in exs], max_length=128, truncation=True, padding=True, return_tensors="pt").input_ids.to(DEV)
            lab[lab == tok.pad_token_id] = -100
            with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
                loss = model(ctx, com, hel, lab).loss
            opt.zero_grad(), loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sch.step()
            tot += loss.item()
        print(f"fusion epoch {ep + 1} loss {tot / (n + 1):.4f} lambda {F.softmax(model.lam, 0).tolist()}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, "fusion.pt"))
    tok.save_pretrained(a.out)
    json.dump({"base": a.base}, open(os.path.join(a.out, "config.json"), "w"))


# ------------------------------------------------------ stage 5: inference
@torch.no_grad()
def strategy_logprobs(model, tok, src):
    enc = tok([src] * len(STRATEGIES), max_length=256, truncation=True, return_tensors="pt", padding=True).to(DEV)
    lab = tok([f"[STRAT] {s} [RESP]" for s in STRATEGIES], return_tensors="pt", padding=True).input_ids.to(DEV)
    lab[lab == tok.pad_token_id] = -100
    logits = model(**enc, labels=lab).logits
    lp = -F.cross_entropy(logits.transpose(1, 2), lab, reduction="none", ignore_index=-100).sum(1)
    return lp.cpu()


class Trie:
    def __init__(self, seqs):
        self.root = {}
        for s in seqs:
            node = self.root
            for t in s:
                node = node.setdefault(t, {})

    def next(self, prefix):
        node = self.root
        for t in prefix:
            node = node.get(t)
            if node is None:
                return []
        return list(node.keys())


@torch.no_grad()
def constrained_beam(model, tok, src, targets, beams):
    """Eq. 4: decode only along the prefix trie of BPE-tokenised candidates."""
    seqs = [tok(t, add_special_tokens=True).input_ids for t in targets]  # ends with </s>
    trie = Trie(seqs)
    start = model.config.decoder_start_token_id
    enc = tok(src, max_length=256, truncation=True, return_tensors="pt").to(DEV)

    def allowed(_, ids):
        nxt = trie.next(ids.tolist()[1:])  # drop decoder start token
        return nxt or [tok.eos_token_id]

    out = model.generate(**enc, num_beams=beams, num_return_sequences=min(beams, len(targets)),
                         prefix_allowed_tokens_fn=allowed, max_new_tokens=160, output_scores=True,
                         return_dict_in_generate=True, length_penalty=1.0, decoder_start_token_id=start)
    texts = tok.batch_decode(out.sequences, skip_special_tokens=False)
    clean = [t.replace("<pad>", "").replace("</s>", "").strip() for t in texts]
    return list(zip(clean, out.sequences_scores.cpu().tolist()))


def split_target(t):
    if "[RESP]" in t:
        head, resp = t.split("[RESP]", 1)
        return head.replace("[STRAT]", "").strip(), resp.strip()
    return None, t.strip()


@torch.no_grad()
def p_valid(model, tok, ctx, responses):
    src = [f"judge: {ctx} response: {r}" for r in responses]
    enc = tok(src, max_length=256, truncation=True, padding=True, return_tensors="pt").to(DEV)
    start = torch.full((len(src), 1), model.config.decoder_start_token_id, device=DEV)
    logits = model(**enc, decoder_input_ids=start).logits[:, 0]
    yes, no = tok("yes", add_special_tokens=False).input_ids[0], tok("no", add_special_tokens=False).input_ids[0]
    return F.softmax(logits[:, [yes, no]], -1)[:, 0].cpu()


def cmd_infer(a):
    seed_all(a.seed)
    tok = get_tok(a.selfret)
    sr = T5ForConditionalGeneration.from_pretrained(a.selfret).to(DEV).eval()
    rq, codes, emb = load_index(a.index)
    train = load(f"data/{a.split}/train.jsonl")
    test = load(f"data/{a.split}/{a.eval_split}.jsonl")
    if a.limit:
        test = test[:a.limit]
    kmap = knowledge_map(a.knowledge, a.eval_split)
    strat_of = [e["strategy"] for e in train]
    fusion = None
    if a.fusion and not a.retrieval_only:
        cfg = json.load(open(os.path.join(a.fusion, "config.json")))
        t5 = T5ForConditionalGeneration.from_pretrained(cfg["base"])
        t5.resize_token_embeddings(len(tok))
        fusion = FusionT5(t5)
        fusion.load_state_dict(torch.load(os.path.join(a.fusion, "fusion.pt"), map_location="cpu"))
        fusion.to(DEV).eval()
    os.makedirs(a.out, exist_ok=True)
    f = open(os.path.join(a.out, "predictions.jsonl"), "w")
    nll_sum = ntok = 0
    for i, e in enumerate(test):
        src = "context: " + ctx_text(e)
        q = embed(sr, tok, [src])[0]
        qcode = rq.codes(q.unsqueeze(0).to(DEV))[0].cpu()
        lp = strategy_logprobs(sr, tok, src)
        sprob = F.softmax(lp / a.tau, 0)
        top_s = [STRATEGIES[j] for j in sprob.topk(a.bs_strat).indices.tolist()]
        cands = []  # (strategy, response, strategy prob)
        for s in top_s:
            mask = torch.tensor([x == s for x in strat_of])
            ids = resid_topk(qcode, q, codes, emb, mask, a.r, use_resid=not a.no_resid)
            if a.no_constraint:
                enc = tok(src, return_tensors="pt", max_length=256, truncation=True).to(DEV)
                dec = tok(f"[STRAT] {s} [RESP]", add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
                start = torch.full((1, 1), sr.config.decoder_start_token_id, device=DEV)
                gen = sr.generate(**enc, decoder_input_ids=torch.cat([start, dec], 1), num_beams=a.br,
                                  num_return_sequences=a.br, max_new_tokens=128, no_repeat_ngram_size=3)
                outs = [split_target(t)[1] for t in tok.batch_decode(gen, skip_special_tokens=False)]
                outs = [o.replace("<pad>", "").replace("</s>", "").strip() for o in outs]
            else:
                targets = [target_text(s, train[j]["response"]) for j in ids]
                outs = [split_target(t)[1] for t, _ in constrained_beam(sr, tok, src, targets, a.br)]
            cands += [(s, o, float(sprob[STRATEGIES.index(s)])) for o in outs if o]
        if not cands:  # Sec. 3.3 fallback
            cands = [("Information", train[0]["response"], 1e-6)]
        # Eq. 5, in log space.
        pv = p_valid(sr, tok, ctx_text(e), [c[1] for c in cands])
        log_s = torch.tensor([math.log(c[2] + 1e-12) for c in cands]) + torch.log(pv + 1e-12) / a.delta
        kn = kmap.get(e["id"], {})
        cemb = embed(sr, tok, [c[1] for c in cands])
        if kn.get("comet") and not a.no_comet:
            ce = embed(sr, tok, kn["comet"])
            log_s += a.alpha * torch.log((cemb @ ce.T).mean(1).clamp_min(1e-6))
        if kn.get("heal") and not a.no_heal:
            he = embed(sr, tok, kn["heal"])
            log_s += a.beta * torch.log((cemb @ he.T).max(1).values.clamp_min(1e-6))
        best = int(log_s.argmax())
        s_star, r_star = cands[best][0], cands[best][1]
        pred = r_star
        if fusion is not None:
            ctx, com, hel = fusion_inputs(tok, [e], [r_star], [s_star], kmap, not a.no_comet, not a.no_heal)
            gen = fusion.generate(ctx, com, hel, num_beams=4, max_new_tokens=128, min_new_tokens=10,
                                  repetition_penalty=1.2, no_repeat_ngram_size=3, length_penalty=1.0)
            pred = tok.decode(gen[0], skip_special_tokens=True).strip()
            # PPL: gold response under the fusion generator, given the gold strategy (ASSUMPTION).
            ctx, com, hel = fusion_inputs(tok, [e], [r_star], [e["strategy"]], kmap, not a.no_comet, not a.no_heal)
            lab = tok([e["response"]], return_tensors="pt", max_length=128, truncation=True).input_ids.to(DEV)
            with torch.no_grad():
                loss = fusion(ctx, com, hel, lab).loss
            nll_sum += loss.item() * lab.shape[1]
            ntok += lab.shape[1]
        f.write(json.dumps({"id": e["id"], "response": e["response"], "strategy": e["strategy"],
                            "pred_strategy": top_s[0], "chosen_strategy": s_star, "prediction": pred,
                            "candidate": r_star}) + "\n")
        if (i + 1) % 50 == 0:
            f.flush()
            print(f"{i + 1}/{len(test)}", flush=True)
    f.close()
    from metrics import evaluate_file
    res = evaluate_file(os.path.join(a.out, "predictions.jsonl"))
    if ntok:
        res["PPL"] = math.exp(nll_sum / ntok)
    res["args"] = vars(a)
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=2))


def cmd_eval_retrieval(a):
    """Tables 4/8. ASSUMPTION: pool = a random --pool-size subset of test responses
    that always contains the gold one (the paper never states its pool; on the full
    3.5k-response test pool BM25 gets R@10 = 11.4, far below the paper's DPR 61.8)."""
    seed_all(a.seed)
    tok = get_tok(a.selfret)
    sr = T5ForConditionalGeneration.from_pretrained(a.selfret).to(DEV).eval()
    rq, _, _ = load_index(a.index)
    test = load(f"data/{a.split}/test.jsonl")
    if a.limit:
        test = test[:a.limit]
    pemb = embed(sr, tok, [e["response"] for e in test])
    pcodes = rq.codes(pemb.to(DEV)).cpu()
    rows = []
    for i, e in enumerate(test):
        pool = random.sample([j for j in range(len(test)) if j != i], min(a.pool_size - 1, len(test) - 1)) + [i]
        src = "context: " + ctx_text(e)
        q = embed(sr, tok, [src])[0]
        qcode = rq.codes(q.unsqueeze(0).to(DEV))[0].cpu()
        sub = torch.tensor(pool)
        pre = [pool[j] for j in resid_topk(qcode, q, pcodes[sub], pemb[sub], torch.ones(len(pool), dtype=torch.bool), a.r)]
        targets = {target_text(test[j]["strategy"], test[j]["response"]): j for j in pre}
        ranked = constrained_beam(sr, tok, src, list(targets), min(10, len(targets)))
        ids = [targets.get(t, -1) for t, _ in ranked]
        ids = [j for j in ids if j >= 0]
        rows.append({"id": e["id"], "ranked_ids": ids, "gold_id": i, "num_prev_turns": e["num_prev_turns"]})
    from metrics import retrieval_metrics
    res = retrieval_metrics([r["ranked_ids"] for r in rows], [r["gold_id"] for r in rows])
    res["pool_size"] = a.pool_size
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "retrieval.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--split", default="paper")
    common.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("train-selfret", parents=[common])
    p.add_argument("--base", default="t5-base")
    p.add_argument("--out", default="ckpt/selfret")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--no-strategy-token", action="store_true")

    p = sub.add_parser("build-index", parents=[common])
    p.add_argument("--model", default="ckpt/selfret")
    p.add_argument("--out", default="ckpt/index")
    p.add_argument("--epochs", type=int, default=50)

    p = sub.add_parser("knowledge", parents=[common])
    p.add_argument("--model", default="ckpt/selfret")
    p.add_argument("--comet", default="")
    p.add_argument("--heal", default="")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--out", default="data/knowledge")

    p = sub.add_parser("train-fusion", parents=[common])
    p.add_argument("--base", default="t5-base")
    p.add_argument("--selfret", default="ckpt/selfret")
    p.add_argument("--index", default="ckpt/index")
    p.add_argument("--knowledge", default="data/knowledge")
    p.add_argument("--out", default="ckpt/fusion")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=16)

    p = sub.add_parser("infer", parents=[common])
    p.add_argument("--selfret", default="ckpt/selfret")
    p.add_argument("--index", default="ckpt/index")
    p.add_argument("--fusion", default="ckpt/fusion")
    p.add_argument("--knowledge", default="data/knowledge")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--out", default="results/sage")
    p.add_argument("--limit", type=int, default=0)
    # ASSUMPTIONS: none of these values are reported in the paper.
    p.add_argument("--bs-strat", type=int, default=3)
    p.add_argument("--r", type=int, default=20)
    p.add_argument("--br", type=int, default=4)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    for flag in ("no-constraint", "no-resid", "no-comet", "no-heal", "retrieval-only"):
        p.add_argument(f"--{flag}", action="store_true")

    p = sub.add_parser("eval-retrieval", parents=[common])
    p.add_argument("--selfret", default="ckpt/selfret")
    p.add_argument("--index", default="ckpt/index")
    p.add_argument("--out", default="results/sage")
    p.add_argument("--pool-size", type=int, default=100)
    p.add_argument("--r", type=int, default=20)
    p.add_argument("--limit", type=int, default=0)

    a = ap.parse_args()
    {"train-selfret": cmd_train_selfret, "build-index": cmd_build_index, "knowledge": cmd_knowledge,
     "train-fusion": cmd_train_fusion, "infer": cmd_infer, "eval-retrieval": cmd_eval_retrieval}[a.cmd](a)


if __name__ == "__main__":
    main()
