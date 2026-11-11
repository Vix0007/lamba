"""Held-out letter/position-bias validation during training (papers' metrics on OUR held-out questions).
Rows (from build_letter_group_rows.py <tag>_heldout.jsonl): id, qid, n_opts, gold_pos, context, order, gold_orig (one row per render).
Every `every` optimizer steps (and once at train start) the training model is run under no_grad in eval mode on <= max_questions held-out
questions x all their cyclic renders; logs accuracy by presented gold position, position-balanced accuracy, RStd, fluctuation, content
consistency across renders, mean gold-vs-top-wrong margin and the predicted-letter distribution."""
import collections, json, logging, math, random
import numpy as np
import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class LetterEvalCallback(TrainerCallback):
    def __init__(self, path, tokenizer, *, every=25, max_questions=200, batch_size=8, seed=1234):
        rows = [json.loads(l) for l in open(path)]
        byq = collections.defaultdict(list)
        for r in rows: byq[r["qid"]].append(r)
        qids = sorted(byq); random.Random(seed).shuffle(qids)
        self.groups = [sorted(byq[q], key=lambda r: r["gold_pos"]) for q in qids[:max_questions]]
        self.every, self.bs, self.trainer = every, batch_size, None
        self.base_correct = None
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        n_max = max(len(g) for g in self.groups)
        _ids = [tokenizer(" " + chr(65 + i), add_special_tokens=False).input_ids for i in range(n_max)]
        assert all(len(x) == 1 for x in _ids) and len({x[0] for x in _ids}) == n_max, f"letter tokens must be single and distinct: {_ids}"
        self.letter_ids = [x[0] for x in _ids]
        self.items = []  # flat list of (group_idx, render)
        for gi, g in enumerate(self.groups):
            for r in g:
                self.items.append((gi, r, tokenizer(r["context"], add_special_tokens=False).input_ids))

    def attach(self, trainer):
        self.trainer = trainer

    @torch.no_grad()
    def evaluate(self, model):
        was_training = model.training; model.eval()
        dev = next(model.parameters()).device
        res = []  # per item: (gi, n, gold_pos, order, letter_logits[n])
        for i in range(0, len(self.items), self.bs):
            b = self.items[i:i + self.bs]; L = max(len(x[2]) for x in b)
            ids = torch.full((len(b), L), self.pad_id, dtype=torch.long); att = torch.zeros_like(ids, dtype=torch.bool)
            for k, (_, _, t) in enumerate(b): ids[k, -len(t):] = torch.tensor(t); att[k, -len(t):] = True
            ids, att = ids.to(dev), att.to(dev)
            pos = (torch.arange(L, device=dev).unsqueeze(0) - (L - att.sum(1)).unsqueeze(1)).clamp(min=0).masked_fill(~att, 0)
            out = model(ids, attention_mask=att, position_ids=pos, use_cache=False, return_dict=True, logits_to_keep=1)
            lg = out.logits[:, -1, :].float()[:, self.letter_ids].cpu().numpy()
            for k, (gi, r, _) in enumerate(b): res.append((gi, r["n_opts"], r["gold_pos"], r["order"], lg[k, :r["n_opts"]]))
        if was_training: model.train()
        pos_acc, margins, pred_cnt, winners = collections.defaultdict(lambda: [0, 0]), [], collections.Counter(), collections.defaultdict(list)
        for gi, n, gp, order, lg in res:
            pred = int(lg.argmax()); pos_acc[gp][0] += 1; pos_acc[gp][1] += int(pred == gp); pred_cnt[pred] += 1
            margins.append(float(lg[gp] - np.max(np.delete(lg, gp)))); winners[gi].append(order[pred])
        accs = np.array([pos_acc[p][1] / pos_acc[p][0] for p in sorted(pos_acc) if p < 4]) * 100
        tot = sum(v[0] for v in pos_acc.values())
        m = {"letter_val_acc": 100 * sum(v[1] for v in pos_acc.values()) / tot,
             "letter_val_pos_balanced_acc": float(accs.mean()), "letter_val_rstd": float(accs.std()),
             "letter_val_fluct": float(accs.std() / max(accs.mean(), 1e-9)), "letter_val_margin": float(np.mean(margins)),
             "letter_val_consistency": 100 * float(np.mean([len(set(w)) == 1 for w in winners.values()]))}
        for p in sorted(pos_acc):
            if p < 4: m[f"letter_val_acc_pos{'ABCD'[p]}"] = 100 * pos_acc[p][1] / pos_acc[p][0]
        for p in range(4): m[f"letter_val_pred_{'ABCD'[p]}"] = 100 * pred_cnt[p] / tot
        # paired comparison vs the FIRST evaluation (step 0 = untrained base) on the same renders
        correct = [int(int(lg.argmax()) == gp) for (_, _, gp, _, lg) in res]
        if self.base_correct is None:
            self.base_correct = correct
        else:
            b = sum(1 for x, y in zip(self.base_correct, correct) if x and not y); c = sum(1 for x, y in zip(self.base_correct, correct) if y and not x)
            n = b + c; pv = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n)
            m.update({"letter_val_vs_base_gain": c - b, "letter_val_vs_base_worse": b, "letter_val_vs_base_better": c, "letter_val_vs_base_p": pv})
        if "letter_val_acc_posA" in m and "letter_val_acc_posB" in m: m["letter_val_gap_A_minus_B"] = m["letter_val_acc_posA"] - m["letter_val_acc_posB"]
        m["letter_val_n"] = len(res)
        return m

    def _emit(self, model, state):
        m = self.evaluate(model)
        logger.info("LETTER-VAL step %s: %s", state.global_step, {k: round(v, 2) for k, v in m.items()})
        if self.trainer is not None:
            self.trainer.log({k: v for k, v in m.items()})

    def on_train_begin(self, args, state, control, model=None, **kw):
        if model is not None: self._emit(model, state)

    def on_step_end(self, args, state, control, model=None, **kw):
        if model is not None and state.global_step > 0 and state.global_step % self.every == 0: self._emit(model, state)
