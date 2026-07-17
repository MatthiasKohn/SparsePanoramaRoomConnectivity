"""
Same-home hard-negative batching for the contrastive door encoder.

The held-out failure is PRECISION on large homes: doors within one building look alike,
so the encoder confuses them -> false connectivity edges. Standard in-batch InfoNCE draws
negatives at random across homes, which are mostly EASY (a door from a different house is
trivially different). This sampler instead packs each batch with pairs from only a FEW homes,
so the in-batch negatives include OTHER doors from the SAME home -- the exact hard negatives
the encoder needs to tell same-building doors apart. No loss change; the hardness comes from
batch composition.

Usage in exp10:  DataLoader(ds, batch_sampler=HomeBatchSampler(rows, bs, homes_per_batch))
"""
from collections import defaultdict
import numpy as np


def home_of(scene):
    return str(scene).split("_floor")[0]


class HomeBatchSampler:
    def __init__(self, rows, batch_size, homes_per_batch=8, seed=0):
        self.by_home = defaultdict(list)
        for i, r in enumerate(rows):
            self.by_home[home_of(r["scene"])].append(i)
        # only homes with >=2 pairs can supply a same-home negative
        self.by_home = {h: v for h, v in self.by_home.items() if len(v) >= 2}
        self.homes = list(self.by_home)
        self.bs = int(batch_size)
        self.hpb = max(2, int(homes_per_batch))
        self.rng = np.random.default_rng(seed)
        self.n_batches = sum(len(v) for v in self.by_home.values()) // self.bs

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        pools = {h: self.rng.permutation(v).tolist() for h, v in self.by_home.items()}
        for _ in range(self.n_batches):
            avail = [h for h in self.homes if pools[h]]
            if len(avail) < 2:
                break
            self.rng.shuffle(avail)
            chosen = avail[:self.hpb]
            batch = []
            while len(batch) < self.bs:
                progressed = False
                for h in chosen:
                    if pools[h]:
                        batch.append(pools[h].pop()); progressed = True
                        if len(batch) == self.bs:
                            break
                if not progressed:                      # chosen homes exhausted -> refill
                    chosen = [h for h in self.homes if pools[h]]
                    if not chosen:
                        break
                    self.rng.shuffle(chosen); chosen = chosen[:self.hpb]
            if len(batch) == self.bs:
                yield batch


def batch_hardness(rows, batches):
    """Diagnostic: avg #same-home negatives per anchor across batches (higher = harder)."""
    homes = [home_of(r["scene"]) for r in rows]
    tot, n = 0.0, 0
    for b in batches:
        hs = [homes[i] for i in b]
        for k, h in enumerate(hs):
            tot += sum(1 for j, hj in enumerate(hs) if j != k and hj == h); n += 1
    return tot / max(n, 1)
