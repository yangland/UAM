import torch


# Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent (Peva Blanchard)
# https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html
def krum(model_sample_weight_dict, f, m):
    # normalize types
    n = int(len(model_sample_weight_dict))
    f = int(f)
    m = int(m)

    # KRUM feasibility (all ints)
    half_minus1 = (n // 2) - 1
    if f >= half_minus1:
        f = max(0, half_minus1 - 1)   # ensure f ≤ n//2 - 2
    assert f < half_minus1, f"f must be < n//2 - 1; got n={n}, f={f}"

    max_m = n - f - 2                  # neighbors kept in KRUM
    if max_m <= 0:
        raise ValueError(f"KRUM requires n ≥ 2f + 3; got n={n}, f={f}")

    m = max(1, min(m, max_m))          # 1 ≤ m ≤ n - f - 2

    # pairwise L2 distances
    l2_dist = {k: [] for k in model_sample_weight_dict.keys()}
    for name1, v1 in model_sample_weight_dict.items():
        v1 = torch.flatten(v1).float().unsqueeze(0)
        for name2, v2 in model_sample_weight_dict.items():
            if name1 == name2:
                continue
            v2 = torch.flatten(v2).float().unsqueeze(0)
            l2_dist[name1].append(torch.cdist(v1, v2, p=2).item())

    # KRUM scores: sum of smallest (n - f - 2) distances
    krum_scores = {}
    take = n - f - 2                   # still an int
    for name, dists in l2_dist.items():
        sd = sorted(dists)
        if take > len(sd):
            raise ValueError(f"take={take} > neighbors={len(sd)}; n={n}, f={f}")
        krum_scores[name] = sum(sd[:take])

    sorted_krum = sorted(krum_scores.items(), key=lambda x: x[1])
    multi_krum_m = sorted_krum[:m]
    krum_clients = [k for k, _ in multi_krum_m]

    # weights
    total_weights = n * 100.0
    ind_weight = total_weights / m
    client_weights = {k: (ind_weight if k in krum_clients else 0.0)
                      for k in model_sample_weight_dict.keys()}

    return multi_krum_m, krum_clients, client_weights
