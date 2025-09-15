import torch


# Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent (Peva Blanchard)
# https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html
def krum(model_sample_weight_dict, f, m):
    n = int(len(model_sample_weight_dict))
    f = int(f); m = int(m)

    # Feasibility: n ≥ 2f + 3  ⇔  f ≤ (n - 3) // 2
    # Your clamp + assert are fine, just ensure f ≤ 8 for n=20
    half_minus1 = (n // 2) - 1
    if f >= half_minus1:
        f = max(0, half_minus1 - 1)
    assert f < half_minus1, f"f must be < n//2 - 1; got n={n}, f={f}"

    max_m = n - f - 2
    if max_m <= 0:
        raise ValueError(f"KRUM requires n ≥ 2f + 3; got n={n}, f={f}")
    m = max(1, min(m, max_m))

    # Pairwise L2
    l2_dist = {k: [] for k in model_sample_weight_dict.keys()}
    for n1, v1 in model_sample_weight_dict.items():
        v1 = torch.flatten(v1).float().unsqueeze(0)
        for n2, v2 in model_sample_weight_dict.items():
            if n1 == n2: 
                continue
            v2 = torch.flatten(v2).float().unsqueeze(0)
            l2_dist[n1].append(torch.cdist(v1, v2, p=2).item())

    # Scores = sum of smallest (n - f - 2)
    take = n - f - 2
    krum_scores = {}
    for name, dists in l2_dist.items():
        sd = sorted(dists)
        if take > len(sd):
            raise ValueError(f"take={take} > neighbors={len(sd)}; n={n}, f={f}")
        krum_scores[name] = sum(sd[:take])

    sorted_krum = sorted(krum_scores.items(), key=lambda x: x[1])
    multi_krum_m = sorted_krum[:m]
    krum_clients = [k for k, _ in multi_krum_m]

    # ✅ Proper mixing weights: 1/m for selected, 0 otherwise
    client_weights = {k: (1.0/m if k in krum_clients else 0.0)
                      for k in model_sample_weight_dict.keys()}

    # Also return scores so caller can log "scores" correctly
    return multi_krum_m, krum_clients, client_weights, krum_scores
