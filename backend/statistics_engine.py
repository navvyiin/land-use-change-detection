"""
Land Use Change Statistics Engine
===================================
Implements a comprehensive suite of change-detection statistics:

  1. Change Matrix + Transition Counts
  2. Markov Chain (transition probabilities, steady-state, mixing time)
  3. Accuracy Metrics (overall accuracy, Kappa, per-class PA/UA)
  4. Moran's I Spatial Autocorrelation (with Z-test)
  5. Landscape Metrics (NP, MPS, PD, ED, LPI, Fragmentation, Contagion)
  6. Information Theory (Shannon Entropy, KL Divergence, Redundancy)
  7. Pontius Decomposition (Net vs Swap change)
  8. Chi-Square Test of Independence
  9. Annual Rate of Change (compound formula)
 10. Vulnerability Index (weighted loss rate)

All results are JSON-serialisable via the *_result() methods.
"""

import numpy as np
from scipy import stats as scipy_stats
from scipy.ndimage import label as scipy_label, convolve as scipy_convolve
from typing import Tuple


PIXEL_AREA_HA = 0.09    # 30 m × 30 m = 900 m² = 0.09 ha


class LandUseStatistics:

    def __init__(self, arr_a: np.ndarray, arr_b: np.ndarray,
                 classes: dict, n_years: int = 10,
                 pixel_area_ha: float = PIXEL_AREA_HA):
        self.a           = arr_a.astype(np.int32)
        self.b           = arr_b.astype(np.int32)
        self.classes     = classes
        self.n_years     = max(n_years, 1)
        self.px_ha       = pixel_area_ha
        self.class_ids   = sorted(k for k in classes if k != 0)
        self.n            = len(self.class_ids)
        self.id2idx      = {cid: i for i, cid in enumerate(self.class_ids)}

        # Computed by run_all()
        self._matrix     : np.ndarray | None = None   # (n × n) int64 pixel counts
        self._prob_matrix: np.ndarray | None = None   # (n × n) float64 row-stochastic
        self._cached     : dict               = {}

    # ── Orchestration ─────────────────────────────────────────────────────────

    def run_all(self):
        self._compute_matrix()
        self._compute_prob_matrix()
        # Pre-compute all results
        for method in [
            self.change_matrix_result, self.markov_result,
            self.accuracy_result, self.morans_i_result,
            self.landscape_result, self.information_result,
            self.pontius_result, self.chi_square_result,
            self.rate_of_change_result, self.vulnerability_result,
        ]:
            try:
                method()
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════════════════════
    # 1. CHANGE MATRIX
    # ══════════════════════════════════════════════════════════════════════════

    def change_matrix_result(self) -> dict:
        if "change_matrix" in self._cached:
            return self._cached["change_matrix"]

        m = self._matrix
        labels = [self.classes[cid]["name"] for cid in self.class_ids]
        colors = [self.classes[cid]["color"] for cid in self.class_ids]

        rows, total_off_diag_ha = [], 0.0
        for i, from_id in enumerate(self.class_ids):
            row_total = int(m[i].sum())
            values = []
            for j, to_id in enumerate(self.class_ids):
                px  = int(m[i, j])
                ha  = round(px * self.px_ha, 2)
                pct = round(px / row_total * 100, 1) if row_total > 0 else 0.0
                if i != j:
                    total_off_diag_ha += ha
                values.append({"to_id": to_id, "to_name": self.classes[to_id]["name"],
                                "pixels": px, "ha": ha, "pct_of_from": pct})
            rows.append({"from_id": from_id, "from_name": self.classes[from_id]["name"],
                         "row_total_ha": round(row_total * self.px_ha, 2), "values": values})

        result = {
            "labels":             labels,
            "colors":             colors,
            "matrix":             rows,
            "total_changed_ha":   round(total_off_diag_ha, 2),
            "units":              "hectares",
        }
        self._cached["change_matrix"] = result
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # 2. MARKOV CHAIN
    # ══════════════════════════════════════════════════════════════════════════

    def markov_result(self) -> dict:
        if "markov" in self._cached:
            return self._cached["markov"]

        P = self._prob_matrix       # (n × n) row-stochastic
        labels = [self.classes[cid]["name"] for cid in self.class_ids]

        # ── Steady-state vector (left eigenvector of P, i.e. right eigenvector of P^T) ──
        eigvals, eigvecs = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        steady_raw = np.real(eigvecs[:, idx])
        steady_raw = np.abs(steady_raw)
        steady_state = (steady_raw / steady_raw.sum()).tolist()

        # ── Predicted distribution after n_years steps ──
        # Start from Year A empirical distribution
        initial_dist = np.array([
            (self.a == cid).sum() for cid in self.class_ids
        ], dtype=float)
        if initial_dist.sum() > 0:
            initial_dist /= initial_dist.sum()

        # P^n using repeated squaring
        Pn = np.linalg.matrix_power(P, self.n_years)
        predicted_dist = (initial_dist @ Pn).tolist()

        # ── Persistence probability (diagonal) ──
        persistence = {self.classes[cid]["name"]: round(float(P[i, i]), 4)
                       for i, cid in enumerate(self.class_ids)}

        # ── Mixing time approximation (second-largest |eigenvalue|) ──
        abs_eigs = np.sort(np.abs(eigvals))[::-1]
        lambda2 = float(abs_eigs[1]) if len(abs_eigs) > 1 else 0.0
        mixing_time = (
            round(-1.0 / np.log(lambda2), 1) if 0 < lambda2 < 1 else None
        )

        # Serialise transition probability matrix rows
        prob_rows = []
        for i, from_id in enumerate(self.class_ids):
            probs = [{"to_name": self.classes[to_id]["name"],
                      "probability": round(float(P[i, j]), 4)}
                     for j, to_id in enumerate(self.class_ids)]
            prob_rows.append({"from_name": self.classes[from_id]["name"], "probs": probs})

        result = {
            "labels":              labels,
            "transition_matrix":   [[round(float(P[i,j]),4) for j in range(self.n)]
                                     for i in range(self.n)],
            "prob_rows":           prob_rows,
            "steady_state":        [round(v, 4) for v in steady_state],
            "predicted_dist":      [round(v, 4) for v in predicted_dist],
            "persistence":         persistence,
            "mixing_time_years":   mixing_time,
            "interpretation": (
                f"At steady state, dominant class is "
                f"'{labels[int(np.argmax(steady_state))]}' "
                f"({round(max(steady_state)*100,1)}%)."
            ),
        }
        self._cached["markov"] = result
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # 3. ACCURACY / KAPPA
    # ══════════════════════════════════════════════════════════════════════════

    def accuracy_result(self) -> dict:
        if "accuracy" in self._cached:
            return self._cached["accuracy"]

        m = self._matrix.astype(float)
        total = m.sum()
        if total == 0:
            return {}

        # Overall accuracy
        oa = float(m.diagonal().sum() / total)

        # Cohen's Kappa
        row_sums = m.sum(axis=1)
        col_sums = m.sum(axis=0)
        Pe = float((row_sums * col_sums).sum()) / (total ** 2)
        Po = oa
        kappa = (Po - Pe) / (1 - Pe) if Pe < 1 else 0.0

        # Per-class Producer's Accuracy (PA) and User's Accuracy (UA)
        per_class = []
        for i, cid in enumerate(self.class_ids):
            pa = float(m[i, i] / col_sums[i]) if col_sums[i] > 0 else 0.0
            ua = float(m[i, i] / row_sums[i]) if row_sums[i] > 0 else 0.0
            f1 = (2 * pa * ua / (pa + ua)) if (pa + ua) > 0 else 0.0
            per_class.append({
                "class_name":          self.classes[cid]["name"],
                "color":               self.classes[cid]["color"],
                "producers_accuracy":  round(pa, 4),
                "users_accuracy":      round(ua, 4),
                "f1_score":            round(f1, 4),
            })

        kappa_interp = (
            "< 0 — Worse than random" if kappa < 0 else
            "0.01–0.20 — Slight" if kappa < 0.21 else
            "0.21–0.40 — Fair" if kappa < 0.41 else
            "0.41–0.60 — Moderate" if kappa < 0.61 else
            "0.61–0.80 — Substantial" if kappa < 0.81 else
            "0.81–1.00 — Almost perfect"
        )

        result = {
            "overall_accuracy":     round(oa, 4),
            "kappa":                round(kappa, 4),
            "kappa_interpretation": kappa_interp,
            "expected_accuracy_Pe": round(Pe, 4),
            "per_class":            per_class,
        }
        self._cached["accuracy"] = result
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # 4. MORAN'S I SPATIAL AUTOCORRELATION
    # ══════════════════════════════════════════════════════════════════════════

    def morans_i_result(self) -> dict:
        if "morans_i" in self._cached:
            return self._cached["morans_i"]

        # Work on the binary change map; subsample for performance
        changed = (self.a != self.b).astype(np.float64)
        changed = self._subsample(changed, max_size=256)

        H, W = changed.shape
        n    = H * W

        z    = changed - changed.mean()
        z_sq_sum = float((z ** 2).sum())

        if z_sq_sum < 1e-12:
            result = {
                "morans_i": 0.0, "expected_i": round(-1/(n-1), 6),
                "z_score": 0.0, "p_value": 1.0,
                "interpretation": "No spatial variation in change map."
            }
            self._cached["morans_i"] = result
            return result

        # Queen contiguity kernel (8-neighbours)
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=float)
        Wz = scipy_convolve(z, kernel, mode="reflect")

        # Number of weights per interior pixel ≈ 8; edge pixels have fewer
        W_count = scipy_convolve(np.ones_like(z), kernel, mode="reflect")
        S0 = float(W_count.sum())                 # sum of all weights

        # Moran's I = (n / S0) * (z · Wz) / (z · z)
        numerator   = float((z * Wz).sum())
        I_val       = (n / S0) * (numerator / z_sq_sum)

        # Expected value and variance under normality assumption
        E_I  = -1.0 / (n - 1)
        # Variance (simplified normal approximation for large n)
        S1   = float(2 * W_count.sum())           # approx
        S2   = float(4 * (W_count ** 2).sum())    # approx
        n_f  = float(n)
        b2   = float((z ** 4).sum() * n_f) / (z_sq_sum ** 2)
        Var_I = (n_f * ((n_f**2 - 3*n_f + 3)*S1 - n_f*S2 + 3*S0**2)
                 - b2 * ((n_f**2 - n_f)*S1 - 2*n_f*S2 + 6*S0**2)
                ) / ((n_f-1)*(n_f-2)*(n_f-3)*S0**2) - E_I**2

        std_I = float(np.sqrt(max(Var_I, 1e-12)))
        Z     = (I_val - E_I) / std_I
        p     = float(2 * (1 - scipy_stats.norm.cdf(abs(Z))))

        interp = (
            "Strong positive spatial autocorrelation — change clusters in patches." if I_val > 0.3 else
            "Moderate positive spatial autocorrelation."                             if I_val > 0.1 else
            "Weak or no spatial autocorrelation — change distributed near-randomly." if I_val > -0.05 else
            "Negative spatial autocorrelation — change is dispersed (checkerboard)."
        )

        result = {
            "morans_i":      round(I_val, 5),
            "expected_i":    round(E_I, 6),
            "variance":      round(Var_I, 8),
            "z_score":       round(Z, 4),
            "p_value":       round(p, 5),
            "significant":   p < 0.05,
            "interpretation": interp,
            "subsample_size": f"{H}×{W}",
        }
        self._cached["morans_i"] = result
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # 5. LANDSCAPE METRICS
    # ══════════════════════════════════════════════════════════════════════════

    def landscape_result(self) -> dict:
        if "landscape" in self._cached:
            return self._cached["landscape"]

        # Subsample for patch detection
        arr_a = self._subsample(self.a, max_size=512)
        arr_b = self._subsample(self.b, max_size=512)
        total_cells = arr_a.size

        metrics_a = self._compute_landscape(arr_a, total_cells)
        metrics_b = self._compute_landscape(arr_b, total_cells)

        # Contagion index (landscape-level; based on Year B)
        contagion = self._contagion_index(arr_b)

        result = {
            "year_a": metrics_a,
            "year_b": metrics_b,
            "contagion_b": round(contagion, 4),
            "note": "Contagion ∈ [0,1]: 1=fully aggregated, 0=maximally dispersed."
        }
        self._cached["landscape"] = result
        return result

    def _compute_landscape(self, arr: np.ndarray, total_cells: int) -> list:
        results = []
        for cid in self.class_ids:
            binary = (arr == cid).astype(np.int32)
            if binary.sum() == 0:
                results.append(self._empty_landscape(cid))
                continue

            labeled, n_patches = scipy_label(binary)
            patch_sizes = np.bincount(labeled.ravel())[1:]   # exclude background

            total_ha  = float(binary.sum()) * self.px_ha
            mps       = float(patch_sizes.mean()) * self.px_ha     # Mean Patch Size (ha)
            lpi       = float(patch_sizes.max()) / total_cells * 100  # Largest Patch Index %
            pd        = n_patches / (total_cells * self.px_ha) * 100  # patches / 100 ha

            # Edge density: perimeter pixels / total area
            # Perimeter = cells adjacent to a different class
            from scipy.ndimage import binary_erosion
            interior = binary_erosion(binary, iterations=1)
            edge_px  = int(binary.sum() - interior.sum())
            ed       = edge_px * 30 / (total_cells * self.px_ha * 10000)  # m / ha

            frag     = 1.0 - float(patch_sizes.max()) / binary.sum()   # fragmentation

            results.append({
                "class_id":           cid,
                "class_name":         self.classes[cid]["name"],
                "color":              self.classes[cid]["color"],
                "n_patches":          int(n_patches),
                "total_area_ha":      round(total_ha, 2),
                "mean_patch_size_ha": round(mps, 3),
                "largest_patch_pct":  round(lpi, 3),
                "patch_density":      round(pd, 4),
                "edge_density_m_ha":  round(ed, 2),
                "fragmentation_idx":  round(frag, 4),
            })
        return results

    def _empty_landscape(self, cid: int) -> dict:
        return {
            "class_id": cid, "class_name": self.classes[cid]["name"],
            "color": self.classes[cid]["color"],
            "n_patches": 0, "total_area_ha": 0.0, "mean_patch_size_ha": 0.0,
            "largest_patch_pct": 0.0, "patch_density": 0.0,
            "edge_density_m_ha": 0.0, "fragmentation_idx": 0.0,
        }

    def _contagion_index(self, arr: np.ndarray) -> float:
        """
        FRAGSTATS contagion index.
        C = 1 + sum_i sum_j [ p_i * p_ij * ln(p_i * p_ij) ] / (2 * ln(n_classes))
        where p_i = proportion of class i, p_ij = P(adj cell is class j | cell is class i).
        """
        n_cls = len(self.class_ids)
        if n_cls < 2:
            return 1.0

        # Build adjacency counts
        adj = np.zeros((n_cls, n_cls), dtype=np.float64)
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            a_shifted = np.roll(arr, (di, dj), axis=(0,1))
            valid = (arr > 0) & (a_shifted > 0)
            for ii, ci in enumerate(self.class_ids):
                mask = valid & (arr == ci)
                for jj, cj in enumerate(self.class_ids):
                    adj[ii, jj] += float(np.sum(mask & (a_shifted == cj)))

        total = adj.sum()
        if total == 0:
            return 0.0

        p_i = adj.sum(axis=1) / total    # proportion of class i adjacencies
        with np.errstate(divide="ignore", invalid="ignore"):
            row_sums = adj.sum(axis=1, keepdims=True)
            p_ij = np.where(row_sums > 0, adj / row_sums, 0.0)

        C_sum = 0.0
        for ii in range(n_cls):
            for jj in range(n_cls):
                v = p_i[ii] * p_ij[ii, jj]
                if v > 0:
                    C_sum += v * np.log(v)

        contagion = 1.0 + C_sum / (2.0 * np.log(n_cls))
        return float(np.clip(contagion, 0.0, 1.0))

    # ══════════════════════════════════════════════════════════════════════════
    # 6. INFORMATION THEORY
    # ══════════════════════════════════════════════════════════════════════════

    def information_result(self) -> dict:
        if "information" in self._cached:
            return self._cached["information"]

        def proportions(arr):
            total = np.sum(arr > 0)
            if total == 0:
                return np.zeros(len(self.class_ids))
            return np.array([np.sum(arr == cid) / total for cid in self.class_ids])

        p_a = proportions(self.a)
        p_b = proportions(self.b)
        max_entropy = np.log2(len(self.class_ids))

        def shannon_entropy(p):
            with np.errstate(divide="ignore", invalid="ignore"):
                return float(-np.sum(np.where(p > 0, p * np.log2(p), 0)))

        H_a = shannon_entropy(p_a)
        H_b = shannon_entropy(p_b)

        # KL Divergence D_KL(P_b || P_a) — nats
        with np.errstate(divide="ignore", invalid="ignore"):
            kl_div = float(np.sum(np.where(
                (p_b > 0) & (p_a > 0),
                p_b * np.log(p_b / p_a), 0.0
            )))

        # Jensen-Shannon Divergence (symmetric, bounded [0,1])
        m = (p_a + p_b) / 2
        jsd = float(0.5 * np.sum(np.where(p_a > 0, p_a * np.log(p_a / np.where(m>0,m,1)), 0))
                  + 0.5 * np.sum(np.where(p_b > 0, p_b * np.log(p_b / np.where(m>0,m,1)), 0)))

        # Redundancy: how much entropy is "not used" relative to maximum
        redundancy_a = (max_entropy - H_a) / max_entropy if max_entropy > 0 else 0
        redundancy_b = (max_entropy - H_b) / max_entropy if max_entropy > 0 else 0

        # Relative entropy change
        delta_H = H_b - H_a

        per_class = [{
            "class_name":  self.classes[cid]["name"],
            "color":       self.classes[cid]["color"],
            "proportion_a": round(float(p_a[i]), 5),
            "proportion_b": round(float(p_b[i]), 5),
            "info_content_a": round(float(-np.log2(p_a[i])) if p_a[i]>0 else 0, 3),
            "info_content_b": round(float(-np.log2(p_b[i])) if p_b[i]>0 else 0, 3),
        } for i, cid in enumerate(self.class_ids)]

        result = {
            "shannon_entropy_a":    round(H_a, 5),
            "shannon_entropy_b":    round(H_b, 5),
            "max_entropy":          round(max_entropy, 5),
            "delta_entropy":        round(delta_H, 5),
            "kl_divergence":        round(kl_div, 5),
            "jensen_shannon_div":   round(jsd, 5),
            "redundancy_a":         round(redundancy_a, 4),
            "redundancy_b":         round(redundancy_b, 4),
            "per_class":            per_class,
            "interpretation": (
                f"Landscape diversity {'increased' if delta_H > 0 else 'decreased'} "
                f"(ΔH = {delta_H:+.3f} bits). "
                f"KL divergence = {kl_div:.4f} nats indicates "
                f"{'significant' if kl_div > 0.1 else 'minor'} compositional shift."
            ),
        }
        self._cached["information"] = result
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # 7. PONTIUS DECOMPOSITION
    # ══════════════════════════════════════════════════════════════════════════

    def pontius_result(self) -> dict:
        if "pontius" in self._cached:
            return self._cached["pontius"]

        per_class = []
        total_net, total_swap = 0.0, 0.0

        for cid in self.class_ids:
            gain_px = int(np.sum((self.a != cid) & (self.b == cid)))
            loss_px = int(np.sum((self.a == cid) & (self.b != cid)))
            pers_px = int(np.sum((self.a == cid) & (self.b == cid)))

            gain_ha = gain_px * self.px_ha
            loss_ha = loss_px * self.px_ha
            pers_ha = pers_px * self.px_ha

            net  = abs(gain_ha - loss_ha)
            swap = 2 * min(gain_ha, loss_ha)
            total_change = gain_ha + loss_ha  # = net + swap

            total_net  += net
            total_swap += swap

            per_class.append({
                "class_name":        self.classes[cid]["name"],
                "color":             self.classes[cid]["color"],
                "persistence_ha":    round(pers_ha,  2),
                "gain_ha":           round(gain_ha,  2),
                "loss_ha":           round(loss_ha,  2),
                "net_change_ha":     round(net,       2),
                "swap_change_ha":    round(swap,      2),
                "total_change_ha":   round(total_change, 2),
                "net_direction":     "gain" if gain_ha > loss_ha else ("loss" if loss_ha > gain_ha else "stable"),
            })

        result = {
            "per_class":          per_class,
            "total_net_ha":       round(total_net, 2),
            "total_swap_ha":      round(total_swap, 2),
            "grand_total_change": round(total_net + total_swap, 2),
            "swap_fraction":      round(total_swap / (total_net + total_swap), 4)
                                  if (total_net + total_swap) > 0 else 0.0,
            "interpretation": (
                "Swap change dominates — most transitions are reciprocal exchanges "
                "(e.g. forest ↔ coffee)."
                if total_swap > total_net else
                "Net change dominates — landscape is undergoing systematic directional shift."
            ),
        }
        self._cached["pontius"] = result
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # 8. CHI-SQUARE TEST
    # ══════════════════════════════════════════════════════════════════════════

    def chi_square_result(self) -> dict:
        if "chi_square" in self._cached:
            return self._cached["chi_square"]

        m = self._matrix.astype(float)
        # Remove all-zero rows/cols for validity
        row_mask = m.sum(axis=1) > 0
        col_mask = m.sum(axis=0) > 0
        m_valid  = m[np.ix_(row_mask, col_mask)]

        if m_valid.min() < 5:
            pass   # warn but still compute

        try:
            chi2, p, dof, expected = scipy_stats.chi2_contingency(m_valid,
                                                                    correction=False)
            # Cramér's V (effect size)
            n = m_valid.sum()
            min_dim = min(m_valid.shape) - 1
            cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if n * min_dim > 0 else 0.0
        except Exception as e:
            return {"error": str(e)}

        result = {
            "chi2_statistic": round(float(chi2), 4),
            "p_value":        round(float(p), 8),
            "degrees_of_freedom": int(dof),
            "significant":    float(p) < 0.05,
            "cramers_v":      round(cramers_v, 4),
            "effect_size": (
                "Negligible" if cramers_v < 0.1 else
                "Small"      if cramers_v < 0.3 else
                "Medium"     if cramers_v < 0.5 else
                "Large"
            ),
            "interpretation": (
                f"χ²({int(dof)}) = {chi2:.2f}, p {'< 0.001' if p < 0.001 else f'= {p:.4f}'}. "
                "Land cover transitions are statistically non-random."
                if float(p) < 0.05 else
                "Land cover transitions do not significantly deviate from independence."
            ),
        }
        self._cached["chi_square"] = result
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # 9. ANNUAL RATE OF CHANGE
    # ══════════════════════════════════════════════════════════════════════════

    def rate_of_change_result(self) -> dict:
        if "rate_of_change" in self._cached:
            return self._cached["rate_of_change"]

        per_class = []
        for cid in self.class_ids:
            area_a = float(np.sum(self.a == cid)) * self.px_ha
            area_b = float(np.sum(self.b == cid)) * self.px_ha

            if area_a > 0 and area_b > 0:
                # FAO compound annual rate of change formula
                # r = (A_b / A_a)^(1/n_years) - 1
                annual_rate = float((area_b / area_a) ** (1.0 / self.n_years) - 1.0)
            elif area_a == 0:
                annual_rate = None
            else:
                annual_rate = -1.0   # Complete loss

            per_class.append({
                "class_name":       self.classes[cid]["name"],
                "color":            self.classes[cid]["color"],
                "area_a_ha":        round(area_a, 2),
                "area_b_ha":        round(area_b, 2),
                "change_ha":        round(area_b - area_a, 2),
                "change_pct":       round((area_b - area_a) / area_a * 100, 2) if area_a > 0 else None,
                "annual_rate_pct":  round(annual_rate * 100, 3) if annual_rate is not None else None,
                "half_life_years":  round(-np.log(2) / np.log(1 + annual_rate), 1)
                                    if annual_rate is not None and -1 < annual_rate < 0 else None,
                "doubling_time_years": round(np.log(2) / np.log(1 + annual_rate), 1)
                                       if annual_rate is not None and annual_rate > 0 else None,
            })

        result = {
            "n_years":   self.n_years,
            "per_class": per_class,
            "formula":   "r = (A_t2 / A_t1)^(1/Δt) − 1   [FAO compound rate]",
        }
        self._cached["rate_of_change"] = result
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # 10. VULNERABILITY INDEX
    # ══════════════════════════════════════════════════════════════════════════

    def vulnerability_result(self) -> dict:
        if "vulnerability" in self._cached:
            return self._cached["vulnerability"]

        per_class = []
        for cid in self.class_ids:
            area_a   = float(np.sum(self.a == cid)) * self.px_ha
            loss_px  = int(np.sum((self.a == cid) & (self.b != cid)))
            loss_ha  = loss_px * self.px_ha

            # Loss rate (fraction lost per year)
            loss_rate = (loss_ha / area_a / self.n_years) if area_a > 0 else 0.0

            # Gain to loss ratio
            gain_px = int(np.sum((self.a != cid) & (self.b == cid)))
            gain_ha = gain_px * self.px_ha
            gl_ratio = gain_ha / loss_ha if loss_ha > 0 else (None if gain_ha == 0 else np.inf)

            # Vulnerability index: 0 = stable, 1 = fully lost
            vuln = min(1.0, loss_rate * self.n_years)

            per_class.append({
                "class_name":     self.classes[cid]["name"],
                "color":          self.classes[cid]["color"],
                "loss_ha":        round(loss_ha, 2),
                "loss_rate_pct_yr": round(loss_rate * 100, 3),
                "gain_loss_ratio": round(float(gl_ratio), 3) if gl_ratio is not None and not np.isinf(gl_ratio) else gl_ratio,
                "vulnerability_index": round(vuln, 4),
                "risk_level": (
                    "Critical" if vuln > 0.5 else
                    "High"     if vuln > 0.3 else
                    "Moderate" if vuln > 0.1 else
                    "Low"
                ),
            })

        per_class.sort(key=lambda x: x["vulnerability_index"], reverse=True)

        result = {
            "per_class": per_class,
            "most_vulnerable": per_class[0]["class_name"] if per_class else None,
        }
        self._cached["vulnerability"] = result
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_matrix(self):
        n   = self.n
        m   = np.zeros((n, n), dtype=np.int64)
        valid = (self.a > 0) & (self.b > 0)
        a_v, b_v = self.a[valid], self.b[valid]
        for i, from_id in enumerate(self.class_ids):
            mask = (a_v == from_id)
            b_sub = b_v[mask]
            for j, to_id in enumerate(self.class_ids):
                m[i, j] = int(np.sum(b_sub == to_id))
        self._matrix = m

    def _compute_prob_matrix(self):
        m   = self._matrix.astype(float)
        row_sums = m.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(row_sums > 0, m / row_sums, 0.0)
        self._prob_matrix = P

    def _subsample(self, arr: np.ndarray, max_size: int = 256) -> np.ndarray:
        H, W = arr.shape
        if H <= max_size and W <= max_size:
            return arr
        step_h = max(H // max_size, 1)
        step_w = max(W // max_size, 1)
        return arr[::step_h, ::step_w]
