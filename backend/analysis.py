"""
Change Analyzer — map computation and matplotlib rendering.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import io

PIXEL_AREA_HA = 0.09


class ChangeAnalyzer:
    def __init__(self, arr_a: np.ndarray, arr_b: np.ndarray,
                 meta: dict, classes: dict,
                 pixel_area_ha: float = PIXEL_AREA_HA):
        self.a          = arr_a.astype(np.int32)
        self.b          = arr_b.astype(np.int32)
        self.meta       = meta
        self.classes    = classes
        self.px_ha      = pixel_area_ha
        self.class_ids  = sorted(k for k in classes if k != 0)
        self._change    = None

    def run_all(self):
        self._change = np.where(self.a != self.b, self.b, 0).astype(np.int32)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        total_px   = self.a.size
        changed_px = int(np.sum(self.a != self.b))
        return {
            "total_area_ha":     round(total_px   * self.px_ha, 2),
            "changed_area_ha":   round(changed_px * self.px_ha, 2),
            "unchanged_area_ha": round((total_px - changed_px) * self.px_ha, 2),
            "change_pct":        round(changed_px / total_px * 100, 2),
            "raster_shape":      list(self.a.shape),
            "n_classes":         len(self.class_ids),
            "pixel_area_ha":     self.px_ha,
        }

    def area_stats(self) -> list:
        return [{
            "class_id":    cid,
            "class_name":  self.classes[cid]["name"],
            "color":       self.classes[cid]["color"],
            "area_a_ha":   round(int(np.sum(self.a == cid)) * self.px_ha, 2),
            "area_b_ha":   round(int(np.sum(self.b == cid)) * self.px_ha, 2),
            "change_ha":   round((int(np.sum(self.b == cid)) - int(np.sum(self.a == cid))) * self.px_ha, 2),
        } for cid in self.class_ids]

    def change_map_stats(self) -> dict:
        changed   = (self.a != self.b)
        gain_px   = int(np.sum((self.a == 0) & (self.b > 0)))
        loss_px   = int(np.sum((self.a > 0) & (self.b == 0)))
        return {
            "total_changed_px": int(changed.sum()),
            "gain_px":          gain_px,
            "loss_px":          loss_px,
            "gain_ha":          round(gain_px * self.px_ha, 2),
            "loss_ha":          round(loss_px * self.px_ha, 2),
        }

    # ── Map rendering ─────────────────────────────────────────────────────────

    def render_map(self, map_type: str) -> bytes:
        dispatch = {
            "raster_2010": lambda: self._render_classified(self.a, "Year A"),
            "raster_2020": lambda: self._render_classified(self.b, "Year B"),
            "change":      self._render_change,
            "gain_loss":   self._render_gain_loss,
        }
        if map_type not in dispatch:
            raise ValueError(f"Unknown map type: {map_type}")
        fig = dispatch[map_type]()
        return self._fig_to_bytes(fig)

    def _cmap_norm(self):
        all_ids = [0] + self.class_ids
        colors  = ["#e8e4d9"] + [self.classes[cid]["color"] for cid in self.class_ids]
        cmap    = ListedColormap(colors)
        bounds  = [i - 0.5 for i in range(len(all_ids) + 1)]
        norm    = BoundaryNorm(bounds, cmap.N)
        return cmap, norm

    def _render_classified(self, arr, title_suffix) -> plt.Figure:
        cmap, norm = self._cmap_norm()
        fig, ax = plt.subplots(figsize=(7, 6), facecolor="#f4f1e8")
        ax.set_facecolor("#e8e4d9")
        ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"Land Cover — {title_suffix}",
                     color="#1b2d20", fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")
        patches = [mpatches.Patch(color=self.classes[cid]["color"],
                                  label=self.classes[cid]["name"])
                   for cid in self.class_ids]
        ax.legend(handles=patches, loc="lower right",
                  framealpha=0.9, facecolor="#f4f1e8",
                  labelcolor="#1b2d20", fontsize=8, edgecolor="#ccd8c4")
        fig.tight_layout()
        return fig

    def _render_change(self) -> plt.Figure:
        cmap, norm = self._cmap_norm()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="#f4f1e8")
        titles = ["Year A", "Year B", "Change Map (→ new class)"]
        arrs   = [self.a, self.b, self._change]
        for ax, arr, title in zip(axes, arrs, titles):
            ax.set_facecolor("#e8e4d9")
            ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_title(title, color="#1b2d20", fontsize=11, fontweight="bold")
            ax.axis("off")
        patches = [mpatches.Patch(color=self.classes[cid]["color"],
                                  label=self.classes[cid]["name"])
                   for cid in self.class_ids]
        fig.legend(handles=patches, loc="lower center", ncol=len(self.class_ids),
                   framealpha=0.9, facecolor="#f4f1e8",
                   labelcolor="#1b2d20", fontsize=8)
        fig.suptitle("Land Cover Change Detection",
                     color="#1b2d20", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        return fig

    def _render_gain_loss(self) -> plt.Figure:
        gl = np.zeros(self.a.shape, dtype=np.int8)
        gl[(self.a != 0) & (self.a != self.b)] = 1   # loss
        gl[(self.b != 0) & (self.a != self.b)] = 2   # gain (overwrites if both non-zero)

        cmap_gl = ListedColormap(["#e8e4d9", "#bc4749", "#2d6a4f"])
        norm_gl = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)

        fig, ax = plt.subplots(figsize=(7, 6), facecolor="#f4f1e8")
        ax.set_facecolor("#e8e4d9")
        ax.imshow(gl, cmap=cmap_gl, norm=norm_gl, interpolation="nearest")
        ax.set_title("Gain / Loss Map", color="#1b2d20",
                     fontsize=13, fontweight="bold", pad=10)
        ax.axis("off")
        legend_items = [
            mpatches.Patch(color="#e8e4d9", label="No Change"),
            mpatches.Patch(color="#bc4749", label="Loss"),
            mpatches.Patch(color="#2d6a4f", label="Gain"),
        ]
        ax.legend(handles=legend_items, loc="lower right",
                  framealpha=0.9, facecolor="#f4f1e8",
                  labelcolor="#1b2d20", fontsize=9, edgecolor="#ccd8c4")
        fig.tight_layout()
        return fig

    @staticmethod
    def _fig_to_bytes(fig) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()
