"""
Export Handler
Produces: PDF statistical report, CSV data bundle, GeoTIFF change map.
"""

import io, csv, tempfile, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ExportHandler:
    def __init__(self, analyzer, stats, year_a: int, year_b: int):
        self.az     = analyzer
        self.st     = stats
        self.year_a = year_a
        self.year_b = year_b

    # ── CSV ───────────────────────────────────────────────────────────────────

    def to_csv(self) -> str:
        buf = io.StringIO()
        w   = csv.writer(buf)

        # Section 1: Summary
        s = self.az.summary()
        w.writerow(["=== SUMMARY ==="])
        w.writerow(["Metric", "Value"])
        for k, v in s.items():
            w.writerow([k, v])
        w.writerow([])

        # Section 2: Area stats
        w.writerow(["=== AREA STATISTICS (hectares) ==="])
        w.writerow(["Class", f"Area {self.year_a} (ha)", f"Area {self.year_b} (ha)", "Change (ha)"])
        for row in self.az.area_stats():
            w.writerow([row["class_name"], row["area_a_ha"], row["area_b_ha"], row["change_ha"]])
        w.writerow([])

        # Section 3: Change matrix
        w.writerow(["=== CHANGE MATRIX (ha) ==="])
        cm = self.st.change_matrix_result()
        header = ["FROM \\ TO"] + cm["labels"]
        w.writerow(header)
        for row in cm["matrix"]:
            w.writerow([row["from_name"]] + [v["ha"] for v in row["values"]])
        w.writerow([])

        # Section 4: Transition probabilities
        w.writerow(["=== TRANSITION PROBABILITY MATRIX ==="])
        mk = self.st.markov_result()
        w.writerow(["FROM \\ TO"] + mk["labels"])
        for row in mk["prob_rows"]:
            w.writerow([row["from_name"]] + [p["probability"] for p in row["probs"]])
        w.writerow([])

        # Section 5: Accuracy
        acc = self.st.accuracy_result()
        w.writerow(["=== ACCURACY METRICS ==="])
        w.writerow(["Overall Accuracy",  acc.get("overall_accuracy")])
        w.writerow(["Cohen's Kappa",     acc.get("kappa")])
        w.writerow(["Kappa Interpretation", acc.get("kappa_interpretation")])
        w.writerow([])
        w.writerow(["Class", "Producer's Accuracy", "User's Accuracy", "F1 Score"])
        for pc in acc.get("per_class", []):
            w.writerow([pc["class_name"], pc["producers_accuracy"],
                        pc["users_accuracy"], pc["f1_score"]])
        w.writerow([])

        # Section 6: Pontius
        po = self.st.pontius_result()
        w.writerow(["=== PONTIUS DECOMPOSITION (ha) ==="])
        w.writerow(["Class", "Persistence", "Gain", "Loss", "Net Change", "Swap", "Total"])
        for pc in po["per_class"]:
            w.writerow([pc["class_name"], pc["persistence_ha"], pc["gain_ha"],
                        pc["loss_ha"], pc["net_change_ha"], pc["swap_change_ha"],
                        pc["total_change_ha"]])
        w.writerow([])

        # Section 7: Rates
        roc = self.st.rate_of_change_result()
        w.writerow(["=== ANNUAL RATE OF CHANGE ==="])
        w.writerow(["Formula", roc["formula"]])
        w.writerow(["Class", f"Area {self.year_a} (ha)", f"Area {self.year_b} (ha)",
                    "Change %", "Annual Rate %", "Half-life (yrs)", "Doubling (yrs)"])
        for pc in roc["per_class"]:
            w.writerow([pc["class_name"], pc["area_a_ha"], pc["area_b_ha"],
                        pc["change_pct"], pc["annual_rate_pct"],
                        pc["half_life_years"], pc["doubling_time_years"]])
        w.writerow([])

        # Section 8: Vulnerability
        vuln = self.st.vulnerability_result()
        w.writerow(["=== VULNERABILITY INDEX ==="])
        w.writerow(["Class", "Loss (ha)", "Loss Rate %/yr", "G/L Ratio", "Index", "Risk"])
        for pc in vuln["per_class"]:
            w.writerow([pc["class_name"], pc["loss_ha"], pc["loss_rate_pct_yr"],
                        pc["gain_loss_ratio"], pc["vulnerability_index"], pc["risk_level"]])

        return buf.getvalue()

    # ── PDF ───────────────────────────────────────────────────────────────────

    def to_pdf(self) -> bytes:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors as rl_colors
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                PageBreak, HRFlowable, Image as RLImage
            )
        except ImportError:
            raise RuntimeError("reportlab is required for PDF export.")

        buf  = io.BytesIO()
        doc  = SimpleDocTemplate(buf, pagesize=A4,
                                  leftMargin=2*cm, rightMargin=2*cm,
                                  topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()

        # Custom styles
        H1 = ParagraphStyle("H1", parent=styles["Heading1"],
                             textColor=rl_colors.HexColor("#2d6a4f"),
                             fontSize=16, spaceAfter=6)
        H2 = ParagraphStyle("H2", parent=styles["Heading2"],
                             textColor=rl_colors.HexColor("#40916c"),
                             fontSize=12, spaceAfter=4)
        BODY = ParagraphStyle("BODY", parent=styles["Normal"],
                               fontSize=9, leading=14)
        MONO = ParagraphStyle("MONO", parent=styles["Code"],
                               fontSize=8, leading=12,
                               backColor=rl_colors.HexColor("#f0f4ee"))
        CAPTION = ParagraphStyle("CAPTION", parent=styles["Normal"],
                                  fontSize=8, textColor=rl_colors.HexColor("#5a7060"),
                                  spaceBefore=2)

        def tbl_style(header_color="#2d6a4f"):
            return TableStyle([
                ("BACKGROUND",  (0,0), (-1,0), rl_colors.HexColor(header_color)),
                ("TEXTCOLOR",   (0,0), (-1,0), rl_colors.white),
                ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",    (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS", (0,1), (-1,-1),
                 [rl_colors.HexColor("#f4f9f4"), rl_colors.white]),
                ("GRID",        (0,0), (-1,-1), 0.3, rl_colors.HexColor("#ccd8c4")),
                ("TOPPADDING",  (0,0), (-1,-1), 3),
                ("BOTTOMPADDING",(0,0), (-1,-1), 3),
                ("LEFTPADDING", (0,0), (-1,-1), 5),
            ])

        story = []

        # ── Title page ──
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph("Land Use Change Analysis Report", H1))
        story.append(HRFlowable(width="100%", thickness=1.5,
                                 color=rl_colors.HexColor("#2d6a4f")))
        story.append(Spacer(1, 0.3*cm))
        s = self.az.summary()
        meta_data = [
            ["Analysis Period", f"{self.year_a} → {self.year_b} ({self.year_b-self.year_a} years)"],
            ["Total Area", f"{s['total_area_ha']:,.2f} ha"],
            ["Changed Area", f"{s['changed_area_ha']:,.2f} ha ({s['change_pct']}%)"],
            ["Raster Dimensions", f"{s['raster_shape'][0]} × {s['raster_shape'][1]} pixels"],
            ["Pixel Resolution", f"30 m × 30 m ({s['pixel_area_ha']} ha/px)"],
            ["Land Cover Classes", str(s['n_classes'])],
        ]
        t = Table(meta_data, colWidths=[5*cm, 10*cm])
        t.setStyle(tbl_style("#40916c"))
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        # ── Section 1: Area Statistics ──
        story.append(Paragraph("1. Area Statistics", H1))
        area = self.az.area_stats()
        hdr  = [f"Class", f"Area {self.year_a} (ha)", f"Area {self.year_b} (ha)",
                "Change (ha)", "Change (%)"]
        rows = [hdr] + [
            [r["class_name"], f"{r['area_a_ha']:,.2f}", f"{r['area_b_ha']:,.2f}",
             f"{r['change_ha']:+,.2f}",
             f"{round((r['change_ha']/r['area_a_ha']*100),1) if r['area_a_ha']>0 else 'N/A'}%"]
            for r in area
        ]
        t = Table(rows, colWidths=[4*cm, 3*cm, 3*cm, 3*cm, 3*cm])
        t.setStyle(tbl_style())
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        # ── Section 2: Change Matrix ──
        story.append(Paragraph("2. Change Matrix (ha)", H1))
        story.append(Paragraph(
            "Rows represent Year A classes; columns represent Year B classes. "
            "Diagonal entries show unchanged areas.", BODY))
        story.append(Spacer(1, 0.2*cm))
        cm_data = self.st.change_matrix_result()
        labels  = cm_data["labels"]
        hdr_row = ["FROM \\ TO"] + labels
        data_rows = [hdr_row]
        for row in cm_data["matrix"]:
            data_rows.append([row["from_name"]] + [f"{v['ha']:.1f}" for v in row["values"]])
        col_w = [3.5*cm] + [1.8*cm] * len(labels)
        t = Table(data_rows, colWidths=col_w)
        t.setStyle(tbl_style())
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        # ── Section 3: Markov Chain ──
        story.append(PageBreak())
        story.append(Paragraph("3. Markov Chain Analysis", H1))
        mk = self.st.markov_result()
        story.append(Paragraph(
            f"Steady-state interpretation: {mk['interpretation']}", BODY))
        if mk.get("mixing_time_years"):
            story.append(Paragraph(
                f"Estimated mixing time: {mk['mixing_time_years']} years "
                "(time for landscape to reach steady-state from any initial condition).", BODY))
        story.append(Spacer(1, 0.2*cm))

        # Transition probability table
        hdr_row = ["FROM \\ TO"] + mk["labels"]
        data_rows = [hdr_row]
        for row in mk["prob_rows"]:
            data_rows.append([row["from_name"]] + [f"{p['probability']:.3f}" for p in row["probs"]])
        col_w = [3.5*cm] + [1.8*cm] * len(mk["labels"])
        t = Table(data_rows, colWidths=col_w)
        t.setStyle(tbl_style())
        story.append(t)
        story.append(Spacer(1, 0.4*cm))

        # Steady-state vs predicted
        story.append(Paragraph("Steady-state vs Current vs Predicted Distributions", H2))
        ss_rows = [["Class", "Current (%)", f"Predicted {self.year_b + (self.year_b-self.year_a)} (%)", "Steady-State (%)"]]
        init = mk.get("predicted_dist", [])
        ss   = mk.get("steady_state", [])
        for i, lbl in enumerate(mk["labels"]):
            area_b_pct = round(float(np.sum(self.az.b == self.az.class_ids[i] if hasattr(self.az, 'class_ids') else 0))
                                / self.az.b.size * 100, 1) if i < len(mk["labels"]) else 0
            ss_rows.append([lbl,
                            f"{round(init[i]*100,1) if init else '—'}%",
                            f"{round(init[i]*100,1) if init else '—'}%",
                            f"{round(ss[i]*100,1) if ss else '—'}%"])
        t = Table(ss_rows, colWidths=[4*cm, 3.5*cm, 3.5*cm, 3.5*cm])
        t.setStyle(tbl_style())
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        # ── Section 4: Accuracy / Kappa ──
        story.append(PageBreak())
        story.append(Paragraph("4. Accuracy Assessment", H1))
        acc = self.st.accuracy_result()
        story.append(Paragraph(
            f"Overall Accuracy: {round(acc.get('overall_accuracy',0)*100,2)}%  |  "
            f"Cohen's Kappa (κ): {acc.get('kappa','N/A')}  |  "
            f"{acc.get('kappa_interpretation','')}", BODY))
        story.append(Spacer(1, 0.2*cm))
        pc_rows = [["Class", "Producer's Acc.", "User's Acc.", "F1 Score"]]
        for pc in acc.get("per_class", []):
            pc_rows.append([pc["class_name"],
                            f"{round(pc['producers_accuracy']*100,1)}%",
                            f"{round(pc['users_accuracy']*100,1)}%",
                            f"{pc['f1_score']:.3f}"])
        t = Table(pc_rows, colWidths=[4*cm, 3.5*cm, 3.5*cm, 3.5*cm])
        t.setStyle(tbl_style())
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        # ── Section 5: Spatial / Moran's I ──
        story.append(Paragraph("5. Spatial Autocorrelation — Moran's I", H1))
        mi = self.st.morans_i_result()
        story.append(Paragraph(
            f"Moran's I = {mi.get('morans_i','N/A')}  |  "
            f"Z-score = {mi.get('z_score','N/A')}  |  "
            f"p-value = {mi.get('p_value','N/A')}  |  "
            f"Significant = {mi.get('significant','N/A')}", BODY))
        story.append(Paragraph(mi.get("interpretation",""), BODY))
        story.append(Spacer(1, 0.5*cm))

        # ── Section 6: Pontius ──
        story.append(PageBreak())
        story.append(Paragraph("6. Pontius Change Decomposition", H1))
        story.append(Paragraph(
            "Total change is decomposed into Net change (systematic directional shift) "
            "and Swap change (reciprocal exchanges between classes).", BODY))
        po = self.st.pontius_result()
        story.append(Paragraph(po.get("interpretation",""), BODY))
        story.append(Spacer(1, 0.2*cm))
        po_rows = [["Class", "Persistence (ha)", "Gain (ha)", "Loss (ha)",
                    "Net (ha)", "Swap (ha)", "Total (ha)"]]
        for pc in po["per_class"]:
            po_rows.append([pc["class_name"], f"{pc['persistence_ha']:,.1f}",
                            f"{pc['gain_ha']:,.1f}", f"{pc['loss_ha']:,.1f}",
                            f"{pc['net_change_ha']:,.1f}", f"{pc['swap_change_ha']:,.1f}",
                            f"{pc['total_change_ha']:,.1f}"])
        col_w = [3*cm] + [2.5*cm]*6
        t = Table(po_rows, colWidths=col_w)
        t.setStyle(tbl_style())
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        # ── Section 7: Rates & Vulnerability ──
        story.append(Paragraph("7. Annual Rate of Change & Vulnerability Index", H1))
        story.append(Paragraph(
            f"Formula: {self.st.rate_of_change_result().get('formula','')}", MONO))
        story.append(Spacer(1, 0.2*cm))
        roc = self.st.rate_of_change_result()
        vuln = {pc["class_name"]: pc for pc in self.st.vulnerability_result()["per_class"]}
        rv_rows = [["Class", "Annual Rate %", "Half-life (yr)", "Vuln. Index", "Risk"]]
        for pc in roc["per_class"]:
            v = vuln.get(pc["class_name"], {})
            rv_rows.append([pc["class_name"],
                            f"{pc['annual_rate_pct']:+.3f}%" if pc['annual_rate_pct'] else "—",
                            str(pc["half_life_years"] or "—"),
                            str(v.get("vulnerability_index","—")),
                            v.get("risk_level","—")])
        t = Table(rv_rows, colWidths=[4*cm, 3*cm, 3*cm, 3*cm, 3*cm])
        t.setStyle(tbl_style())
        story.append(t)

        # ── Section 8: Chi-Square ──
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("8. Chi-Square Test of Transition Independence", H1))
        chi = self.st.chi_square_result()
        story.append(Paragraph(chi.get("interpretation",""), BODY))
        story.append(Paragraph(
            f"χ²({chi.get('degrees_of_freedom','?')}) = {chi.get('chi2_statistic','?')}  |  "
            f"Cramér's V = {chi.get('cramers_v','?')} ({chi.get('effect_size','?')} effect)  |  "
            f"p = {chi.get('p_value','?')}", BODY))

        # ── Section 9: Information Theory ──
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("9. Information-Theoretic Analysis", H1))
        info = self.st.information_result()
        story.append(Paragraph(
            f"Shannon Entropy: H({self.year_a}) = {info.get('shannon_entropy_a','?')} bits  |  "
            f"H({self.year_b}) = {info.get('shannon_entropy_b','?')} bits  |  "
            f"ΔH = {info.get('delta_entropy','?')} bits", BODY))
        story.append(Paragraph(
            f"KL Divergence = {info.get('kl_divergence','?')} nats  |  "
            f"Jensen-Shannon Divergence = {info.get('jensen_shannon_div','?')}", BODY))
        story.append(Paragraph(info.get("interpretation",""), BODY))

        doc.build(story)
        buf.seek(0)
        return buf.read()

    # ── GeoTIFF ───────────────────────────────────────────────────────────────

    def to_geotiff(self) -> bytes:
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            raise RuntimeError("rasterio is required for GeoTIFF export.")

        change_map = np.where(
            self.az.a != self.az.b, self.az.b, 0
        ).astype(np.int32)

        buf = io.BytesIO()
        meta = self.az.meta.copy()
        meta.update(dtype=rasterio.int32, count=1, driver="GTiff")

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmp = f.name

        try:
            with rasterio.open(tmp, "w", **meta) as dst:
                dst.write(change_map[np.newaxis])
                dst.update_tags(
                    DESCRIPTION="Land cover change map: 0=unchanged, value=new class ID",
                    YEAR_A=str(self.year_a),
                    YEAR_B=str(self.year_b),
                )
            with open(tmp, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp)
