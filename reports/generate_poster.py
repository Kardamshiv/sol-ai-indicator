import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_poster():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # Background
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = RGBColor(245, 247, 250)
    bg.line.color.rgb = RGBColor(245, 247, 250)

    # -------------------------------------------------------------
    # 1. Header Banner
    # -------------------------------------------------------------
    header_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.2), Inches(0.12), Inches(12.933), Inches(1.05))
    header_box.fill.solid()
    header_box.fill.fore_color.rgb = RGBColor(255, 255, 255)
    header_box.line.color.rgb = RGBColor(200, 210, 225)
    header_box.line.width = Pt(1.5)

    # IIT Patna Logo (Left)
    iitp_logo_path = "reports/assets/iit_patna_logo.png"
    if os.path.exists(iitp_logo_path):
        slide.shapes.add_picture(iitp_logo_path, Inches(0.35), Inches(0.16), width=Inches(1.0), height=Inches(0.96))

    # IMPACT Lab Logo (Right)
    impact_logo_path = "reports/assets/impact_lab_logo.png"
    if os.path.exists(impact_logo_path):
        slide.shapes.add_picture(impact_logo_path, Inches(11.65), Inches(0.24), width=Inches(1.35), height=Inches(0.78))

    # Header Text (Center)
    title_box = slide.shapes.add_textbox(Inches(1.45), Inches(0.14), Inches(10.1), Inches(1.0))
    tf = title_box.text_frame
    tf.word_wrap = True
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)

    p0 = tf.paragraphs[0]
    p0.text = "Leakage-Free Machine Learning Directional Predictor & Calibrated Indicator for Solana Trading"
    p0.font.name = "Times New Roman"
    p0.font.size = Pt(17)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(10, 37, 64)
    p0.alignment = PP_ALIGN.CENTER

    p1 = tf.add_paragraph()
    p1.text = "Kardam Shiv (Roll: 2201CS01) | Supervisor: Dr. Research Supervisor"
    p1.font.name = "Times New Roman"
    p1.font.size = Pt(10.5)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(0, 82, 204)
    p1.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = "IMPACT Lab, Department of Computer Science & Engineering, Indian Institute of Technology Patna"
    p2.font.name = "Times New Roman"
    p2.font.size = Pt(9.5)
    p2.font.color.rgb = RGBColor(80, 95, 115)
    p2.alignment = PP_ALIGN.CENTER

    # Helper function to create section cards
    def add_card(left, top, width, height, title, subtitle=None):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = RGBColor(255, 255, 255)
        card.line.color.rgb = RGBColor(215, 225, 235)
        card.line.width = Pt(1.2)

        # Section Header Banner
        tb = slide.shapes.add_textbox(left + Inches(0.1), top + Inches(0.06), width - Inches(0.2), Inches(0.32))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Times New Roman"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = RGBColor(15, 45, 90)
        return card

    # -------------------------------------------------------------
    # 2. Left Column (Width: 3.75 in)
    # -------------------------------------------------------------
    col1_l = Inches(0.2)
    col1_w = Inches(3.75)

    # Card 1: Abstract & Problem Statement
    add_card(col1_l, Inches(1.25), col1_w, Inches(5.35), "1. Abstract & Problem Statement")

    tb_left = slide.shapes.add_textbox(col1_l + Inches(0.12), Inches(1.62), col1_w - Inches(0.24), Inches(4.88))
    tf_l = tb_left.text_frame
    tf_l.word_wrap = True
    tf_l.margin_left = tf_l.margin_top = tf_l.margin_right = tf_l.margin_bottom = 0

    p = tf_l.paragraphs[0]
    p.text = "Abstract"
    p.font.name = "Times New Roman"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = RGBColor(180, 83, 9)

    p = tf_l.add_paragraph()
    p.text = "High-frequency cryptocurrency markets exhibit severe non-stationarity, microstructure noise, and regime shifts. Existing technical indicators suffer from lookahead bias and indicator repainting, producing inflated paper returns that collapse in live execution [1]. We present an end-to-end leakage-free machine learning framework for Solana (SOLUSDT) 15-minute directional prediction over a 1-hour forward horizon (4 bars). Evaluated across 70,079 Binance klines (2024–2026), our LightGBM model achieves 54.7% Purged Cross-Validation accuracy and 58.3% test accuracy at high confidence, outperforming baseline models with 0 repainting."
    p.font.name = "Times New Roman"
    p.font.size = Pt(8.5)
    p.font.color.rgb = RGBColor(40, 50, 65)

    p = tf_l.add_paragraph()
    p.text = "\nIntroduction & Motivation"
    p.font.name = "Times New Roman"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = RGBColor(180, 83, 9)

    p = tf_l.add_paragraph()
    p.text = "• Cryptocurrency trading requires robust probabilistic direction estimates under significant volatility [1].\n• Traditional momentum indicators (RSI, MACD) generate high false-positive rates and repaint historical signals.\n• Machine learning models often suffer from catastrophic leakage via rolling scalers or overlapping forward return targets [2]."
    p.font.name = "Times New Roman"
    p.font.size = Pt(8.3)
    p.font.color.rgb = RGBColor(40, 50, 65)

    p = tf_l.add_paragraph()
    p.text = "\nProblem Statement"
    p.font.name = "Times New Roman"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = RGBColor(180, 83, 9)

    p = tf_l.add_paragraph()
    p.text = "Develop a strictly leak-free directional predictor and production streaming architecture that forecasts 1h SOLUSDT price direction with calibrated probabilities, statistically significant edge over random walk, and zero live repainting."
    p.font.name = "Times New Roman"
    p.font.size = Pt(8.5)
    p.font.color.rgb = RGBColor(40, 50, 65)

    # -------------------------------------------------------------
    # 3. Middle Column (Width: 5.25 in)
    # -------------------------------------------------------------
    col2_l = Inches(4.05)
    col2_w = Inches(5.25)

    add_card(col2_l, Inches(1.25), col2_w, Inches(5.35), "2. Methodology & System Architecture")

    # Key Contributions Box
    tb_mid = slide.shapes.add_textbox(col2_l + Inches(0.12), Inches(1.6), col2_w - Inches(0.24), Inches(1.15))
    tf_m = tb_mid.text_frame
    tf_m.word_wrap = True
    tf_m.margin_left = tf_m.margin_top = tf_m.margin_right = tf_m.margin_bottom = 0

    p = tf_m.paragraphs[0]
    p.text = "Key Contributions"
    p.font.name = "Times New Roman"
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = RGBColor(15, 45, 90)

    p = tf_m.add_paragraph()
    p.text = "1. Strict Point-in-Time Pipeline: 33 features computed exclusively on closed bars (t <= T).\n2. 5-Fold Purged & Embargoed CV: Prevents target overlap leakage across train/validation splits.\n3. Post-Hoc Platt Calibration: Corrects tree probability distortion, optimizing Brier score to 0.244.\n4. Real-Time Streaming Indicator: Binance WebSocket + FastAPI + TradingView lightweight-charts."
    p.font.name = "Times New Roman"
    p.font.size = Pt(8.2)
    p.font.color.rgb = RGBColor(30, 40, 55)

    # Figure 1: Architecture Image
    arch_img_path = "reports/assets/architecture_diagram.png"
    if os.path.exists(arch_img_path):
        slide.shapes.add_picture(arch_img_path, col2_l + Inches(0.12), Inches(2.78), width=col2_w - Inches(0.24), height=Inches(1.85))

    # Caption
    tb_cap = slide.shapes.add_textbox(col2_l + Inches(0.12), Inches(4.65), col2_w - Inches(0.24), Inches(0.22))
    p = tb_cap.text_frame.paragraphs[0]
    p.text = "Figure 1: End-to-End Leak-Free Machine Learning Architecture & Real-Time Production Flow"
    p.font.name = "Times New Roman"
    p.font.size = Pt(8.0)
    p.font.bold = True
    p.font.color.rgb = RGBColor(80, 95, 115)
    p.alignment = PP_ALIGN.CENTER

    # Algorithm 1 Box
    algo_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, col2_l + Inches(0.12), Inches(4.90), col2_w - Inches(0.24), Inches(1.62))
    algo_box.fill.solid()
    algo_box.fill.fore_color.rgb = RGBColor(248, 250, 252)
    algo_box.line.color.rgb = RGBColor(210, 220, 235)
    algo_box.line.width = Pt(1.0)

    tb_alg = slide.shapes.add_textbox(col2_l + Inches(0.18), Inches(4.92), col2_w - Inches(0.36), Inches(1.58))
    tf_a = tb_alg.text_frame
    tf_a.word_wrap = True
    tf_a.margin_left = tf_a.margin_top = tf_a.margin_right = tf_a.margin_bottom = 0

    p = tf_a.paragraphs[0]
    p.text = "Algorithm 1: Leak-Free Directional Trading Signal Generation"
    p.font.name = "Times New Roman"
    p.font.size = Pt(8.5)
    p.font.bold = True
    p.font.color.rgb = RGBColor(15, 45, 90)

    p = tf_a.add_paragraph()
    p.text = "Require: 15m OHLCV stream S = {(O_t, H_t, L_t, C_t, V_t)}; Horizon h = 4; Threshold τ = 0.53\nEnsure: Calibrated signal y_hat ∈ {BUY, NEUTRAL, SELL} with confidence P_t\n1. At candle close t, compute 33 causal features X_t using only past bars.\n2. Label target: y_t = 1 if C_{t+4} > C_t else 0. Split with 5-Fold Purged CV (embargo = 4 bars).\n3. Train LightGBM model f* and fit Platt calibrator P(y=1|f*) = 1 / (1 + exp(A·f* + B)).\n4. For live bar t: P_t = Calibrate(f*(X_t)). If P_t >= τ -> BUY; if P_t <= 1 - τ -> SELL; else NEUTRAL.\n5. Broadcast deterministic signal to WebSocket clients with immutable timestamp (0 repainting)."
    p.font.name = "Courier New"
    p.font.size = Pt(7.0)
    p.font.color.rgb = RGBColor(25, 35, 50)

    # -------------------------------------------------------------
    # 4. Right Column (Width: 3.75 in)
    # -------------------------------------------------------------
    col3_l = Inches(9.4)
    col3_w = Inches(3.73)

    add_card(col3_l, Inches(1.25), col3_w, Inches(5.35), "3. Results & Performance Discussion")

    # Table 1: Model Comparison
    tb_t1 = slide.shapes.add_textbox(col3_l + Inches(0.1), Inches(1.58), col3_w - Inches(0.2), Inches(0.22))
    p = tb_t1.text_frame.paragraphs[0]
    p.text = "Table 1: Benchmark Comparison on 70,079 15m Candles"
    p.font.name = "Times New Roman"
    p.font.size = Pt(8.5)
    p.font.bold = True
    p.font.color.rgb = RGBColor(15, 45, 90)

    # Insert Table
    table_shape = slide.shapes.add_table(5, 5, col3_l + Inches(0.1), Inches(1.82), col3_w - Inches(0.2), Inches(1.15))
    tbl = table_shape.table
    tbl.columns[0].width = Inches(1.15)
    tbl.columns[1].width = Inches(0.60)
    tbl.columns[2].width = Inches(0.60)
    tbl.columns[3].width = Inches(0.58)
    tbl.columns[4].width = Inches(0.60)

    headers = ["Model / Rule", "CV AUC", "CV Acc", "Brier", "Test Acc"]
    rows = [
        ["Always Up (Maj.)", "0.500", "50.3%", "0.250", "49.9%"],
        ["RSI (30/70 Overb.)", "0.522", "51.4%", "0.251", "51.4%"],
        ["Random Forest", "0.523", "51.5%", "0.251", "56.5%"],
        ["LightGBM (Ours)", "0.525", "54.7%", "0.244", "58.3%*"]
    ]

    for col_idx, h in enumerate(headers):
        cell = tbl.cell(0, col_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(15, 45, 90)
        p = cell.text_frame.paragraphs[0]
        p.font.name = "Times New Roman"
        p.font.size = Pt(7.0)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER

    for row_idx, r_data in enumerate(rows):
        for col_idx, val in enumerate(r_data):
            cell = tbl.cell(row_idx + 1, col_idx)
            cell.text = val
            cell.fill.solid()
            if row_idx == 3:  # LightGBM row highlight
                cell.fill.fore_color.rgb = RGBColor(235, 245, 255)
            else:
                cell.fill.fore_color.rgb = RGBColor(255, 255, 255) if row_idx % 2 == 0 else RGBColor(245, 248, 252)
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Times New Roman"
            p.font.size = Pt(7.0)
            p.font.bold = (row_idx == 3)
            p.font.color.rgb = RGBColor(15, 45, 90) if row_idx == 3 else RGBColor(30, 40, 50)
            p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT

    # Figure 2: Composite Results Image
    comp_img_path = "reports/assets/results_composite.png"
    if os.path.exists(comp_img_path):
        slide.shapes.add_picture(comp_img_path, col3_l + Inches(0.1), Inches(3.05), width=col3_w - Inches(0.2), height=Inches(1.85))

    tb_cap2 = slide.shapes.add_textbox(col3_l + Inches(0.1), Inches(4.92), col3_w - Inches(0.2), Inches(0.22))
    p = tb_cap2.text_frame.paragraphs[0]
    p.text = "Figure 2: Permutation Feature Importance & Walk-Forward Simulated Equity"
    p.font.name = "Times New Roman"
    p.font.size = Pt(7.5)
    p.font.bold = True
    p.font.color.rgb = RGBColor(80, 95, 115)
    p.alignment = PP_ALIGN.CENTER

    # Conclusion & Live Status Box
    tb_c = slide.shapes.add_textbox(col3_l + Inches(0.1), Inches(5.15), col3_w - Inches(0.2), Inches(1.35))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True
    tf_c.margin_left = tf_c.margin_top = tf_c.margin_right = tf_c.margin_bottom = 0

    p = tf_c.paragraphs[0]
    p.text = "Conclusion & Production Deployment"
    p.font.name = "Times New Roman"
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = RGBColor(180, 83, 9)

    p = tf_c.add_paragraph()
    p.text = "• LightGBM achieves 58.3% test accuracy on confident signals, outperforming technical baselines.\n• Platt calibration achieves optimal probability alignment (Brier score 0.244).\n• Fully operational on Render cloud with Binance live WebSocket and interactive What-If backtest."
    p.font.name = "Times New Roman"
    p.font.size = Pt(7.8)
    p.font.color.rgb = RGBColor(40, 50, 65)

    # -------------------------------------------------------------
    # 5. Bottom References Banner
    # -------------------------------------------------------------
    ref_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.2), Inches(6.68), Inches(12.933), Inches(0.70))
    ref_box.fill.solid()
    ref_box.fill.fore_color.rgb = RGBColor(255, 255, 255)
    ref_box.line.color.rgb = RGBColor(215, 225, 235)
    ref_box.line.width = Pt(1.0)

    tb_ref = slide.shapes.add_textbox(Inches(0.3), Inches(6.70), Inches(12.733), Inches(0.65))
    tf_r = tb_ref.text_frame
    tf_r.word_wrap = True
    tf_r.margin_left = tf_r.margin_top = tf_r.margin_right = tf_r.margin_bottom = 0

    p = tf_r.paragraphs[0]
    p.text = "References:"
    p.font.name = "Times New Roman"
    p.font.size = Pt(8.5)
    p.font.bold = True
    p.font.color.rgb = RGBColor(15, 45, 90)

    p = tf_r.add_paragraph()
    p.text = "[1] M. López de Prado, Advances in Financial Machine Learning. Hoboken, NJ: John Wiley & Sons, 2018.\n[2] D. R. Aronson, Evidence-Based Technical Analysis: Applying the Scientific Method and Statistical Inference to Trading Signals. Hoboken, NJ: John Wiley & Sons, 2006.\n[3] G. Ke et al., 'LightGBM: A highly efficient gradient boosting decision tree,' Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 3146-3154, 2017."
    p.font.name = "Times New Roman"
    p.font.size = Pt(7.2)
    p.font.color.rgb = RGBColor(70, 80, 95)

    output_path = "reports/poster_iitp.pptx"
    prs.save(output_path)
    print(f"Saved poster presentation to: {output_path}")

if __name__ == "__main__":
    create_poster()
