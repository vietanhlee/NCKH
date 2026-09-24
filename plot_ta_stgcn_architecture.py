"""
Script: plot_ta_stgcn_architecture.py
Purpose: Generate ultra-compact, horizontal, publication-grade architectural diagrams
         for TA-STGCN (Figure 2 in the paper) with zero overflow and pixel-perfect margins.
Outputs:
  - fig/ta_stgcn_architecture.pdf
  - fig/ta_stgcn_architecture.png (300 DPI)
  - fig/at_stgcn_architecture.pdf (backward compatibility)
  - fig/at_stgcn_architecture.png
"""

import os
import matplotlib
matplotlib.use('Agg')  # Headless backend to avoid GUI issues
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_ta_stgcn_architecture():
    # Typography & Mathtext Configuration
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    # Dimensions: 16.0 x 6.0 inches (ample vertical breathing room)
    fig = plt.figure(figsize=(16.0, 6.0), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 60)
    ax.axis('off')

    # Force bounding box limit anchors so tight crop never clips edges
    ax.plot([0, 160], [0, 60], color='none', alpha=0)

    # Academic Palette
    C_INPUT = "#EBF4F6"
    C_INPUT_BORDER = "#2C5E7A"
    
    C_STGCN = "#E6F0FA"
    C_STGCN_BORDER = "#1E6091"
    
    C_ATTN_MAIN = "#FFF3E0"
    C_ATTN_BORDER = "#E65100"
    
    C_MHA = "#FCE4EC"
    C_MHA_BORDER = "#C2185B"
    
    C_ADDNORM = "#EDE7F6"
    C_ADDNORM_BORDER = "#5E35B1"
    
    C_FFN = "#E8F5E9"
    C_FFN_BORDER = "#2E7D32"
    
    C_CHEB = "#FFFDE7"
    C_CHEB_BORDER = "#F57F17"
    
    C_OUT = "#E8F5E9"
    C_OUT_BORDER = "#1B5E20"
    
    C_TEXT = "#0F172A"
    C_SUBTEXT = "#1E293B"
    C_ARROW = "#1E293B"

    # Helper function for drawing rounded boxes
    def draw_box(x, y, w, h, bg_color, border_color, title, subtitle=None, sub2=None, lw=1.8,
                 title_fs=11.5, sub_fs=9.8, sub2_fs=9.0):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                             linewidth=lw, edgecolor=border_color, facecolor=bg_color,
                             zorder=2)
        ax.add_patch(box)
        
        cx = x + w / 2.0
        if subtitle is None and sub2 is None:
            ax.text(cx, y + h / 2.0, title, ha='center', va='center',
                    fontsize=title_fs, fontweight='bold', color=C_TEXT, zorder=3)
        elif sub2 is None:
            ax.text(cx, y + h * 0.65, title, ha='center', va='center',
                    fontsize=title_fs, fontweight='bold', color=C_TEXT, zorder=3)
            ax.text(cx, y + h * 0.28, subtitle, ha='center', va='center',
                    fontsize=sub_fs, fontweight='bold', color=C_SUBTEXT, zorder=3)
        else:
            ax.text(cx, y + h * 0.74, title, ha='center', va='center',
                    fontsize=title_fs, fontweight='bold', color=C_TEXT, zorder=3)
            ax.text(cx, y + h * 0.44, subtitle, ha='center', va='center',
                    fontsize=sub_fs, fontweight='bold', color=C_SUBTEXT, zorder=3)
            ax.text(cx, y + h * 0.17, sub2, ha='center', va='center',
                    fontsize=sub2_fs, fontweight='bold', color="#334155", zorder=3)

    # Helper function for drawing connecting arrows
    def draw_arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.8):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='-|>', mutation_scale=14,
                                color=color, linewidth=lw, zorder=4)
        ax.add_patch(arrow)

    # Helper function for curved residual bypass arrow
    def draw_residual(x1, y1, x2, y2, bend_y, color="#5E35B1", label="Residual (+)"):
        ax.plot([x1, x1, x2, x2], [y1, bend_y, bend_y, y2 + 0.8],
                color=color, linestyle='--', linewidth=1.5, zorder=3)
        arrow = FancyArrowPatch((x2, y2 + 1.2), (x2, y2),
                                arrowstyle='-|>', mutation_scale=12,
                                color=color, linewidth=1.5, zorder=4)
        ax.add_patch(arrow)
        ax.text((x1 + x2) / 2.0, bend_y + 0.5, label, ha='center', va='bottom',
                fontsize=9.8, color=color, fontweight='bold', zorder=5)

    # =========================================================================
    # PART (a): OVERALL PIPELINE (HORIZONTAL STREAM, Y = 29.0 to 53.0)
    # =========================================================================
    ax.text(2.0, 56.5, "(a) End-to-End Forecasting Architecture (Horizontal Flow)",
            fontsize=13.0, fontweight='bold', color="#0F172A", ha='left', va='center')

    # Container (a) from X = 1.5 to 158.5 (width = 157.0)
    box_a = FancyBboxPatch((1.5, 29.0), 157.0, 25.0, boxstyle="round,pad=0.2",
                           linewidth=1.2, edgecolor="#94A3B8", facecolor="#F8FAFC", zorder=1)
    ax.add_patch(box_a)

    top_y = 32.5
    box_h = 17.5
    cy_a = top_y + box_h / 2.0

    # 1. Input Tensor [3.5, 23.5]
    bx1 = 3.5; bw1 = 20.0
    draw_box(bx1, top_y, bw1, box_h, C_INPUT, C_INPUT_BORDER,
             "Input Tensor",
             r"$\mathbf{X} \in \mathbb{R}^{B \times T_{\mathrm{in}} \times N \times F}$",
             r"$T_{\mathrm{in}}{=}24, N{=}608, F{=}5$",
             title_fs=11.8, sub_fs=10.0, sub2_fs=9.5)

    # Arrow 1 -> 2
    draw_arrow(bx1 + bw1, cy_a, bx1 + bw1 + 3.5, cy_a)

    # 2. ST-Conv Block 1 [27.0, 50.5]
    bx2 = bx1 + bw1 + 3.5; bw2 = 23.5
    draw_box(bx2, top_y, bw2, box_h, C_STGCN, C_STGCN_BORDER,
             "ST-Conv Block 1",
             "Spatio-Temporal Conv",
             r"$C_{\mathrm{in}}{=}5 \rightarrow C_{\mathrm{out}}{=}80$",
             title_fs=11.8, sub_fs=10.2, sub2_fs=9.8)

    # Arrow 2 -> 3
    draw_arrow(bx2 + bw2, cy_a, bx2 + bw2 + 3.5, cy_a)

    # 3. Temporal Self-Attention [54.0, 83.0] (Highlighted)
    bx3 = bx2 + bw2 + 3.5; bw3 = 29.0
    draw_box(bx3, top_y, bw3, box_h, C_ATTN_MAIN, C_ATTN_BORDER,
             "Temporal Self-Attention",
             r"Multi-Head ($h{=}4$) + FFN",
             r"Decoupled $T_{\mathrm{in}} \times T_{\mathrm{in}}$ Context (see (c))",
             lw=2.2, title_fs=11.8, sub_fs=10.2, sub2_fs=9.2)
    
    # Highlight badge
    badge = FancyBboxPatch((bx3 + bw3/2.0 - 8.0, top_y + box_h - 1.0), 16.0, 2.6,
                           boxstyle="round,pad=0.1",
                           linewidth=1.2, edgecolor="#C2410C", facecolor="#FFEDD5", zorder=4)
    ax.add_patch(badge)
    ax.text(bx3 + bw3/2.0, top_y + box_h + 0.3, "PROPOSED ADAPTATION",
            fontsize=8.0, fontweight='bold', color="#9A3412", ha='center', va='center', zorder=5)

    # Arrow 3 -> 4
    draw_arrow(bx3 + bw3, cy_a, bx3 + bw3 + 3.5, cy_a)

    # 4. ST-Conv Block 2 [86.5, 110.0]
    bx4 = bx3 + bw3 + 3.5; bw4 = 23.5
    draw_box(bx4, top_y, bw4, box_h, C_STGCN, C_STGCN_BORDER,
             "ST-Conv Block 2",
             "Spatio-Temporal Conv",
             r"$C_{\mathrm{in}}{=}80 \rightarrow C_{\mathrm{out}}{=}80$",
             title_fs=11.8, sub_fs=10.2, sub2_fs=9.8)

    # Arrow 4 -> 5
    draw_arrow(bx4 + bw4, cy_a, bx4 + bw4 + 3.5, cy_a)

    # 5. Output Projection [113.5, 135.5]
    bx5 = bx4 + bw4 + 3.5; bw5 = 22.0
    draw_box(bx5, top_y, bw5, box_h, C_OUT, C_OUT_BORDER,
             "Output Projection",
             r"1D Conv ($T_{\mathrm{in}} \rightarrow 1$)",
             r"$C{=}80 \rightarrow H \times 2$",
             title_fs=11.8, sub_fs=10.2, sub2_fs=9.8)

    # Arrow 5 -> 6
    draw_arrow(bx5 + bw5, cy_a, bx5 + bw5 + 3.5, cy_a)

    # 6. Forecast Output [139.0, 157.5]
    bx6 = bx5 + bw5 + 3.5; bw6 = 18.5
    draw_box(bx6, top_y, bw6, box_h, C_INPUT, C_INPUT_BORDER,
             "Forecast Output",
             r"$\hat{\mathbf{Y}} \in \mathbb{R}^{B \times H \times N \times 2}$",
             r"$H{=}6$ Steps (30 mins)",
             title_fs=11.8, sub_fs=10.0, sub2_fs=9.5)

    # =========================================================================
    # PART (b) & (c): INTERNAL MODULE DETAILS (Y = 1.5 to 26.0)
    # =========================================================================
    bot_y = 3.5
    sub_h = 13.5
    cy_sub = bot_y + sub_h / 2.0

    # -------------------------------------------------------------------------
    # PART (b): ST-Conv Block Internal Architecture (X = 1.5 to 77.5)
    # -------------------------------------------------------------------------
    ax.text(2.0, 25.4, "(b) ST-Conv Block Internal Architecture",
            fontsize=12.0, fontweight='bold', color="#0F172A", ha='left', va='center')

    box_b = FancyBboxPatch((1.5, 1.5), 76.0, 24.5, boxstyle="round,pad=0.2",
                           linewidth=1.2, edgecolor="#94A3B8", facecolor="#F8FAFC", zorder=1)
    ax.add_patch(box_b)

    # b1: Input [3.0, 11.5]
    sx0 = 3.0; sw0 = 8.5
    draw_box(sx0, bot_y + 0.5, sw0, sub_h - 1.0, C_INPUT, C_INPUT_BORDER, "Input",
             title_fs=11.8)

    draw_arrow(sx0 + sw0, cy_sub, sx0 + sw0 + 1.7, cy_sub)

    # b2: Temporal GLU 1 [13.2, 24.7]
    sx1 = sx0 + sw0 + 1.7; sw1 = 11.5
    draw_box(sx1, bot_y, sw1, sub_h, C_STGCN, C_STGCN_BORDER,
             "Temporal\nGLU", r"$K_t{=}3$",
             title_fs=11.2, sub_fs=10.0)

    draw_arrow(sx1 + sw1, cy_sub, sx1 + sw1 + 1.7, cy_sub)

    # b3: Chebyshev Graph Conv [26.4, 38.9]
    sx2 = sx1 + sw1 + 1.7; sw2 = 12.5
    draw_box(sx2, bot_y, sw2, sub_h, C_CHEB, C_CHEB_BORDER,
             "Chebyshev\nGraph Conv", r"Spectral $K{=}3$",
             title_fs=11.0, sub_fs=9.5)

    draw_arrow(sx2 + sw2, cy_sub, sx2 + sw2 + 1.7, cy_sub)

    # b4: Temporal GLU 2 [40.6, 52.1]
    sx3 = sx2 + sw2 + 1.7; sw3 = 11.5
    draw_box(sx3, bot_y, sw3, sub_h, C_STGCN, C_STGCN_BORDER,
             "Temporal\nGLU", r"$K_t{=}3$",
             title_fs=11.2, sub_fs=10.0)

    draw_arrow(sx3 + sw3, cy_sub, sx3 + sw3 + 1.7, cy_sub)

    # b5: LayerNorm [53.8, 65.3]
    sx4 = sx3 + sw3 + 1.7; sw4 = 11.5
    draw_box(sx4, bot_y, sw4, sub_h, C_ADDNORM, C_ADDNORM_BORDER,
             "LayerNorm\n+ Dropout", r"$p{=}0.25$",
             title_fs=11.0, sub_fs=9.5)

    draw_arrow(sx4 + sw4, cy_sub, sx4 + sw4 + 1.7, cy_sub)

    # b6: Output [67.0, 75.5]
    sx5 = sx4 + sw4 + 1.7; sw5 = 8.5
    draw_box(sx5, bot_y + 0.5, sw5, sub_h - 1.0, C_INPUT, C_INPUT_BORDER, "Output",
             title_fs=11.8)

    # Residual arrow for ST-Conv (from Input to LayerNorm)
    draw_residual(sx0 + sw0/2.0, bot_y + sub_h - 0.5, sx4 + sw4/2.0, bot_y + sub_h, 
                  bend_y=bot_y + sub_h + 3.2, color="#1E6091", label="Residual (+)")

    # -------------------------------------------------------------------------
    # PART (c): Temporal Self-Attention Module Architecture (X = 81.5 to 158.5)
    # -------------------------------------------------------------------------
    ax.text(82.0, 25.4, "(c) Temporal Self-Attention Module Internal Architecture",
            fontsize=12.0, fontweight='bold', color="#0F172A", ha='left', va='center')

    box_c = FancyBboxPatch((81.5, 1.5), 77.0, 24.5, boxstyle="round,pad=0.2",
                           linewidth=1.2, edgecolor="#94A3B8", facecolor="#F8FAFC", zorder=1)
    ax.add_patch(box_c)

    # c1: Input z [83.0, 91.5]
    tx0 = 83.0; tw0 = 8.5
    draw_box(tx0, bot_y + 0.5, tw0, sub_h - 1.0, C_INPUT, C_INPUT_BORDER, r"Input $\mathbf{z}$",
             r"$(BN, T, C)$", title_fs=11.5, sub_fs=9.8)

    draw_arrow(tx0 + tw0, cy_sub, tx0 + tw0 + 1.7, cy_sub)

    # c2: Multi-Head Attention [93.2, 106.2]
    tx1 = tx0 + tw0 + 1.7; tw1 = 13.0
    draw_box(tx1, bot_y, tw1, sub_h, C_MHA, C_MHA_BORDER,
             "Multi-Head\nAttention", r"$h{=}4$ heads",
             title_fs=11.5, sub_fs=10.0)

    draw_arrow(tx1 + tw1, cy_sub, tx1 + tw1 + 1.7, cy_sub)

    # c3: Add & Norm 1 [107.9, 118.9]
    tx2 = tx1 + tw1 + 1.7; tw2 = 11.0
    draw_box(tx2, bot_y, tw2, sub_h, C_ADDNORM, C_ADDNORM_BORDER,
             "Add &\nNorm", r"LayerNorm",
             title_fs=11.5, sub_fs=9.8)

    # Residual 1 arrow for TSA (from Input z to Add & Norm 1)
    draw_residual(tx0 + tw0/2.0, bot_y + sub_h - 0.5, tx2 + tw2/2.0, bot_y + sub_h,
                  bend_y=bot_y + sub_h + 3.2, color="#C2185B", label="Residual 1 (+)")

    draw_arrow(tx2 + tw2, cy_sub, tx2 + tw2 + 1.7, cy_sub)

    # c4: Feed-Forward Network [120.6, 133.6]
    tx3 = tx2 + tw2 + 1.7; tw3 = 13.0
    draw_box(tx3, bot_y, tw3, sub_h, C_FFN, C_FFN_BORDER,
             "Feed-Forward\nNetwork", r"GELU activation",
             title_fs=11.2, sub_fs=9.5)

    draw_arrow(tx3 + tw3, cy_sub, tx3 + tw3 + 1.7, cy_sub)

    # c5: Add & Norm 2 [135.3, 146.3]
    tx4 = tx3 + tw3 + 1.7; tw4 = 11.0
    draw_box(tx4, bot_y, tw4, sub_h, C_ADDNORM, C_ADDNORM_BORDER,
             "Add &\nNorm", r"LayerNorm",
             title_fs=11.5, sub_fs=9.8)

    # Residual 2 arrow for TSA (from Add & Norm 1 to Add & Norm 2)
    draw_residual(tx2 + tw2/2.0, bot_y + sub_h, tx4 + tw4/2.0, bot_y + sub_h,
                  bend_y=bot_y + sub_h + 3.2, color="#2E7D32", label="Residual 2 (+)")

    draw_arrow(tx4 + tw4, cy_sub, tx4 + tw4 + 1.7, cy_sub)

    # c6: Output z' [148.0, 157.4] (symmetrical 9.4 width matching Input z!)
    tx5 = tx4 + tw4 + 1.7; tw5 = 9.4
    draw_box(tx5, bot_y + 0.5, tw5, sub_h - 1.0, C_INPUT, C_INPUT_BORDER, r"Output $\mathbf{z}'$",
             r"$(BN, T, C)$", title_fs=11.5, sub_fs=9.8)

    # =========================================================================
    # EXPORT FILES
    # =========================================================================
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    paper_fig_dir = os.path.join(curr_dir, "paper", "fig")
    os.makedirs(paper_fig_dir, exist_ok=True)

    targets = [
        os.path.join(paper_fig_dir, "ta_stgcn_architecture.pdf"),
        os.path.join(paper_fig_dir, "ta_stgcn_architecture.png"),
        os.path.join(paper_fig_dir, "at_stgcn_architecture.pdf"),
        os.path.join(paper_fig_dir, "at_stgcn_architecture.png"),
        os.path.join(paper_fig_dir, "ta_stgcn_architecture_compact.pdf"),
        os.path.join(paper_fig_dir, "ta_stgcn_architecture_compact.png"),
    ]

    print("\n🚀 Đang xuất hình ảnh kiến trúc chuẩn tọa độ...")
    for t in targets:
        try:
            plt.savefig(t, dpi=300, bbox_inches='tight', pad_inches=0.03, facecolor='white')
            print(f"  ✅ Đã lưu thành công: {t}")
        except PermissionError:
            print(f"  ⚠️ KHÔNG THỂ GHI ĐÈ: {t}")
            print(f"     => File đang bị mở và khóa bởi VS Code / PDF Viewer trên Windows.")
            print(f"     => Vui lòng đóng tab xem file này rồi chạy lại script.")
        except Exception as e:
            print(f"  ❌ Lỗi khi lưu {t}: {e}")

    plt.close()
    print("✨ Hoàn tất xuất ảnh căn lề chuẩn 100%!\n")

if __name__ == "__main__":
    draw_ta_stgcn_architecture()
