#!/usr/bin/env python3
"""Build the Chinese IBM backend progress report PDF."""

from pathlib import Path

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = Path("output/pdf/ibm_backend_progress_report_zh.pdf")
FONT = Path(r"C:\Windows\Fonts\NotoSansTC-VF.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\msjhbd.ttc")

NAVY = HexColor("#102A43")
BLUE = HexColor("#1769AA")
CYAN = HexColor("#29B6C8")
PALE = HexColor("#EAF4FA")
GREEN = HexColor("#1B7F5A")
AMBER = HexColor("#C47F00")
RED = HexColor("#B64242")
INK = HexColor("#243B53")
MUTED = HexColor("#627D98")
GRID = HexColor("#CBD5E1")


def register_fonts():
    pdfmetrics.registerFont(TTFont("NotoTC", str(FONT)))
    try:
        pdfmetrics.registerFont(TTFont("NotoTC-Bold", str(FONT_BOLD)))
    except Exception:
        pdfmetrics.registerFont(TTFont("NotoTC-Bold", str(FONT)))


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="NotoTC-Bold", fontSize=27, leading=36, textColor=colors.white, alignment=TA_LEFT),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontName="NotoTC", fontSize=12, leading=19, textColor=HexColor("#D9EAF4")),
        "cover_body": ParagraphStyle("cover_body", parent=base["Normal"], fontName="NotoTC", fontSize=11, leading=18, textColor=MUTED),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="NotoTC-Bold", fontSize=20, leading=27, textColor=NAVY, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="NotoTC-Bold", fontSize=13, leading=19, textColor=BLUE, spaceBefore=8, spaceAfter=5),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="NotoTC", fontSize=10, leading=16, textColor=INK, spaceAfter=6),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName="NotoTC", fontSize=8.2, leading=12.5, textColor=MUTED),
        "callout": ParagraphStyle("callout", parent=base["BodyText"], fontName="NotoTC-Bold", fontSize=11, leading=17, textColor=NAVY),
        "metric": ParagraphStyle("metric", parent=base["BodyText"], fontName="NotoTC-Bold", fontSize=19, leading=22, textColor=BLUE, alignment=TA_CENTER),
        "metric_label": ParagraphStyle("metric_label", parent=base["BodyText"], fontName="NotoTC", fontSize=8, leading=11, textColor=MUTED, alignment=TA_CENTER),
        "table": ParagraphStyle("table", parent=base["BodyText"], fontName="NotoTC", fontSize=8.3, leading=12, textColor=INK),
        "table_head": ParagraphStyle("table_head", parent=base["BodyText"], fontName="NotoTC-Bold", fontSize=8.3, leading=12, textColor=colors.white, alignment=TA_CENTER),
    }


def P(text, style):
    return Paragraph(text, style)


def header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 13 * mm, w, 13 * mm, fill=1, stroke=0)
    canvas.setFont("NotoTC-Bold", 8)
    canvas.setFillColor(colors.white)
    canvas.drawString(18 * mm, h - 8.5 * mm, "Qiskit DRL VQC｜IBM Backend 進度報告")
    canvas.setStrokeColor(GRID)
    canvas.line(18 * mm, 14 * mm, w - 18 * mm, 14 * mm)
    canvas.setFont("NotoTC", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "NTU-IBM Quantum System 使用者大會 Qiskit Hackathon 2026")
    canvas.drawRightString(w - 18 * mm, 9 * mm, f"{doc.page}")
    canvas.restoreState()


def table(data, widths, s, header=True):
    cooked = []
    for r, row in enumerate(data):
        cooked.append([P(str(v), s["table_head"] if r == 0 and header else s["table"]) for v in row])
    t = Table(cooked, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
        for r in range(1, len(data)):
            commands.append(("BACKGROUND", (0, r), (-1, r), colors.white if r % 2 else PALE))
    t.setStyle(TableStyle(commands))
    return t


def metric_cards(s):
    data = [
        [P("46", s["metric"]), P("2/2", s["metric"]), P("93", s["metric"]), P("34", s["metric"])],
        [P("VQC 可訓練參數", s["metric_label"]), P("硬體動作一致", s["metric_label"]), P("ISA 電路深度", s["metric_label"]), P("雙量子位閘", s["metric_label"])],
    ]
    t = Table(data, colWidths=[41 * mm] * 4, rowHeights=[11 * mm, 8 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.6, HexColor("#A9CFE3")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, HexColor("#A9CFE3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def q_error_chart():
    d = Drawing(470, 205)
    chart = VerticalBarChart()
    chart.x = 50
    chart.y = 35
    chart.height = 130
    chart.width = 380
    chart.data = [(9.372, 11.805), (2.851, 11.494)]
    chart.categoryAxis.categoryNames = ["State 0", "State 1"]
    chart.categoryAxis.labels.fontName = "NotoTC"
    chart.categoryAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = 14
    chart.valueAxis.valueStep = 2
    chart.valueAxis.labels.fontName = "NotoTC"
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = BLUE
    chart.bars[1].fillColor = CYAN
    chart.barSpacing = 3
    chart.groupSpacing = 14
    d.add(chart)
    d.add(String(50, 185, "IBM hardware 相對 exact simulation 的 |Q-value error|", fontName="NotoTC-Bold", fontSize=11, fillColor=NAVY))
    d.add(String(135, 8, "深藍：Q0　青色：Q1", fontName="NotoTC", fontSize=8, fillColor=MUTED))
    return d


def build():
    register_fonts()
    s = styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm, bottomMargin=18 * mm, title="IBM Backend 進度與實驗規劃報告", author="Qiskit DRL VQC Team")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=header_footer)])
    story = []

    # Cover
    story += [Spacer(1, 28 * mm)]
    cover = Table([[P("IBM Backend 進度<br/>與下一階段實驗規劃", s["title"]), P("", s["body"])]], colWidths=[128 * mm, 36 * mm], rowHeights=[55 * mm])
    cover.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (0, 0), 14 * mm)]))
    story += [cover, Spacer(1, 8 * mm), P("Hybrid Quantum-Classical DQN for CartPole-v1", s["h2"]), P("目的：說明 IBM backend 已完成的串接、真實 QPU 結果、與 Classical／Quantum Torch 實驗的對齊方式，並提出可成功完成正式比較的步驟。", s["cover_body"]), Spacer(1, 38 * mm), P("報告日期：2026-08-13", s["body"]), P("Branch：quantum-part--seanliu", s["small"]), PageBreak()]

    # Executive summary
    story += [P("1. 執行摘要", s["h1"]), metric_cards(s), Spacer(1, 7 * mm)]
    summary_box = Table([[P("目前已打通完整鏈路：<b>已訓練 Torch VQC checkpoint → 同一組 normalization 與參數 → Qiskit exact 驗證 → IBM ISA transpilation → Runtime Estimator → results JSON</b>。真實硬體對兩個挑選狀態皆做出相同 action，但 State 0 的硬體 margin 僅 0.025，因此現在應稱為 preliminary hardware evaluation，而非完整 robustness 證明。", s["callout"])]], colWidths=[164 * mm])
    summary_box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), 1, CYAN), ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    story += [summary_box, Spacer(1, 5 * mm), P("最重要的定位", s["h2"]), P("IBM QPU 目前不是 DQN 的訓練 backend。訓練仍由 PyTorch 可微分 statevector 完成；IBM backend 負責訓練完成後的固定狀態推論與噪聲穩健度評估。這與黑客松時間和 QPU queue 限制相容，也能維持公平比較。", s["body"]), P("下一個里程碑", s["h2"]), P("將兩狀態流程泛化為 12 個固定狀態（high／medium／low exact margin 各 4 個，action 0／1 平衡），先本機驗證，再用單一 Runtime job 批次執行，產生 action agreement、Q-value error 與 margin 分層結果。", s["body"]), PageBreak()]

    # Architecture alignment
    story += [P("2. 三條實驗路徑如何對齊", s["h1"])]
    story += [table([
        ["項目", "Classical Zoo", "Quantum Torch", "IBM QPU evaluation"],
        ["用途", "主要 classical baseline", "訓練 compact VQC policy", "測試已訓練 VQC 的硬體噪聲"],
        ["Q-network", "256×256 MLP", "4-qubit、3-layer data-reuploading VQC", "同一個 VQC checkpoint 與電路"],
        ["可訓練參數", "67,586", "46", "不重新訓練；固定 46 個參數"],
        ["訓練 backend", "PyTorch CPU", "Torch differentiable statevector", "不適用（post-training inference）"],
        ["共享流程", "CartPole、normalize_obs、DQN loop、replay、target network、greedy eval", "同左", "沿用相同 normalization、selected states 與 action 定義"],
        ["主要輸出", "reward／solved／時間／參數", "reward／solved／時間／參數", "expectations／Q-values／action agreement／error"],
    ], [28*mm, 43*mm, 48*mm, 45*mm], s)]
    story += [Spacer(1, 7 * mm), P("公平比較的核心", s["h2"]), P("Classical 與 Quantum Torch 的訓練比較只替換 Q-function approximator；環境、觀測正規化、訓練迴圈和評估規則保持一致。IBM 評估則固定已訓練 policy，只替換 expectation-value execution backend，因此回答的是 hardware robustness，而不是訓練速度或 sample efficiency。", s["body"]), PageBreak()]

    # Progress
    story += [P("3. IBM backend 已完成進度", s["h1"])]
    story += [table([
        ["階段", "狀態", "產出／證據"],
        ["帳戶與 backend 連線", "完成", "可取得 operational QPU；使用 ibm_marrakesh（156 qubits）"],
        ["ISA-compatible VQC", "完成", "原始 depth 24 → ISA depth 93；two-qubit gates 34"],
        ["Runtime smoke test", "完成", "Job d9ujrck98n5s7392v2q0；驗證提交與取回結果"],
        ["Trained-policy local preview", "完成", "Torch 與 Qiskit exact max |ΔQ| = 9.375e-05"],
        ["Submission manifest", "完成", "2 states × 2 observables；44 circuit values/state"],
        ["首次 policy job", "保留為部分結果", "d9uks0l35hes73fk2fv0；發現 element-wise broadcasting"],
        ["Broadcast 修正", "完成", "observable (1,2) × parameter (2,1,44) → result (2,2)"],
        ["完整 hardware job", "完成", "d9ulamt35hes73fk3400；action agreement 2/2"],
        ["可重現摘要", "完成", "JSON + Markdown summary 與分析程式已提交 branch"],
    ], [39*mm, 29*mm, 96*mm], s)]
    story += [Spacer(1, 6 * mm), P("Data re-uploading 的位置", s["h2"]), P("Data re-uploading 已包含在 VQC 建構中：4 維 observation 經 input_scale 後，在 3 個 variational layers 重複編碼，共形成 12 個 input angles；另外有 32 個 θ weights 與 2 個 output scales，總計 46 個可訓練參數。", s["body"]), PageBreak()]

    # Results
    story += [P("4. 真實 IBM QPU 結果", s["h1"])]
    story += [table([
        ["State", "Qiskit exact Q", "IBM hardware Q", "Exact action", "Hardware action", "Hardware margin"],
        ["0", "(101.719, 89.470)", "(92.346, 92.321)", "0", "0", "0.025"],
        ["1", "(87.361, 101.617)", "(75.556, 90.123)", "1", "1", "14.567"],
    ], [15*mm, 39*mm, 41*mm, 24*mm, 27*mm, 25*mm], s), Spacer(1, 5 * mm), q_error_chart()]
    story += [P("結果解讀", s["h2"]), P("兩個 action 都一致（2/2），mean absolute Q-value error 為 8.881，最大誤差 11.805。State 1 的 action margin 仍大；State 0 的兩個 hardware Q-values 幾乎相等，表示其 action 對 finite-shot／hardware noise 很敏感。", s["body"]), P("限制：樣本只有兩個，而且是從單一 greedy episode 中挑選的 high exact-margin states；目前不能推論整條 policy 在狀態空間上的 hardware robustness，也不能把 100% 當成具統計代表性的準確率。", s["small"]), PageBreak()]

    # Classical vs quantum current evidence
    story += [P("5. 與 Classical／Quantum 實驗的現況比較", s["h1"])]
    story += [table([
        ["實驗", "參數", "Greedy eval mean", "Solved", "Elapsed", "可回答的問題"],
        ["Classical Zoo", "67,586", "500.0", "是", "83.0 s", "強 baseline 與解題能力"],
        ["Quantum Torch", "46", "201.44", "否", "349.0 s", "極低參數下可學到多少 policy"],
        ["IBM QPU（2 states）", "固定 46", "不適用", "不適用", "含 queue", "固定 policy 在硬體上的 action 保真度"],
    ], [31*mm, 21*mm, 29*mm, 18*mm, 22*mm, 43*mm], s)]
    story += [Spacer(1, 7 * mm), P("目前可支持的結論", s["h2"]), P("Compact data-reuploading VQC 僅用 46 個 trainable parameters，遠少於 67,586 參數的 Classical Zoo MLP；它在 Torch statevector 上學到非平凡 policy（greedy mean 201.44），但尚未解出 CartPole。真實 IBM QPU 對兩個固定狀態保留了 action，顯示硬體推論鏈路可行；尚未證明整體 robustness。", s["body"]), P("注意：`comparison.json` 中的 classical run（greedy 123.51）不是最強 Zoo baseline。正式報告應以 `classical_zoo_run.json` 的 greedy 500.0 作主要 classical 比較，並清楚標示不同設定。", s["callout"]), PageBreak()]

    # Next experiment
    story += [P("6. 下一階段：成功跑出可比較的正式實驗", s["h1"])]
    story += [table([
        ["步驟", "本機／QPU", "具體工作", "完成條件"],
        ["1. 建立狀態集", "本機", "從相同 greedy evaluation seeds 收集候選 states；依 exact margin 分 high／medium／low", "12 states；每層 4 個；action 0／1 平衡"],
        ["2. 精確驗證", "本機", "Torch 與 Qiskit StatevectorEstimator 比較 Q-values", "所有 states action 一致；差異 ≤ 1e-4"],
        ["3. 送出前預覽", "本機 + backend read", "固定 checkpoint hash、參數順序、ISA metrics、broadcast shape", "不送 job；manifest 可審查"],
        ["4. 批次硬體執行", "IBM QPU", "單一 PUB 批次執行 12×2 expectations，precision 0.1", "保存 job ID；得到 (12,2)"],
        ["5. 穩健度分析", "本機", "overall／margin-group agreement、MAE、max error、action flips", "JSON、CSV／圖表、Markdown 摘要"],
        ["6. 重複執行", "IBM QPU", "額度允許時，以相同 states 與參數再跑一次", "hardware-to-hardware action consistency"],
    ], [16*mm, 25*mm, 76*mm, 47*mm], s)]
    story += [Spacer(1, 6 * mm), P("為什麼先做 12 states", s["h2"]), P("比兩個 state 更能觀察 decision boundary，又可控制 QPU 成本。high-margin 檢查明確決策是否保留；medium-margin 反映一般狀態；low-margin 專門測試 noise 是否造成 action flip。", s["body"]), P("成功後再擴展到 20–50 states，而不是直接把完整 CartPole episode 放在 QPU queue 中逐步互動。後者延遲高、成本大，也不適合三天黑客松的可重現實驗。", s["body"]), PageBreak()]

    # Operational checklist
    story += [P("7. 執行與成果交付清單", s["h1"])]
    story += [table([
        ["類型", "應保留檔案", "用途"],
        ["訓練模型", "results/quantum_policy.pt", "固定 46-parameter VQC checkpoint"],
        ["Torch 訓練結果", "results/quantum_run.json", "reward、時間、參數、greedy evaluation"],
        ["IBM 前置驗證", "artifacts/ibm/validation/ibm_policy_preview.json", "selected states 與 exact validation"],
        ["IBM 送出 manifest", "artifacts/ibm/validation/ibm_policy_submission_preview.json", "checkpoint hash、ISA 與參數 binding"],
        ["IBM 正式結果", "artifacts/ibm/core/ibm_policy_evaluation_<job-id>.json", "expectations、Q-values、actions、errors"],
        ["報告摘要", "artifacts/ibm/core/ibm_policy_hardware_summary.{json,md}", "GitHub 與簡報可直接引用"],
    ], [30*mm, 73*mm, 61*mm], s)]
    story += [Spacer(1, 7 * mm), P("重要操作原則", s["h2"]), P("每次真正送出前先產生 preview；確認 `qpu_job_submitted=false`、checkpoint SHA-256、state 數量、observable 數量與 broadcast result shape。送出後立即保存 job ID；終端中斷時使用 `--resume JOB_ID`，不要重新執行提交指令，以免產生重複 job。", s["body"]), P("建議接續 commit", s["h2"]), P("下一個 commit 應只包含 12-state 泛化程式、local preview 與分析模板；真正 QPU 結果完成後再以獨立 commit 加入，方便追蹤實驗版本。", s["body"]), Spacer(1, 12 * mm), P("結論：IBM backend 技術串接已完成。接下來的重點不再是『能否連線』，而是建立公平、批次化、可重現且具分層分析的 hardware robustness experiment。", s["callout"])]

    doc.build(story)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    build()
