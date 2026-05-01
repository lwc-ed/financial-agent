const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageNumber, Header, Footer
} = require('docx');
const fs = require('fs');

// ── Data ──────────────────────────────────────────────────────────────────────

const models = ['bigru', 'bigru_TL_alignment', 'bilstm', 'bilstm_TL_alignment', 'gru_TL_alignment', 'xgboost_TL_alignment'];

const modelLabels = {
  bigru:               'BiGRU',
  bigru_TL_alignment:  'BiGRU + TL',
  bilstm:              'BiLSTM',
  bilstm_TL_alignment: 'BiLSTM + TL',
  gru_TL_alignment:    'GRU + TL',
  xgboost_TL_alignment:'XGBoost + TL',
};

function parseSummary(base, model) {
  const txt = fs.readFileSync(`${base}/${model}/summary.txt`, 'utf8');
  const g = (key) => { const m = txt.match(new RegExp(key + '[:\\s]+([0-9.]+)')); return m ? parseFloat(m[1]) : null; };
  return {
    binary: {
      Accuracy: g('Accuracy'),
      Precision: g('Precision'),
      Recall: g('Recall'),
      'F1-score': g('F1-score'),
    },
    risk4: {
      Accuracy: (() => { const m = txt.match(/Risk 4-Class[\s\S]*?Accuracy:\s*([0-9.]+)/); return m ? parseFloat(m[1]) : null; })(),
      'Macro Precision': g('Macro Precision'),
      'Macro Recall': g('Macro Recall'),
      'Macro F1': g('Macro F1'),
      'Weighted F1': g('Weighted F1'),
    },
    regression: {
      MAE: g('MAE'),
      RMSE: g('RMSE'),
      MAPE: g('MAPE'),
      SMAPE: g('SMAPE'),
    },
  };
}

function loadMetrics(base, model) {
  const jsonPath = `${base}/${model}/metrics_alarm_binary.json`;
  if (!fs.existsSync(jsonPath)) {
    return parseSummary(base, model);
  }
  const read = (f) => JSON.parse(fs.readFileSync(`${base}/${model}/${f}`, 'utf8'));
  return {
    binary:     read('metrics_alarm_binary.json'),
    risk4:      read('metrics_risk_4class.json'),
    regression: read('metrics_regression.json'),
  };
}

const walmart = {};
const ibm     = {};
for (const m of models) {
  walmart[m] = loadMetrics('/Users/liweichen/financial-agent/ml/model_outputs', m);
  ibm[m]     = loadMetrics('/Users/liweichen/financial-agent/ml_ibm/model_outputs', m);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const border = { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' };
const borders = { top: border, bottom: border, left: border, right: border };
const thickBorderBottom = { style: BorderStyle.SINGLE, size: 4, color: '2E75B6' };

function fmt(v, digits = 4) { return typeof v === 'number' ? v.toFixed(digits) : v; }
function pct(v, digits = 2) { return typeof v === 'number' ? (v * 100).toFixed(digits) + '%' : v; }
function delta(ibmVal, walmartVal, higherBetter = true) {
  const diff = ibmVal - walmartVal;
  const sign = diff > 0 ? '+' : '';
  return { text: sign + diff.toFixed(4), up: higherBetter ? diff > 0 : diff < 0 };
}

function cell(text, opts = {}) {
  const { bold = false, bg = null, align = AlignmentType.CENTER, color = '000000', italic = false } = opts;
  return new TableCell({
    borders,
    width: { size: opts.width || 1400, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    shading: bg ? { fill: bg, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text: String(text), bold, color, font: 'Arial', size: 18, italics: italic })],
    })],
  });
}

function headerCell(text, bg = '2E75B6', width = 1400) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    shading: { fill: bg, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, color: 'FFFFFF', font: 'Arial', size: 18 })],
    })],
  });
}

function deltaCell(ibmVal, walmartVal, higherBetter = true, width = 1200) {
  const d = delta(ibmVal, walmartVal, higherBetter);
  const color = d.up ? '1A7A1A' : 'CC0000';
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: d.text, bold: true, color, font: 'Arial', size: 18 })],
    })],
  });
}

function sectionHeading(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 300, after: 120 },
    children: [new TextRun({ text, font: 'Arial', size: 28, bold: true, color: '2E75B6' })],
  });
}

function note(text) {
  return new Paragraph({
    spacing: { before: 80, after: 160 },
    children: [new TextRun({ text, font: 'Arial', size: 16, color: '666666', italics: true })],
  });
}

// ── Table builders ────────────────────────────────────────────────────────────

// Binary Alarm table
// Cols: Model | Walmart Acc/Prec/Rec/F1 | IBM Acc/Prec/Rec/F1 | Delta F1
function buildBinaryTable() {
  const COL = [1600, 1100, 1100, 1100, 1100, 1100, 1100, 1100, 1100, 1000];
  const totalW = COL.reduce((a, b) => a + b, 0);

  const headerRow = new TableRow({ tableHeader: true, children: [
    headerCell('模型', '1F4E79', COL[0]),
    headerCell('Accuracy', '2E75B6', COL[1]),
    headerCell('Precision', '2E75B6', COL[2]),
    headerCell('Recall', '2E75B6', COL[3]),
    headerCell('F1', '2E75B6', COL[4]),
    headerCell('Accuracy', '1A6B8A', COL[5]),
    headerCell('Precision', '1A6B8A', COL[6]),
    headerCell('Recall', '1A6B8A', COL[7]),
    headerCell('F1', '1A6B8A', COL[8]),
    headerCell('ΔF1', '4A4A4A', COL[9]),
  ]});

  const subHeaderRow = new TableRow({ tableHeader: true, children: [
    headerCell('', '1F4E79', COL[0]),
    headerCell('Walmart (ml/)', '2E75B6', COL[1] + COL[2] + COL[3] + COL[4]),
    headerCell('IBM (ml_ibm/)', '1A6B8A', COL[5] + COL[6] + COL[7] + COL[8]),
    headerCell('IBM - Walmart', '4A4A4A', COL[9]),
  ]});

  // Actually build header as two separate rows is complex with colspan.
  // Let's just do one header row with all columns.

  const rows = models.map((m, i) => {
    const w = walmart[m].binary;
    const ib = ibm[m].binary;
    const bg = i % 2 === 0 ? 'F7F9FC' : 'FFFFFF';
    return new TableRow({ children: [
      cell(modelLabels[m], { width: COL[0], bold: true, align: AlignmentType.LEFT, bg }),
      cell(pct(w.Accuracy), { width: COL[1], bg }),
      cell(pct(w.Precision), { width: COL[2], bg }),
      cell(pct(w.Recall), { width: COL[3], bg }),
      cell(pct(w['F1-score']), { width: COL[4], bg }),
      cell(pct(ib.Accuracy), { width: COL[5], bg }),
      cell(pct(ib.Precision), { width: COL[6], bg }),
      cell(pct(ib.Recall), { width: COL[7], bg }),
      cell(pct(ib['F1-score']), { width: COL[8], bg }),
      deltaCell(ib['F1-score'], w['F1-score'], true, COL[9]),
    ]});
  });

  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: COL,
    rows: [headerRow, ...rows],
  });
}

// Risk 4-Class table
// Cols: Model | Walmart (Acc, MacroF1, WtdF1) | IBM (Acc, MacroF1, WtdF1) | Delta MacroF1 | Delta WtdF1
function buildRisk4Table() {
  const COL = [1600, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200];
  const totalW = COL.reduce((a, b) => a + b, 0);

  const headerRow = new TableRow({ tableHeader: true, children: [
    headerCell('模型', '1F4E79', COL[0]),
    headerCell('Accuracy', '2E75B6', COL[1]),
    headerCell('Macro F1', '2E75B6', COL[2]),
    headerCell('Wtd F1', '2E75B6', COL[3]),
    headerCell('Accuracy', '1A6B8A', COL[4]),
    headerCell('Macro F1', '1A6B8A', COL[5]),
    headerCell('Wtd F1', '1A6B8A', COL[6]),
    headerCell('ΔMacro F1', '4A4A4A', COL[7]),
    headerCell('ΔWtd F1', '4A4A4A', COL[8]),
  ]});

  const rows = models.map((m, i) => {
    const w = walmart[m].risk4;
    const ib = ibm[m].risk4;
    const bg = i % 2 === 0 ? 'F7F9FC' : 'FFFFFF';
    return new TableRow({ children: [
      cell(modelLabels[m], { width: COL[0], bold: true, align: AlignmentType.LEFT, bg }),
      cell(pct(w.Accuracy), { width: COL[1], bg }),
      cell(pct(w['Macro F1']), { width: COL[2], bg }),
      cell(pct(w['Weighted F1']), { width: COL[3], bg }),
      cell(pct(ib.Accuracy), { width: COL[4], bg }),
      cell(pct(ib['Macro F1']), { width: COL[5], bg }),
      cell(pct(ib['Weighted F1']), { width: COL[6], bg }),
      deltaCell(ib['Macro F1'], w['Macro F1'], true, COL[7]),
      deltaCell(ib['Weighted F1'], w['Weighted F1'], true, COL[8]),
    ]});
  });

  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: COL,
    rows: [headerRow, ...rows],
  });
}

// Regression table
// Cols: Model | Walmart (MAE, RMSE, MAPE, SMAPE) | IBM (MAE, RMSE, MAPE, SMAPE) | ΔMAE | ΔRMSE
function buildRegressionTable() {
  const COL = [1600, 1050, 1050, 1000, 1000, 1050, 1050, 1000, 1000, 900, 900];
  const totalW = COL.reduce((a, b) => a + b, 0);

  const headerRow = new TableRow({ tableHeader: true, children: [
    headerCell('模型', '1F4E79', COL[0]),
    headerCell('MAE', '2E75B6', COL[1]),
    headerCell('RMSE', '2E75B6', COL[2]),
    headerCell('MAPE%', '2E75B6', COL[3]),
    headerCell('SMAPE%', '2E75B6', COL[4]),
    headerCell('MAE', '1A6B8A', COL[5]),
    headerCell('RMSE', '1A6B8A', COL[6]),
    headerCell('MAPE%', '1A6B8A', COL[7]),
    headerCell('SMAPE%', '1A6B8A', COL[8]),
    headerCell('ΔMAE', '4A4A4A', COL[9]),
    headerCell('ΔRMSE', '4A4A4A', COL[10]),
  ]});

  const rows = models.map((m, i) => {
    const w = walmart[m].regression;
    const ib = ibm[m].regression;
    const bg = i % 2 === 0 ? 'F7F9FC' : 'FFFFFF';
    return new TableRow({ children: [
      cell(modelLabels[m], { width: COL[0], bold: true, align: AlignmentType.LEFT, bg }),
      cell(w.MAE.toFixed(1), { width: COL[1], bg }),
      cell(w.RMSE.toFixed(1), { width: COL[2], bg }),
      cell(w.MAPE.toFixed(1), { width: COL[3], bg }),
      cell(w.SMAPE.toFixed(1), { width: COL[4], bg }),
      cell(ib.MAE.toFixed(1), { width: COL[5], bg }),
      cell(ib.RMSE.toFixed(1), { width: COL[6], bg }),
      cell(ib.MAPE.toFixed(1), { width: COL[7], bg }),
      cell(ib.SMAPE.toFixed(1), { width: COL[8], bg }),
      deltaCell(ib.MAE, w.MAE, false, COL[9]),  // lower MAE is better
      deltaCell(ib.RMSE, w.RMSE, false, COL[10]),
    ]});
  });

  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: COL,
    rows: [headerRow, ...rows],
  });
}

// ── Summary table ─────────────────────────────────────────────────────────────

function buildSummaryTable() {
  const COL = [1600, 1100, 1100, 1100, 1100, 1100, 1300];
  const totalW = COL.reduce((a, b) => a + b, 0);

  const headerRow = new TableRow({ tableHeader: true, children: [
    headerCell('模型', '1F4E79', COL[0]),
    headerCell('Binary F1\nWalmart', '2E75B6', COL[1]),
    headerCell('Binary F1\nIBM', '2E75B6', COL[2]),
    headerCell('Macro F1\nWalmart', '1A6B8A', COL[3]),
    headerCell('Macro F1\nIBM', '1A6B8A', COL[4]),
    headerCell('MAE\nWalmart→IBM', '4A4A4A', COL[5]),
    headerCell('整體判斷', '1F4E79', COL[6]),
  ]});

  function verdict(m) {
    const wB = walmart[m].binary['F1-score'];
    const iB = ibm[m].binary['F1-score'];
    const wR = walmart[m].risk4['Macro F1'];
    const iR = ibm[m].risk4['Macro F1'];
    const wM = walmart[m].regression.MAE;
    const iM = ibm[m].regression.MAE;
    let ups = 0, downs = 0;
    if (iB > wB) ups++; else downs++;
    if (iR > wR) ups++; else downs++;
    if (iM < wM) ups++; else downs++;
    if (ups === 3) return { text: '全面提升 ↑↑↑', color: '1A7A1A' };
    if (ups === 2) return { text: '部分提升 ↑↑↓', color: '2E75B6' };
    if (ups === 1) return { text: '部分退步 ↑↓↓', color: 'CC7700' };
    return { text: '全面退步 ↓↓↓', color: 'CC0000' };
  }

  const rows = models.map((m, i) => {
    const wB = walmart[m].binary['F1-score'];
    const iB = ibm[m].binary['F1-score'];
    const wR = walmart[m].risk4['Macro F1'];
    const iR = ibm[m].risk4['Macro F1'];
    const wM = walmart[m].regression.MAE;
    const iM = ibm[m].regression.MAE;
    const v = verdict(m);
    const bg = i % 2 === 0 ? 'F7F9FC' : 'FFFFFF';

    return new TableRow({ children: [
      cell(modelLabels[m], { width: COL[0], bold: true, align: AlignmentType.LEFT, bg }),
      cell(pct(wB), { width: COL[1], bg }),
      cell(pct(iB), { width: COL[2], bg }),
      cell(pct(wR), { width: COL[3], bg }),
      cell(pct(iR), { width: COL[4], bg }),
      cell(`${wM.toFixed(0)} → ${iM.toFixed(0)}`, { width: COL[5], bg }),
      new TableCell({
        borders,
        width: { size: COL[6], type: WidthType.DXA },
        verticalAlign: VerticalAlign.CENTER,
        shading: { fill: bg, type: ShadingType.CLEAR },
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: v.text, bold: true, color: v.color, font: 'Arial', size: 18 })],
        })],
      }),
    ]});
  });

  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: COL,
    rows: [headerRow, ...rows],
  });
}

// ── Document ──────────────────────────────────────────────────────────────────

const doc = new Document({
  styles: {
    default: { document: { run: { font: 'Arial', size: 22 } } },
    paragraphStyles: [
      {
        id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 36, bold: true, font: 'Arial', color: '1F4E79' },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 },
      },
      {
        id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, font: 'Arial', color: '2E75B6' },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1 },
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 15840, height: 12240 }, // Landscape A4-ish, actually let's use landscape Letter
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '2E75B6', space: 1 } },
          children: [new TextRun({ text: 'Walmart → IBM 模型遷移效果比較報告', font: 'Arial', size: 20, color: '1F4E79', bold: true })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [
            new TextRun({ text: '第 ', font: 'Arial', size: 18, color: '888888' }),
            new TextRun({ children: [PageNumber.CURRENT], font: 'Arial', size: 18, color: '888888' }),
            new TextRun({ text: ' 頁', font: 'Arial', size: 18, color: '888888' }),
          ],
        })],
      }),
    },
    children: [
      // Title
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        children: [new TextRun({ text: 'Walmart → IBM 模型遷移效果比較報告', font: 'Arial', size: 40, bold: true, color: '1F4E79' })],
      }),
      new Paragraph({
        spacing: { after: 80 },
        children: [new TextRun({ text: '比較基準：ml/model_outputs（Walmart）vs ml_ibm/model_outputs（IBM）', font: 'Arial', size: 20, color: '444444' })],
      }),
      new Paragraph({
        spacing: { after: 300 },
        children: [new TextRun({ text: '生成日期：2026-05-01', font: 'Arial', size: 20, color: '444444' })],
      }),

      // Section 1: Summary
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        children: [new TextRun({ text: '1. 整體摘要', font: 'Arial', size: 28, bold: true, color: '2E75B6' })],
      }),
      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun({ text: '以下摘要三項測試（Binary Alarm F1、Risk 4-Class Macro F1、Regression MAE）在兩個資料集上的表現，並給出整體遷移判斷。', font: 'Arial', size: 20, color: '333333' })],
      }),
      buildSummaryTable(),
      note('↑ = IBM 比 Walmart 更好；↓ = IBM 比 Walmart 更差。MAE 以絕對值呈現，箭頭以數值下降為優。'),

      // Section 2: Binary
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 400 },
        children: [new TextRun({ text: '2. Test 1 — Binary Alarm 分類', font: 'Arial', size: 28, bold: true, color: '2E75B6' })],
      }),
      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun({ text: '判斷帳戶是否觸發警報的二元分類任務。ΔF1 = IBM F1 − Walmart F1，正值代表 IBM 資料上進步。', font: 'Arial', size: 20, color: '333333' })],
      }),
      buildBinaryTable(),
      note('藍色欄位 = Walmart 數值；深藍欄位 = IBM 數值；ΔF1 欄：綠色正值 = 進步，紅色負值 = 退步。'),

      // Section 3: Risk 4-class
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 400 },
        children: [new TextRun({ text: '3. Test 2 — Risk 4-Class 風險分級', font: 'Arial', size: 28, bold: true, color: '2E75B6' })],
      }),
      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun({ text: '將帳戶分為 no_alarm / low_risk / mid_risk / high_risk 四類。Macro F1 反映對少數類別的掌握；Weighted F1 反映整體加權效果。', font: 'Arial', size: 20, color: '333333' })],
      }),
      buildRisk4Table(),
      note('ΔMacro F1 / ΔWtd F1：綠色正值 = 進步，紅色負值 = 退步。'),

      // Section 4: Regression
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 400 },
        children: [new TextRun({ text: '4. Test 3 — Regression 回歸預測', font: 'Arial', size: 28, bold: true, color: '2E75B6' })],
      }),
      new Paragraph({
        spacing: { after: 120 },
        children: [new TextRun({ text: '預測未來 7 天可用現金（future_available_7d）。MAE/RMSE 數值越小越好；ΔMAE 負值代表在 IBM 資料上誤差縮小（進步）。', font: 'Arial', size: 20, color: '333333' })],
      }),
      buildRegressionTable(),
      note('ΔMAE / ΔRMSE：綠色負值 = 誤差下降（進步），紅色正值 = 誤差上升（退步）。'),

      // Section 5: Observations
      new Paragraph({
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 400 },
        children: [new TextRun({ text: '5. 重點觀察', font: 'Arial', size: 28, bold: true, color: '2E75B6' })],
      }),
      ...[
        'Binary Alarm：BiLSTM + TL 在 IBM 資料上 F1 從 84.2% 提升至 86.0%，為六個模型中 IBM F1 最高；BiGRU（無TL）則從 85.6% 下滑至 81.6%，是唯一明顯退步的模型。',
        'Risk 4-Class：BiGRU + TL 在 IBM 上 Macro F1 從 46.1% 大幅提升至 52.9%，顯示對少數風險類別的辨識力改善最顯著；BiLSTM + TL 在 IBM 上 Macro F1 反而從 42.1% 下滑至 46.7%，整體較弱。',
        'Regression：GRU + TL 在 IBM 上 MAE 從 1,090 微降至 1,040，且 RMSE 最低（1,521），回歸預測為六模型最佳；XGBoost + TL 在 IBM 上 MAE 從 949 上升至 1,232，為唯一在回歸任務上顯著退步的模型。',
        '整體遷移表現：BiGRU + TL 及 GRU + TL 在 IBM 資料上表現最穩定，可視為遷移到新資料集後最值得優先採用的架構。XGBoost + TL 在分類上可接受，但回歸能力在 IBM 資料上明顯不足。',
      ].map(txt => new Paragraph({
        spacing: { before: 80, after: 80 },
        bullet: undefined,
        children: [
          new TextRun({ text: '• ', font: 'Arial', size: 20, bold: true, color: '2E75B6' }),
          new TextRun({ text: txt, font: 'Arial', size: 20, color: '222222' }),
        ],
      })),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('/Users/liweichen/financial-agent/model_comparison_report.docx', buf);
  console.log('Done: model_comparison_report.docx');
});
