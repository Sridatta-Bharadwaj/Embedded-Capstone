// Shared helpers for building the E0 report with docx-js.
const fs = require("fs");
const path = require("path");
const d = require("docx");

const FONT = "Times New Roman";
const FIG = path.join(__dirname, "figures");
const TEXT_W = 11906 - 2160 - 1440; // A4 width minus left 1.5" and right 1" margins (DXA)

// Figure / table registries: numbers follow order of appearance, defined up front so text can cite them.
const FIGS = ["arch", "samples", "classdist", "method", "cleaning", "folds", "model", "curves0", "aucall",
  "perfold", "roc", "cm", "final", "gradcam"];
const TABS = ["tech", "dataset", "backbones", "folds", "hparams", "env", "tests", "perfold", "pooled", "repro",
  "final"];
const F = (k) => { const i = FIGS.indexOf(k); if (i < 0) throw new Error("fig " + k); return i + 1; };
const T = (k) => { const i = TABS.indexOf(k); if (i < 0) throw new Error("tab " + k); return i + 1; };

// Inline markup: **bold**, *italic*, {fig:key}, {tab:key}
function runs(text, base = {}) {
  text = text.replace(/\{fig:(\w+)\}/g, (_, k) => `Fig. ${F(k)}`).replace(/\{tab:(\w+)\}/g, (_, k) => `Table ${T(k)}`);
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new d.TextRun({ text: text.slice(last, m.index), ...base }));
    const s = m[0];
    if (s.startsWith("**")) out.push(new d.TextRun({ text: s.slice(2, -2), bold: true, ...base }));
    else out.push(new d.TextRun({ text: s.slice(1, -1), italics: true, ...base }));
    last = m.index + s.length;
  }
  if (last < text.length) out.push(new d.TextRun({ text: text.slice(last), ...base }));
  return out;
}

// Every heading and caption is recorded (in document order) so a static contents list can be built from the rendered PDF.
const OUTLINE = [];
const P = (text, opt = {}) => new d.Paragraph({ style: "Body", children: runs(text), ...opt });
const PN = (text, opt = {}) => new d.Paragraph({ style: "BodyNoIndent", children: runs(text), ...opt });
const H1 = (text) => (OUTLINE.push({ kind: "toc", level: 1, text }), new d.Paragraph({ heading: d.HeadingLevel.HEADING_1, children: [new d.TextRun(text)] }));
const H2 = (text) => (OUTLINE.push({ kind: "toc", level: 2, text }), new d.Paragraph({ heading: d.HeadingLevel.HEADING_2, children: [new d.TextRun(text)] }));
const H3 = (text) => (OUTLINE.push({ kind: "toc", level: 3, text }), new d.Paragraph({ heading: d.HeadingLevel.HEADING_3, children: [new d.TextRun(text)] }));
const Bullets = (items) => items.map((t) => new d.Paragraph({ style: "BodyNoIndent", numbering: { reference: "bullets", level: 0 }, children: runs(t) }));
const Blank = (n = 1) => Array.from({ length: n }, () => new d.Paragraph({ children: [] }));
const PageBreak = () => new d.Paragraph({ children: [new d.PageBreak()] });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}

function Img(file, widthIn, opt = {}) {
  const p = path.join(FIG, file);
  const [w, h] = pngSize(p);
  const wpx = widthIn * 96;
  return new d.Paragraph({
    alignment: d.AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 }, ...opt,
    children: [new d.ImageRun({ type: "png", data: fs.readFileSync(p), transformation: { width: wpx, height: wpx * h / w } })],
  });
}

function Fig(key, file, widthIn, caption) {
  OUTLINE.push({ kind: "fig", text: `Fig. ${F(key)}. ${caption}` });
  return [Img(file, widthIn), new d.Paragraph({ style: "FigCaption", children: runs(`Fig. ${F(key)}. ${caption}`) })];
}

const border = { style: d.BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(text, width, { header = false, align = d.AlignmentType.LEFT } = {}) {
  return new d.TableCell({
    borders, width: { size: width, type: d.WidthType.DXA },
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    shading: header ? { fill: "E7E6E6", type: d.ShadingType.CLEAR, color: "auto" } : undefined,
    verticalAlign: d.VerticalAlign.CENTER,
    children: [new d.Paragraph({ style: "TableText", alignment: header ? d.AlignmentType.CENTER : align, children: runs(String(text), header ? { bold: true } : {}) })],
  });
}

// widths: relative weights scaled to the text width; first column left-aligned, the rest centred unless leftCols says otherwise.
function Table(key, caption, headers, rows, widths, { leftCols = [0], scale = 1 } = {}) {
  const total = Math.round(TEXT_W * scale);
  const sum = widths.reduce((a, b) => a + b, 0);
  const cw = widths.map((w) => Math.floor((w / sum) * total));
  cw[cw.length - 1] += total - cw.reduce((a, b) => a + b, 0);
  OUTLINE.push({ kind: "tab", text: `Table ${T(key)} — ${caption}` });
  const al = (i) => (leftCols.includes(i) ? d.AlignmentType.LEFT : d.AlignmentType.CENTER);
  return [
    new d.Paragraph({ style: "TabCaption", children: runs(`Table ${T(key)} — ${caption}`) }),
    new d.Table({
      width: { size: total, type: d.WidthType.DXA }, columnWidths: cw, alignment: d.AlignmentType.CENTER,
      rows: [
        new d.TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, cw[i], { header: true })) }),
        ...rows.map((r) => new d.TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, cw[i], { align: al(i) })) })),
      ],
    }),
    new d.Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

// Display equation with a right-aligned label like "(4.1)".
function Eq(mathChildren, label) {
  return new d.Paragraph({
    alignment: d.AlignmentType.CENTER, spacing: { before: 120, after: 120 },
    tabStops: [{ type: d.TabStopType.RIGHT, position: TEXT_W }],
    children: [new d.Math({ children: mathChildren }), new d.TextRun({ text: `\t${label}` })],
  });
}
const mr = (t) => new d.MathRun(t);
const frac = (num, den) => new d.MathFraction({ numerator: num, denominator: den });
const sub = (base, s) => new d.MathSubScript({ children: [mr(base)], subScript: [mr(s)] });

module.exports = { OUTLINE, d, FONT, TEXT_W, F, T, runs, P, PN, H1, H2, H3, Bullets, Blank, PageBreak, Img, Fig, Table, Eq, mr, frac, sub };
