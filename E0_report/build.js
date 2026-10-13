// Assemble the E0 project report into E0_Project_Report.docx.
const fs = require("fs");
const path = require("path");
const { d, FONT } = require("./lib");
const { ch1, ch2 } = require("./ch1_2");
const ch3 = require("./ch3");
const ch4 = require("./ch4");
const ch5 = require("./ch5");
const { ch6, references } = require("./ch6");
const { OUTLINE } = require("./lib");
fs.writeFileSync(path.join(__dirname, "outline.json"), JSON.stringify(OUTLINE, null, 1));
const front = require("./front"); // reads outline.json, so it must load after the chapters

const OUT = path.join(__dirname, "E0_Project_Report.docx");
const LINE = 360; // 1.5 line spacing

const para = (id, name, run, p) => ({ id, name, basedOn: "Normal", next: "Body", quickFormat: true, run, paragraph: p });

const styles = {
  default: { document: { run: { font: FONT, size: 24 } } },
  paragraphStyles: [
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Body", quickFormat: true,
      run: { size: 32, bold: true, font: FONT, color: "000000" },
      paragraph: { alignment: d.AlignmentType.CENTER, pageBreakBefore: true, spacing: { before: 240, after: 360 }, outlineLevel: 0 } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Body", quickFormat: true,
      run: { size: 26, bold: true, font: FONT, color: "000000" },
      paragraph: { keepNext: true, spacing: { before: 300, after: 160 }, outlineLevel: 1 } },
    { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Body", quickFormat: true,
      run: { size: 24, bold: true, font: FONT, color: "000000" },
      paragraph: { keepNext: true, spacing: { before: 240, after: 120 }, outlineLevel: 2 } },
    para("Body", "Body Text E0", { size: 24 }, { alignment: d.AlignmentType.JUSTIFIED, indent: { firstLine: 720 }, spacing: { line: LINE, after: 160 } }),
    para("BodyNoIndent", "Body No Indent", { size: 24 }, { alignment: d.AlignmentType.JUSTIFIED, spacing: { line: LINE, after: 100 } }),
    para("FrontTitle", "Front Title", { size: 28, bold: true }, { alignment: d.AlignmentType.CENTER, spacing: { before: 240, after: 360 } }),
    para("FrontTitleToc", "Front Title TOC", { size: 28, bold: true }, { alignment: d.AlignmentType.CENTER, spacing: { before: 240, after: 360 } }),
    para("FigCaption", "Figure Caption", { size: 22, bold: true }, { alignment: d.AlignmentType.CENTER, spacing: { before: 60, after: 240 } }),
    para("TabCaption", "Table Caption", { size: 22, bold: true }, { alignment: d.AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 100 } }),
    para("TableText", "Table Text", { size: 21 }, { spacing: { before: 0, after: 0 } }),
    para("Code", "Code Listing", { size: 18, font: "Consolas" }, { spacing: { before: 0, after: 0 }, indent: { left: 360 } }),
    para("Reference", "Reference", { size: 23 }, { alignment: d.AlignmentType.JUSTIFIED, indent: { left: 567, hanging: 567 }, tabStops: [{ type: d.TabStopType.LEFT, position: 567 }], spacing: { line: 312, after: 160 } }),
  ],
};

const numbering = { config: [{ reference: "bullets", levels: [{ level: 0, format: d.LevelFormat.BULLET, text: "•", alignment: d.AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] };

const page = { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 2160, right: 1440, header: 708, footer: 708 } };
const footer = new d.Footer({ children: [new d.Paragraph({ alignment: d.AlignmentType.CENTER, children: [new d.TextRun({ children: [d.PageNumber.CURRENT], size: 22 })] })] });
const emptyFooter = new d.Footer({ children: [new d.Paragraph({ children: [] })] });

const doc = new d.Document({
  creator: "Parupudi Sridatta Bharadwaj",
  title: front.TITLE,
  description: "Project report for the E0 baseline model",
  styles, numbering,
  sections: [
    { properties: { page: { ...page, pageNumbers: { start: 1, formatType: d.NumberFormat.LOWER_ROMAN } } }, footers: { default: emptyFooter }, children: front.cover },
    { properties: { type: d.SectionType.NEXT_PAGE, page: { ...page, pageNumbers: { start: 2, formatType: d.NumberFormat.LOWER_ROMAN } } }, footers: { default: footer }, children: front.front },
    { properties: { type: d.SectionType.NEXT_PAGE, page: { ...page, pageNumbers: { start: 1, formatType: d.NumberFormat.DECIMAL } } }, footers: { default: footer },
      children: [...ch1, ...ch2, ...ch3, ...ch4, ...ch5, ...ch6, ...references] },
  ],
});

d.Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, (buf.length / 1e6).toFixed(2), "MB"); });
