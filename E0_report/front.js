// Cover page and front matter (declaration, abstract, acknowledgement, contents lists, abbreviations).
const fs = require("fs");
const path = require("path");
const { d, runs, P, Blank, PageBreak, Img, TEXT_W } = require("./lib");

const TITLE = "DEEP TRANSFER LEARNING FOR MALIGNANCY DETECTION IN OVARIAN ULTRASOUND IMAGES USING MOBILENETV3";
const NAME = "PARUPUDI SRIDATTA BHARADWAJ";
const REG = "CH.SC.U4CSE24234";
const GUIDE = "Dr. Nirmalraj J";
const C = d.AlignmentType.CENTER;

const line = (text, { size = 24, bold = false, italics = false, before = 0, after = 120 } = {}) =>
  new d.Paragraph({ alignment: C, spacing: { before, after, line: 300 }, children: [new d.TextRun({ text, size, bold, italics })] });

const cover = [
  ...Blank(1),
  line(TITLE, { size: 30, bold: true, after: 360 }),
  line("A PROJECT REPORT", { size: 26, bold: true, after: 360 }),
  line("Submitted By", { italics: true, after: 60 }),
  line(NAME, { bold: true, after: 60 }),
  line(`[Reg. No. ${REG}]`, { bold: true, after: 360 }),
  line("in partial fulfillment for the award of the degree of", { italics: true, after: 360 }),
  line("BACHELOR OF TECHNOLOGY IN COMPUTER SCIENCE AND ENGINEERING", { size: 28, bold: true, after: 480 }),
  line("Under the guidance of", { italics: true, after: 60 }),
  line(GUIDE, { bold: true, after: 360 }),
  line("Submitted to", { italics: true, after: 120 }),
  Img("logo.png", 3.0, { keepNext: false }),
  line("AMRITA VISHWA VIDYAPEETHAM", { bold: true, before: 360, after: 40 }),
  line("AMRITA SCHOOL OF COMPUTING", { bold: true, after: 40 }),
  line("CHENNAI – 601103", { bold: true, after: 360 }),
  line("October 2026", { bold: true }),
];

const title = (text) => new d.Paragraph({ style: "FrontTitle", children: [new d.TextRun(text)] });
const titleToc = (text) => new d.Paragraph({ style: "FrontTitleToc", children: [new d.TextRun(text)] });

// Logo + "SCHOOL OF COMPUTING / CHENNAI" header used on the declaration page, as in the sample.
const none = { style: d.BorderStyle.NONE, size: 0, color: "FFFFFF" };
const nb = { top: none, bottom: none, left: none, right: none };
const schoolHeader = new d.Table({
  width: { size: 5600, type: d.WidthType.DXA }, columnWidths: [3300, 2300], alignment: C,
  borders: { ...nb, insideHorizontal: none, insideVertical: none },
  rows: [new d.TableRow({ children: [
    new d.TableCell({ borders: nb, width: { size: 3300, type: d.WidthType.DXA }, verticalAlign: d.VerticalAlign.CENTER, children: [Img("logo.png", 2.2, { keepNext: false, spacing: { before: 0, after: 0 } })] }),
    new d.TableCell({ borders: nb, width: { size: 2300, type: d.WidthType.DXA }, verticalAlign: d.VerticalAlign.CENTER,
      children: ["SCHOOL OF", "COMPUTING", "CHENNAI"].map((t) => new d.Paragraph({ spacing: { after: 0 }, children: [new d.TextRun({ text: t, font: "Arial", size: 26, color: "8B1E3F" })] })) }),
  ] })],
});

const signature = (lines) => lines.map((t) => new d.Paragraph({ alignment: d.AlignmentType.RIGHT, spacing: { after: 40 }, children: [new d.TextRun({ text: t, bold: true })] }));

const declaration = [
  schoolHeader,
  ...Blank(1),
  title("DECLARATION BY THE CANDIDATE"),
  P(`I declare that the report entitled **${TITLE}** submitted by me for the degree of Bachelor of Technology is the record of the project work carried out by me under the guidance of **${GUIDE}** and this work has not formed the basis for the award of any degree, diploma, associateship, fellowship, titled in this or any other University or other similar institution of higher learning.`),
  ...Blank(5),
  ...signature(["(Signature)", NAME, `[Reg. No. ${REG}]`]),
  PageBreak(),
];

const abstract = [
  title("ABSTRACT"),
  P("Ovarian cancer is among the deadliest gynaecological cancers, largely because it is often found late, and transvaginal ultrasound is the first imaging test used to decide whether an adnexal mass is benign or malignant. This project develops and rigorously evaluates a deep-learning baseline, named **E0**, that classifies a single two-dimensional B-mode ovarian ultrasound image as benign or malignant."),
  P("The work uses the public OTU_2D dataset (1,469 images in eight tumour categories), in which only high-grade serous carcinoma is malignant. Preprocessing removes on-screen annotations, crops and letterboxes every frame to 224 × 224 pixels and removes near-duplicate frames, leaving 1,343 images of which only 50 (3.7%) are malignant. The data is divided into five grouped, stratified cross-validation folds so that near-identical frames never appear on both sides of a split."),
  P("The E0 model is an ImageNet-pretrained MobileNetV3-Large network with a global-average-pooling, dropout and single sigmoid output head, trained with inverse-frequency class weights and light data augmentation. Training runs in two stages: 30 epochs on the classification head with the backbone frozen, followed by 90 epochs of fine-tuning on the upper half of the backbone. For every fold the checkpoint with the best validation AUC is kept and the decision threshold is chosen on the validation fold using Youden’s J statistic, so the test fold is never used for any decision."),
  P("Pooled over all five folds, E0 reached an out-of-fold ROC-AUC of **0.819** (fold mean 0.841 ± 0.056) and detected **34 of the 50** malignant images (sensitivity 68%, 95% CI 54–79%) at a specificity of **87.9%**. An independent earlier run of the same configuration gave 35/50 and a pooled AUC of 0.823, confirming that the pipeline is reproducible. Learning curves and Grad-CAM analysis point to the small number of malignant examples as the main limitation. The baseline provides a reproducible reference point for future work on lighter and more robust models."),
  new d.Paragraph({ style: "BodyNoIndent", spacing: { before: 240 }, children: runs("**Keywords:** Deep Learning, Transfer Learning, Ovarian Tumour Classification, Ultrasound Imaging, MobileNetV3, Class Imbalance, Grouped Cross-Validation, Grad-CAM, Medical Image Analysis") }),
  PageBreak(),
];

const acknowledgement = [
  title("ACKNOWLEDGEMENT"),
  P("This project work would not have been possible without the contribution of many people. It gives me immense pleasure to express my profound gratitude to our honourable Chancellor for her blessings and for being a constant source of inspiration. I am indebted to the Director of Amrita School of Computing for providing all the facilities and support that made a valuable learning experience possible."),
  P("I register my special thanks to the Principal, the Chairperson and the Program Chair of the Department of Computer Science and Engineering for the support given to me in the successful conduct of this project. I wish to express my sincere gratitude to my project guide for the inspiring guidance, personal involvement and constant encouragement throughout the course of this work."),
  P("I am grateful to the Project Coordinator, the Review Panel Members and the entire faculty of the Department of Computer Science and Engineering for their constructive criticism and valuable suggestions, which have been a rich source of improvement for this work. Finally, I thank my family and friends for their patience and encouragement."),
  ...Blank(3),
  ...signature([NAME, `[Reg. No. ${REG}]`]),
  PageBreak(),
];

// Static contents lists. Page numbers come from toc_pages.json, which make_toc.py reads off the rendered PDF
// (build -> render -> make_toc -> build again). Without it, every entry gets "?" so the layout is still the same.
const pagesFile = path.join(__dirname, "toc_pages.json");
const pages = fs.existsSync(pagesFile) ? JSON.parse(fs.readFileSync(pagesFile, "utf8")) : null;
const outline = JSON.parse(fs.readFileSync(path.join(__dirname, "outline.json"), "utf8"));
const entry = (text, page, { level = 1, bold = false } = {}) => new d.Paragraph({
  spacing: { after: 60, line: 276 }, indent: { left: (level - 1) * 400, hanging: 0 },
  tabStops: [{ type: d.TabStopType.RIGHT, position: TEXT_W, leader: "dot" }],
  children: [new d.TextRun({ text, bold }), new d.TextRun({ text: `\t${page}`, bold })],
});
const pageOf = (text) => (pages && pages[text]) || "?";
const FRONT_LISTS = ["LIST OF TABLES", "LIST OF FIGURES", "LIST OF SYMBOLS AND ABBREVIATIONS"];

const toc = [
  title("TABLE OF CONTENTS"),
  ...FRONT_LISTS.map((t) => entry(t, pageOf(t), { bold: true })),
  ...outline.filter((o) => o.kind === "toc").map((o) => entry(o.text, pageOf(o.text), { level: o.level, bold: o.level === 1 })),
  PageBreak(),
  titleToc("LIST OF TABLES"),
  ...outline.filter((o) => o.kind === "tab").map((o) => entry(o.text, pageOf(o.text))),
  PageBreak(),
  titleToc("LIST OF FIGURES"),
  ...outline.filter((o) => o.kind === "fig").map((o) => entry(o.text, pageOf(o.text))),
  PageBreak(),
];

const abbr = [
  ["AUC", "Area Under the Curve"], ["BCE", "Binary Cross-Entropy"], ["CI", "Confidence Interval"],
  ["CNN", "Convolutional Neural Network"], ["CPU / GPU", "Central / Graphics Processing Unit"], ["CV", "Cross-Validation"],
  ["FN / FP", "False Negative / False Positive"], ["Grad-CAM", "Gradient-weighted Class Activation Mapping"],
  ["HGSC", "High-Grade Serous Carcinoma"], ["IOTA", "International Ovarian Tumour Analysis"],
  ["J", "Youden’s J statistic (sensitivity + specificity − 1)"], ["lr", "Learning rate"],
  ["MMOTU", "Multi-Modality Ovarian Tumour Ultrasound dataset"], ["NPV", "Negative Predictive Value"],
  ["O-RADS", "Ovarian-Adnexal Reporting and Data System"], ["OTU_2D", "Two-dimensional ovarian tumour ultrasound subset of MMOTU"],
  ["pHash", "Perceptual Hash"], ["PPV", "Positive Predictive Value (precision)"], ["PR-AUC", "Area under the Precision–Recall curve"],
  ["ROC", "Receiver Operating Characteristic"], ["SE", "Squeeze-and-Excitation"],
  ["TN / TP", "True Negative / True Positive"], ["WSL", "Windows Subsystem for Linux"],
];
const abbrTable = new d.Table({
  width: { size: TEXT_W, type: d.WidthType.DXA }, columnWidths: [2200, TEXT_W - 2200],
  borders: { ...nb, insideHorizontal: none, insideVertical: none },
  rows: abbr.map(([a, b]) => new d.TableRow({ children: [
    new d.TableCell({ borders: nb, width: { size: 2200, type: d.WidthType.DXA }, children: [new d.Paragraph({ spacing: { after: 80 }, children: [new d.TextRun({ text: a, bold: true })] })] }),
    new d.TableCell({ borders: nb, width: { size: TEXT_W - 2200, type: d.WidthType.DXA }, children: [new d.Paragraph({ spacing: { after: 80 }, children: [new d.TextRun(b)] })] }),
  ] })),
});
const abbreviations = [titleToc("LIST OF SYMBOLS AND ABBREVIATIONS"), abbrTable];

module.exports = { TITLE, NAME, REG, cover, front: [...declaration, ...abstract, ...acknowledgement, ...toc, ...abbreviations] };
