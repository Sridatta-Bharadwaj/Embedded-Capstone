// Chapter VI (Conclusions and further work) and references.
const { d, P, H1, H2, Bullets } = require("./lib");

const ch6 = [
  H1("VI. CONCLUSIONS AND SCOPE FOR FURTHER WORK"),
  H2("6.1. Conclusions"),
  P("This project designed, implemented and evaluated E0, a deep transfer-learning baseline for detecting malignancy in two-dimensional ovarian ultrasound images. The system cleans the OTU_2D images of on-screen annotations, removes near-duplicate frames, builds grouped and stratified cross-validation folds that keep near-identical frames apart, and fine-tunes an ImageNet-pretrained MobileNetV3-Large network in two stages with class weighting and light augmentation. Every model-selection decision is made on validation data only."),
  P("Under this strict protocol, E0 achieved a pooled out-of-fold ROC-AUC of 0.819 and detected 34 of the 50 malignant images (sensitivity 68%, 95% CI 54–79%) at a specificity of 87.9%. A second, independent run of the same configuration reproduced the result (AUC 0.823, 35/50). The training curves, the fold-to-fold variation and the Grad-CAM analysis consistently show that the network has enough capacity to learn the task, and in fact memorises its training data, and that the small and narrow malignant class is the main factor limiting performance. The project therefore delivers a clean, reproducible and honestly evaluated reference baseline against which future models can be compared."),

  H2("6.2. Limitations"),
  ...Bullets([
    "**Very few malignant examples.** Only 50 malignant images (about 46 independent cases) are available, so sensitivity can only be measured to within roughly ±12 percentage points, and one image changes a fold’s sensitivity by about 10 points.",
    "**Only one cancer subtype.** All malignant images are high-grade serous carcinoma; borderline tumours and other carcinomas are absent, so the model has not learned “malignant” in general.",
    "**No patient identifiers.** Patient-level separation is approximated by image-similarity grouping; frames of the same patient that look different could still fall into different folds.",
    "**Single centre, still images.** All images come from one hospital, so performance on other scanners, operators and populations is unknown, and no video or Doppler information is used.",
    "**Residual artefacts.** Inpainted measurement tables leave grey boxes in 139 images and a few caliper traces remain; Grad-CAM shows the model sometimes attends to them.",
    "**Unstable operating point.** The decision threshold is chosen on 9–12 malignant validation images per fold, which makes the specificity at that threshold vary noticeably between runs.",
  ]),

  H2("6.3. Future Enhancements"),
  P("The most immediate next step is to **train a lightweight model** that keeps the accuracy of E0 at a fraction of its size and computation. E0 already uses an efficient backbone, but MobileNetV3-Small has less than half the parameters and about a quarter of the computation of MobileNetV3-Large ({tab:backbones}), and a lighter network is also less able to memorise a small training set. The planned work is to fine-tune MobileNetV3-Small and other compact backbones on exactly the same folds and protocol so that their results are directly comparable with E0, and to use knowledge distillation [28], with E0 acting as the teacher, to transfer its knowledge into the smaller student network. A lighter model would train faster, make repeated cross-validation experiments cheaper, and give quicker predictions on ordinary computers."),
  P("Other directions for future work are:"),
  ...Bullets([
    "**More and more varied malignant data:** adding pathology-confirmed images from other centres and other cancer subtypes, and running a learning-curve experiment to confirm how much more data would help.",
    "**Stronger artefact removal:** blacking out measurement tables rather than inpainting them, and training on lesion-focused crops so the network cannot use screen artefacts as shortcuts.",
    "**Better imbalance handling:** focal loss [16], minority over-sampling and ultrasound-specific augmentation such as speckle noise and gain changes.",
    "**More stable thresholds:** choosing the operating point from pooled validation predictions or nested cross-validation, and calibrating the output probabilities.",
    "**Multi-class and multi-frame models:** predicting all eight lesion types, and combining several frames or video clips of the same lesion.",
    "**External validation:** testing the model on an independent dataset from another hospital before any clinical use is considered.",
  ]),
];

const refs = [
  "H. Sung, J. Ferlay, R. L. Siegel, M. Laversanne, I. Soerjomataram, A. Jemal and F. Bray, “Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries,” CA: A Cancer Journal for Clinicians, vol. 71, no. 3, pp. 209–249, 2021.",
  "D. Timmerman et al., “Simple ultrasound-based rules for the diagnosis of ovarian cancer,” Ultrasound in Obstetrics & Gynecology, vol. 31, no. 6, pp. 681–690, 2008.",
  "R. F. Andreotti et al., “O-RADS US Risk Stratification and Management System: A Consensus Guideline from the ACR Ovarian-Adnexal Reporting and Data System Committee,” Radiology, vol. 294, no. 1, pp. 168–185, 2020.",
  "Q. Zhao, S. Lyu, W. Bai, L. Cai, B. Liu, M. Wu, X. Sang, M. Yang and L. Chen, “A Multi-Modality Ovarian Tumor Ultrasound Image Dataset for Unsupervised Cross-Domain Semantic Segmentation,” arXiv preprint arXiv:2207.06799, 2022.",
  "F. Christiansen et al., “Ultrasound image analysis using deep neural networks for discriminating between benign and malignant ovarian tumors: comparison with expert subjective assessment,” Ultrasound in Obstetrics & Gynecology, vol. 57, no. 1, pp. 155–163, 2021.",
  "Y. Gao et al., “Deep learning-enabled pelvic ultrasound images for accurate diagnosis of ovarian cancer in China: a retrospective, multicentre, diagnostic study,” The Lancet Digital Health, vol. 4, no. 3, pp. e179–e187, 2022.",
  "G. Litjens et al., “A survey on deep learning in medical image analysis,” Medical Image Analysis, vol. 42, pp. 60–88, 2017.",
  "A. Esteva et al., “Dermatologist-level classification of skin cancer with deep neural networks,” Nature, vol. 542, pp. 115–118, 2017.",
  "M. Raghu, C. Zhang, J. Kleinberg and S. Bengio, “Transfusion: Understanding Transfer Learning for Medical Imaging,” in Advances in Neural Information Processing Systems (NeurIPS), 2019.",
  "J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2009, pp. 248–255.",
  "K. He, X. Zhang, S. Ren and J. Sun, “Deep Residual Learning for Image Recognition,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770–778.",
  "A. G. Howard et al., “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,” arXiv preprint arXiv:1704.04861, 2017.",
  "M. Sandler, A. Howard, M. Zhu, A. Zhmoginov and L.-C. Chen, “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4510–4520.",
  "A. Howard et al., “Searching for MobileNetV3,” in Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV), 2019, pp. 1314–1324.",
  "M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” in Proc. Int. Conf. Machine Learning (ICML), 2019, pp. 6105–6114.",
  "T.-Y. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, “Focal Loss for Dense Object Detection,” in Proc. IEEE Int. Conf. Computer Vision (ICCV), 2017, pp. 2980–2988.",
  "T. Saito and M. Rehmsmeier, “The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets,” PLoS ONE, vol. 10, no. 3, e0118432, 2015.",
  "W. J. Youden, “Index for rating diagnostic tests,” Cancer, vol. 3, no. 1, pp. 32–35, 1950.",
  "E. B. Wilson, “Probable Inference, the Law of Succession, and Statistical Inference,” Journal of the American Statistical Association, vol. 22, no. 158, pp. 209–212, 1927.",
  "R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh and D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” in Proc. IEEE Int. Conf. Computer Vision (ICCV), 2017, pp. 618–626.",
  "J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano and E. K. Oermann, “Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study,” PLoS Medicine, vol. 15, no. 11, e1002683, 2018.",
  "R. Geirhos et al., “Shortcut learning in deep neural networks,” Nature Machine Intelligence, vol. 2, pp. 665–673, 2020.",
  "D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” in Proc. Int. Conf. Learning Representations (ICLR), 2015.",
  "N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever and R. Salakhutdinov, “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” Journal of Machine Learning Research, vol. 15, pp. 1929–1958, 2014.",
  "S. Ioffe and C. Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,” in Proc. Int. Conf. Machine Learning (ICML), 2015, pp. 448–456.",
  "M. Abadi et al., “TensorFlow: A System for Large-Scale Machine Learning,” in Proc. 12th USENIX Symp. Operating Systems Design and Implementation (OSDI), 2016, pp. 265–283.",
  "F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.",
  "G. Hinton, O. Vinyals and J. Dean, “Distilling the Knowledge in a Neural Network,” arXiv preprint arXiv:1503.02531, 2015.",
  "J. Hu, L. Shen and G. Sun, “Squeeze-and-Excitation Networks,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2018, pp. 7132–7141.",
  "R. Kohavi, “A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection,” in Proc. Int. Joint Conf. Artificial Intelligence (IJCAI), 1995, pp. 1137–1143.",
];

const references = [
  H1("VII. REFERENCES"),
  ...refs.map((r, i) => new d.Paragraph({ style: "Reference", children: [new d.TextRun(`[${i + 1}]\t${r}`)] })),
];


module.exports = { ch6, references };
