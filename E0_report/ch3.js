// Chapter III: System analysis and methodology.
const { P, H1, H2, H3, Fig, Table } = require("./lib");

module.exports = [
  H1("III. SYSTEM ANALYSIS AND METHODOLOGY"),
  H2("3.1. System Architecture"),
  ...Fig("arch", "fig_architecture.png", 6.2, "System architecture of the E0 ovarian ultrasound classification pipeline"),
  P("The system is a staged pipeline that turns raw ultrasound frames into a calibrated malignancy decision and an honest estimate of how well that decision performs. As shown in {fig:arch}, it consists of five layers: data, preprocessing, splitting, learning and evaluation. Each layer reads the output of the previous one from disk and writes its own results, so any stage can be re-run or inspected on its own."),
  P("The **data layer** holds the 1,469 OTU_2D images and maps the eight diagnostic categories to a binary label, with high-grade serous carcinoma as the positive (malignant) class. The **preprocessing layer** removes on-screen calipers, measurement tables and text, crops each frame to the ultrasound fan, letterboxes it to 224 × 224 pixels and uses perceptual hashing to remove near-duplicate frames and group similar ones. The **splitting layer** assigns every image to one of five folds with a grouped, stratified procedure, and defines for every run which fold is used for training, validation and testing."),
  P("The **learning layer** fine-tunes an ImageNet-pretrained MobileNetV3-Large network in two stages, first training only the new classification head and then fine-tuning the upper half of the backbone. The **evaluation layer** keeps the checkpoint with the best validation AUC, fixes a decision threshold on the validation fold, evaluates the untouched test fold, pools the five test folds into one overall result with confidence intervals, and produces Grad-CAM heat maps to show what the network looks at."),

  H2("3.2. Technology Stack"),
  P("The system is written entirely in Python and relies on widely used open-source libraries for deep learning, image processing and evaluation, as summarised in {tab:tech}. Training ran on a consumer NVIDIA GPU inside the Windows Subsystem for Linux (WSL 2), which gives a full Linux CUDA environment on a Windows workstation."),
  ...Table("tech", "Technology Stack Summary", ["Component", "Tools", "Purpose"], [
    ["Programming language", "Python 3.12", "Core development environment"],
    ["Deep-learning framework", "TensorFlow 2.20 / Keras", "Model definition, training, checkpointing"],
    ["Pretrained backbone", "Keras Applications MobileNetV3-Large", "ImageNet-initialised feature extractor"],
    ["Image processing", "OpenCV, Pillow, imagehash", "Overlay removal, cropping, resizing, perceptual hashing"],
    ["Data handling", "NumPy, Pandas", "Arrays, fold tables, logs and prediction files"],
    ["Evaluation", "scikit-learn, SciPy", "ROC/PR metrics, thresholds, grouped folds, clustering"],
    ["Visualisation", "Matplotlib", "Training curves, ROC curves, confusion matrices, Grad-CAM"],
    ["Execution environment", "WSL 2 (Ubuntu 24.04), CUDA, NVIDIA GTX 1650", "GPU-accelerated training"],
  ], [2.2, 3.2, 3.6], { leftCols: [0, 1, 2] }),

  H3("3.2.1. Python Environment"),
  P("Python was chosen because nearly every modern deep-learning and image-analysis library is available for it, and because it allows the whole pipeline, from cleaning to plotting, to be written in one language. All dependencies are installed in a dedicated virtual environment inside WSL, which isolates the project from system packages and keeps library versions fixed. The training script records its full argument list and the GPU name with every result, so each output file can be traced back to the exact configuration that produced it."),
  H3("3.2.2. TensorFlow and Keras"),
  P("TensorFlow with its high-level Keras API is used to build and train the network. Keras Applications provides MobileNetV3-Large with ImageNet weights and an in-built input rescaling layer, so raw 0–255 pixel values can be passed straight to the model. The tf.data API builds the input pipeline: images are decoded, cached in memory, shuffled, batched, augmented and prefetched, which keeps the GPU busy and makes an epoch take only a few seconds. Keras callbacks are used for checkpointing on validation AUC, for restoring the best weights at the end of each stage, and for a custom per-epoch logger."),
  H3("3.2.3. Image Processing Libraries"),
  P("OpenCV performs the image cleaning: detecting coloured overlay pixels, finding connected components, inpainting small annotations, locating the ultrasound fan as the largest bright region and cropping to it. Pillow reads and writes PNG files, and the imagehash library computes 64-bit perceptual hashes (pHash) that are used to measure how visually similar two frames are. SciPy’s hierarchical clustering turns the pairwise hash distances into similarity groups."),
  H3("3.2.4. Evaluation and Visualisation Libraries"),
  P("scikit-learn provides the evaluation functions (ROC curve, ROC-AUC, average precision) and the StratifiedGroupKFold splitter used to build the folds. Pandas stores the fold assignment, the per-epoch training logs and the per-image predictions as CSV files, and JSON files store the final metrics. Matplotlib draws all training curves, ROC curves, confusion matrices and Grad-CAM overlays, and the plots are regenerated after every epoch so training can be watched live."),

  H2("3.3. Dataset Description"),
  P("The project uses OTU_2D, the two-dimensional B-mode subset of the MMOTU dataset [4], collected at Beijing Shijitan Hospital. It contains 1,469 still transvaginal ultrasound images in eight categories, together with lesion segmentation masks. Image sizes vary from 302 to 1,135 pixels wide and 266 to 794 pixels high. The dataset does not include patient identifiers. For this project only high-grade serous carcinoma is treated as malignant; the other seven categories are benign lesions or normal ovaries. {tab:dataset} lists the number of images per category before and after cleaning, and {fig:samples} shows one cleaned example of each."),
  ...Table("dataset", "Summary of the Dataset Used", ["Category", "Label", "Raw", "Dropped", "Kept"], [
    ["Chocolate cyst (endometrioma)", "Benign", "336", "30", "306"],
    ["Teratoma", "Benign", "336", "32", "304"],
    ["Normal ovary", "Benign", "267", "24", "243"],
    ["Serous cystadenoma", "Benign", "219", "20", "199"],
    ["Mucinous cystadenoma", "Benign", "104", "8", "96"],
    ["Theca cell tumour", "Benign", "88", "6", "82"],
    ["Simple cyst", "Benign", "66", "3", "63"],
    ["**High-grade serous carcinoma**", "**Malignant**", "**53**", "**3**", "**50**"],
    ["**Total**", "", "**1,469**", "**126**", "**1,343**"],
  ], [3.6, 1.4, 1.2, 1.2, 1.2]),
  ...Fig("samples", "fig_samples.png", 6.0, "One cleaned 224 × 224 example from each of the eight categories"),
  P("Of the 126 dropped images, 110 were near-duplicate frames and 16 were near-duplicates whose label disagreed with their twin, which suggests a labelling error in one of the two. The cleaned set has **1,293 benign and 50 malignant images**, a positive rate of only 3.7% ({fig:classdist}). The 50 malignant images fall into about 46 visually distinct groups, so the number of truly independent malignant cases is smaller than 50, and the number of distinct patients may be smaller still."),
  ...Fig("classdist", "fig_class_dist.png", 5.6, "Class distribution of the cleaned dataset"),

  H2("3.4. Model Selection"),
  P("The choice of backbone network was driven by three requirements: the network must be available with reliable ImageNet weights, it must be small enough not to overfit 1,300 training images too badly, and it must train quickly on a 4 GB consumer GPU so that the full five-fold experiment can be repeated. {tab:backbones} compares the candidate architectures using the approximate figures reported in their original papers."),
  ...Table("backbones", "Comparison of Candidate Backbone Networks (ImageNet, 224 × 224 input)", ["Network", "Parameters", "Multiply-adds", "ImageNet top-1", "Remark"], [
    ["ResNet-50 [11]", "25.6 M", "≈ 4.1 B", "≈ 76%", "Large; high overfitting risk on 1.3k images"],
    ["EfficientNet-B0 [15]", "5.3 M", "≈ 0.39 B", "≈ 77%", "Accurate but slower to fine-tune"],
    ["MobileNetV2 [13]", "3.4 M", "≈ 0.30 B", "≈ 72%", "Light; lower accuracy"],
    ["**MobileNetV3-Large [14]**", "**5.4 M**", "**≈ 0.22 B**", "**≈ 75%**", "**Selected for E0**"],
    ["MobileNetV3-Small [14]", "2.5 M", "≈ 0.06 B", "≈ 67%", "Lightest; candidate for future work"],
  ], [2.5, 1.5, 1.5, 1.5, 2.6], { leftCols: [0, 4] }),
  P("MobileNetV3-Large gives the best balance. Its ImageNet accuracy is close to ResNet-50 while it needs about one-fifth of the parameters and a twentieth of the computation. Without its 1,000-class ImageNet classifier the backbone has about **3.0 million parameters**, and the complete E0 model has 2,997,313 parameters. Its squeeze-and-excitation blocks [29] let it re-weight feature channels, which is useful for picking out subtle texture differences in ultrasound. A full five-fold run of E0 takes under an hour on the available GPU. MobileNetV3-Small, the lightest model in the table, is kept in reserve: once the baseline is established, training a lighter model of this kind is a natural next step (Section 6.3)."),

  H2("3.5. Methodology Overview"),
  P("The overall methodology is shown in {fig:method}. It starts with acquiring the raw images and cleaning them, then builds the grouped folds once and stores them in a file so every experiment uses exactly the same split. For each of the five folds, the model is trained with class weights and augmentation in two stages, the best checkpoint and the decision threshold are chosen on the validation fold, and the predictions on the test fold are saved. After all five folds have run, the test predictions are pooled and summarised. Finally, a deployable model is trained once on folds 0–3, with fold 4 used for checkpoint and threshold selection."),
  ...Fig("method", "fig_methodology.png", 6.2, "Methodology overview of the E0 training and evaluation procedure"),
  P("Two principles guide every step. The first is that **the test fold is never used for any decision**: checkpoint selection, restoration of the best weights and the threshold all use the validation fold, and test metrics are logged for monitoring only. The second is **reproducibility**: the folds are fixed in a CSV file, every random seed is set, and the complete configuration is saved with each result."),
];
