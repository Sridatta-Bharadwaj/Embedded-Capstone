// Chapter IV: System design and implementation.
const { d, P, H1, H2, Bullets, Fig, Table, Eq, mr, frac, sub } = require("./lib");

const code = (lines) => lines.map((t, i) => new d.Paragraph({ style: "Code", keepNext: i < lines.length - 1, children: [new d.TextRun(t || " ")] }));

module.exports = [
  H1("IV. SYSTEM DESIGN AND IMPLEMENTATION"),
  H2("4.1. Overall System Design"),
  P("The system is built as a set of small, single-purpose programs that communicate through files. The cleaning and splitting stages run once and write a processed image folder and a fold-assignment table (cv_folds.csv). The training program, train_e0.py, runs one fold at a time in its own process, which keeps GPU memory clean between folds and means that a crash in one fold does not affect the others. A summary program, summarize_cv.py, reads the five sets of test predictions and produces the pooled result. A shell script, run_e0.sh, runs all five folds, the summary and the final model in sequence, writes everything to a log, and skips any fold that has already finished, so an interrupted run can be resumed."),
  P("Every training run writes the same set of artefacts: a per-epoch log (epoch_log.csv) with training, validation and test metrics, one plot per metric redrawn after every epoch, the kept checkpoint (best.keras), the last-epoch weights (final.keras), the chosen threshold (threshold.json), per-image predictions for the validation and test folds, and a metrics.json file that records the full configuration and the results."),

  H2("4.2. Data Cleaning and Preprocessing"),
  P("Raw OTU_2D frames contain much more than the ultrasound image itself. Inside the scan fan there are coloured calipers, dotted measurement lines and measurement tables; outside it there is machine text, scale bars and colour bars. If these were left in place, the network could learn, for example, that frames with many measurements are more likely to be malignant. The cleaning stage therefore applies the following steps to every frame:"),
  ...Bullets([
    "**Overlay detection:** a pixel is treated as coloured if the spread between its largest and smallest RGB value exceeds 40. Coloured connected components smaller than 0.4% of the frame are marked as annotations; larger coloured regions, which are colour-Doppler flow, are left alone.",
    "**Inpainting:** the marked annotation pixels are filled in from surrounding tissue with the Telea inpainting algorithm, and the frame is converted to grayscale.",
    "**Fan isolation:** the largest bright connected region is taken as the scan fan, filled using its convex hull; everything outside it is set to black and the frame is cropped to the fan’s bounding box.",
    "**Letterboxing:** the cropped frame is scaled to fit inside 224 × 224 pixels while keeping its aspect ratio, and padded with black, so anatomy is never stretched. The single gray channel is repeated three times to match the network’s RGB input.",
  ]),
  ...Fig("cleaning", "fig_cleaning.png", 6.2, "Raw OTU_2D frames (top) and the corresponding cleaned 224 × 224 images (bottom)"),
  P("{fig:cleaning} shows the effect of cleaning on eight frames. Calipers and text are removed and the fan fills the image. Some residual marks remain: where a large measurement table was inpainted, a uniform grey rectangle is left in 139 images, and a few faint caliper traces survive. These residues are discussed again in the Grad-CAM analysis of Chapter V."),
  P("**Near-duplicate removal.** For each cleaned frame a 64-bit perceptual hash is computed. Two frames whose hashes differ in at most 4 bits are considered near-identical and only one is kept; when the two copies carried different labels, both were dropped. This removed 126 images and left 1,343."),

  H2("4.3. Leakage-aware Cross-validation Design"),
  P("OTU_2D has no patient identifiers, so several frames of the same lesion may exist even after de-duplication. To approximate patient-level separation, frames are clustered with complete-linkage hierarchical clustering on their pHash distances: every pair of frames inside a group differs by at most 10 bits. This produces 1,117 groups. Complete linkage was chosen over simple chaining because chaining linked dissimilar frames into one very large group, which made balanced folds impossible."),
  P("The groups are then divided into five folds with scikit-learn’s StratifiedGroupKFold, stratified on the eight original categories so each fold has a similar mix of lesion types. Fifty random seeds were tried and the most balanced assignment, measured by how evenly the images and the malignant images are spread, was kept and saved to cv_folds.csv. It was never re-randomised. A check over the saved file confirms that **none of the 1,117 groups appears in more than one fold**."),
  P("Each of the five runs uses one fold for testing, the next fold for validation and the remaining three for training, as shown in {fig:folds}. Across the five runs, every image, and therefore every one of the 50 malignant images, is tested exactly once by a model that did not see it during training. {tab:folds} gives the composition of each fold."),
  ...Fig("folds", "fig_folds.png", 5.6, "Rotation of training, validation and test folds across the five cross-validation runs"),
  ...Table("folds", "Cross-validation Fold Composition", ["Test fold", "Test images (malignant)", "Validation fold", "Training images (malignant)", "Malignant class weight"], [
    ["0", "266 (10)", "1", "812 (31)", "13.10"],
    ["1", "265 (9)", "2", "808 (29)", "13.93"],
    ["2", "270 (12)", "3", "806 (29)", "13.90"],
    ["3", "267 (9)", "4", "801 (31)", "12.92"],
    ["4", "275 (10)", "0", "802 (30)", "13.37"],
  ], [1.2, 2.0, 1.5, 2.3, 1.9]),

  H2("4.4. Input Pipeline and Data Augmentation"),
  P("Images are loaded with tf.data, decoded from PNG, cached in memory and served in batches of 32. The training stream is shuffled every epoch and repeated, and each batch is augmented on the fly. The validation and test streams are never augmented. The E0 augmentation is intentionally mild, so that it does not create unrealistic ultrasound images:"),
  ...Bullets([
    "random horizontal flip;",
    "random rotation of up to ±15°, with the exposed corners filled with black;",
    "random brightness change of up to ±10% of the pixel range;",
    "random contrast change of up to ±10%.",
  ]),
  P("After augmentation, pixel values are clipped back to the range 0–255. The network’s own input layer then rescales them as MobileNetV3 expects."),

  H2("4.5. Handling Class Imbalance"),
  P("With about 26 benign images for every malignant one, an unweighted loss would be dominated by the benign class. E0 uses inverse-frequency class weights, computed separately for each training set as in Eq. (4.1), where N is the number of training images and N_c the number of images in class c:"),
  Eq([sub("w", "c"), mr(" = "), frac([mr("N")], [mr("2 · "), sub("N", "c")])], "(4.1)"),
  P("For the fold-0 training set (812 images, 31 malignant) this gives a weight of 0.52 for benign and 13.10 for malignant images, so each malignant image counts about 25 times as much as a benign one. The weights multiply the binary cross-entropy loss of each image, as in Eq. (4.2), where y_i is the true label and p_i the predicted probability:"),
  Eq([mr("L = −"), frac([mr("1")], [mr("N")]), mr(" Σ "), sub("w", "y"), mr(" [ y log p + (1 − y) log(1 − p) ]")], "(4.2)"),

  H2("4.6. Network Architecture"),
  P("The E0 network ({fig:model}) takes a 224 × 224 × 3 image. The MobileNetV3-Large backbone, loaded with ImageNet weights and without its original classifier, turns the image into a 7 × 7 × 960 feature map. Global average pooling reduces it to a 960-dimensional vector, dropout with rate 0.3 [24] is applied for regularisation, and a single dense unit with a sigmoid activation outputs the probability that the image is malignant. The backbone is always called in inference mode, so its batch-normalisation statistics [25] stay fixed even when its weights are being fine-tuned; with so little data, updating these statistics would make training unstable."),
  ...Fig("model", "fig_model.png", 6.2, "E0 network architecture: MobileNetV3-Large backbone with a pooled, regularised sigmoid head"),
  P("The core of the model definition in train_e0.py is shown below."),
  ...code([
    "backbone = tf.keras.applications.MobileNetV3Large(",
    "    weights=\"imagenet\", include_top=False,",
    "    input_shape=(224, 224, 3), include_preprocessing=True)",
    "backbone.trainable = False",
    "inp = tf.keras.Input((224, 224, 3), name=\"image\")",
    "x = backbone(inp, training=False)      # BatchNorm stays in inference mode",
    "x = tf.keras.layers.GlobalAveragePooling2D()(x)",
    "x = tf.keras.layers.Dropout(0.3)(x)",
    "out = tf.keras.layers.Dense(1, activation=\"sigmoid\", name=\"malignancy\")(x)",
    "model = tf.keras.Model(inp, out)",
  ]),

  H2("4.7. Two-stage Training Strategy"),
  P("Training a newly initialised head together with a pretrained backbone from the start can damage the pretrained features, because the large, random gradients from the head flow back into the backbone. E0 therefore trains in two stages."),
  P("**Stage A (head training)** freezes the whole backbone and trains only the dense head for 30 epochs with the Adam optimiser [23] at a learning rate of 1 × 10⁻³. At the end of stage A the weights with the lowest validation loss are restored. **Stage B (fine-tuning)** then unfreezes the top 50% of the backbone’s layers, except the batch-normalisation layers, and continues for 90 epochs at a ten-times smaller learning rate of 1 × 10⁻⁴. There is no early stopping: all 120 epochs are run, and the epoch with the best validation AUC across both stages is kept. {tab:hparams} lists every setting."),
  ...Table("hparams", "E0 Training Configuration", ["Setting", "Value"], [
    ["Backbone", "MobileNetV3-Large, ImageNet weights, no top"],
    ["Input", "224 × 224, grayscale replicated to 3 channels"],
    ["Head", "Global average pooling → Dropout 0.3 → Dense(1, sigmoid)"],
    ["Loss", "Binary cross-entropy with inverse-frequency class weights"],
    ["Optimiser", "Adam"],
    ["Stage A", "Backbone frozen, 30 epochs, lr 1 × 10⁻³, restore best validation loss"],
    ["Stage B", "Top 50% of backbone unfrozen (BatchNorm frozen), 90 epochs, lr 1 × 10⁻⁴"],
    ["Batch size", "32"],
    ["Augmentation", "Flip, rotation ±15°, brightness ±10%, contrast ±10%"],
    ["Checkpoint selection", "Best validation ROC-AUC over all 120 epochs"],
    ["Decision threshold", "Youden’s J on the validation fold (lowest threshold among ties)"],
    ["Random seed", "42 + fold index"],
  ], [3, 6], { leftCols: [0, 1] }),

  H2("4.8. Checkpoint Selection and Decision Threshold"),
  P("The sigmoid output is a probability, but a clinical decision needs a yes/no answer, so a threshold t must be chosen. Because the data is so imbalanced and the class weights shift the output probabilities, the conventional value of 0.5 is not appropriate. E0 chooses the threshold on the validation fold using Youden’s J statistic [18], defined in Eq. (4.3), and takes the threshold at which J is largest:"),
  Eq([mr("J(t) = Sensitivity(t) + Specificity(t) − 1")], "(4.3)"),
  P("When several thresholds give the same maximum J, the lowest is taken, which favours sensitivity, since missing a cancer is worse than a false alarm. The kept checkpoint is reloaded, its predictions on the validation fold are used to fix t, and only then is the test fold scored with that fixed threshold. The test fold therefore plays no part in choosing either the weights or the threshold."),

  H2("4.9. Evaluation Metrics"),
  P("Writing TP, FN, TN and FP for the numbers of true positives, false negatives, true negatives and false positives, the main threshold-dependent metrics are:"),
  Eq([mr("Sensitivity = "), frac([mr("TP")], [mr("TP + FN")]), mr(",   Specificity = "), frac([mr("TN")], [mr("TN + FP")])], "(4.4)"),
  Eq([mr("PPV = "), frac([mr("TP")], [mr("TP + FP")]), mr(",   NPV = "), frac([mr("TN")], [mr("TN + FN")])], "(4.5)"),
  P("Sensitivity (recall) is the fraction of malignant images that are detected and is the most important metric for this application. Specificity is the fraction of benign images correctly cleared. The positive predictive value (PPV, or precision) tells how often an alarm is really cancer, which depends strongly on how rare cancer is. Two threshold-free metrics summarise ranking quality: the **ROC-AUC**, the probability that a randomly chosen malignant image receives a higher score than a randomly chosen benign one, and the **PR-AUC** (average precision), which is more sensitive to false alarms under imbalance [17]."),
  P("The headline numbers are **pooled out-of-fold** metrics: the 1,343 test predictions from the five folds, each made at its own fold’s threshold, are combined and evaluated together. With only 50 malignant images, 95% confidence intervals for sensitivity and specificity are computed with the Wilson score interval [19], which behaves well for small counts and proportions near 0 or 1."),

  H2("4.10. Implementation and Software Organisation"),
  P("The implementation is organised so that the complete experiment can be reproduced with a single command. The main components are:"),
  ...Bullets([
    "**train_e0.py:** trains one fold (--mode cv --fold k) or the final model (--mode final). It refuses to run if no GPU is visible, checks that every image exists before training, logs every epoch, and writes the checkpoint, threshold, predictions and metrics.",
    "**summarize_cv.py:** pools the five test-fold prediction files, computes the pooled metrics with Wilson intervals, compares them with an earlier run, and draws the pooled ROC curve and all-fold training curves.",
    "**run_e0.sh:** runs folds 0–4, the summary and the final model in sequence inside WSL, appends all output to run.log and stops if any fold fails.",
    "**cv_folds.csv:** the fixed fold table, with the file name, category, label, similarity group and fold of every image.",
  ]),
  P("A custom Keras callback writes one row per epoch to epoch_log.csv, evaluates the test fold for monitoring, saves the latest weights so a run can be inspected at any time, and redraws six metric plots (loss, accuracy, ROC-AUC, PR-AUC, recall and precision). A second callback remembers the best weights during a stage and restores them at the end of that stage. A finished run is detected from its metrics.json file and skipped on re-run unless --force is given."),
];
