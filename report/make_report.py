"""Build the Assignment 4 final project report (Word .docx) from the project's real results.

Run with Python 3.12 (python-docx installed):
    python report/make_report.py
Numbers come from cnn/outputs/results.json and the Webots logs; figures from cnn/outputs, docs and webots.
"""
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "cnn/outputs/results.json").read_text(encoding="utf-8"))
OUT = ROOT / "submission" / "8757HIQT_HarishAshokkumar_ARAIP_Final_Report.docx"

INK = RGBColor(0x1F, 0x29, 0x37)
ACCENT = RGBColor(0x0F, 0x76, 0x6E)
MUTED = RGBColor(0x55, 0x60, 0x6E)

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.3))
sec.top_margin, sec.bottom_margin = Cm(2.2), Cm(2.2)

styles = doc.styles
styles["Normal"].font.name = "Calibri"
styles["Normal"].font.size = Pt(11)
styles["Normal"].paragraph_format.space_after = Pt(6)
styles["Normal"].paragraph_format.line_spacing = 1.15
for name, size, colour in (("Heading 1", 16, ACCENT), ("Heading 2", 13, INK), ("Heading 3", 11.5, INK)):
    st = styles[name]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = colour
    st.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
    st.paragraph_format.space_after = Pt(4)
    st.paragraph_format.keep_with_next = True
    rpr = st.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(attr), "Calibri")

FIG = [0]
TAB = [0]


def para(text, bold_lead=None, italic=False, size=None, align=None, colour=None, after=None):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if colour:
        r.font.color.rgb = colour
    if align:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullets(items, style="List Bullet"):
    for item in items:
        p = doc.add_paragraph(style=style)
        if isinstance(item, tuple):
            p.add_run(item[0]).bold = True
            p.add_run(item[1])
        else:
            p.add_run(item)
        p.paragraph_format.space_after = Pt(2)


KEYS = {}


def figure(path, caption, width_cm=15.5, key=None):
    FIG[0] += 1
    if key:
        KEYS[key] = FIG[0]
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(ROOT / path), width=Cm(width_cm))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(f"Figure {FIG[0]}: ")
    r.bold = True
    r.font.size = Pt(9.5)
    r = c.add_run(caption)
    r.font.size = Pt(9.5)
    r.font.color.rgb = MUTED
    return FIG[0]


def shade(cell, hex_fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def table(rows, caption, widths=None, size=9.5, bold_last=False):
    TAB[0] += 1
    c = doc.add_paragraph()
    c.paragraph_format.keep_with_next = True
    r = c.add_run(f"Table {TAB[0]}: ")
    r.bold = True
    r.font.size = Pt(9.5)
    r = c.add_run(caption)
    r.font.size = Pt(9.5)
    r.font.color.rgb = MUTED
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            run = p.add_run(str(val))
            run.font.size = Pt(size)
            if i == 0 or (bold_last and i == len(rows) - 1):
                run.bold = True
            if i == 0:
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                shade(cell, "0F766E")
            elif i % 2 == 0:
                shade(cell, "F1F5F4")
            if widths:
                cell.width = Cm(widths[j])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return TAB[0]


def code(text):
    for line in text.strip("\n").splitlines():
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.left_indent = Cm(0.4)
        r = p.add_run(line if line else " ")
        r.font.name = "Consolas"
        r.font.size = Pt(8.5)
        r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def top(entry):
    label, p = entry["top3"][0]
    return label, p


OI, ORS = R["optimisation_improved"], R["optimisation_resnet9"]
R9 = R["resnet9"]
P9 = R["predictions_resnet9"]

# ======================================================================= title page (same layout as the ESD report)
def cover_line(text, size, bold=False, before=None, after=None, font="Cambria"):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(before) if before is not None else None
    p.paragraph_format.space_after = Pt(after) if after is not None else None
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    if font:
        r.font.name = font
        r._element.rPr.rFonts.set(qn("w:eastAsia"), font)
        r._element.rPr.rFonts.set(qn("w:hAnsi"), font)
    return p


doc.add_paragraph().paragraph_format.space_after = Pt(30)
logo = doc.add_paragraph()
logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
logo.paragraph_format.space_after = Pt(30)
logo.add_run().add_picture(str(ROOT / "report/assets/psb_uh_logo.png"), width=Cm(11.5))
cover_line("UNIVERSITY OF HERTFORDSHIRE", 14, after=10)
cover_line("School of Engineering and Technology, PSB Academy", 14, after=10)
cover_line("Applied Robotics and AI Project", 14, after=10)
cover_line("BEng (Hons) Robotics and Artificial Intelligence", 14, after=10)
cover_line("ASSIGNMENT 4", 19, bold=True, before=70, after=8)
cover_line("(Project Part 2 – Final Project Report)", 13, bold=True, font=None)
cover_line("From CIFAR-10 to the e-puck Camera: Training, Optimising and Deploying a CNN Object Classifier "
           "with Autonomous Colour-Block Search in Webots", 13, before=30, after=0).paragraph_format.left_indent = Cm(1.5)
doc.paragraphs[-1].paragraph_format.right_indent = Cm(1.5)
cover_line("By", 13, before=95, after=24)
cover_line("HARISH ASHOKKUMAR, 8757HIQT", 13, bold=True)
sec.different_first_page_header_footer = True
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

# project title and details at the top of page 2
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.paragraph_format.space_after = Pt(2)
r = t.add_run("From CIFAR-10 to the e-puck Camera: Training, Optimising and Deploying a CNN Object Classifier "
              "with Autonomous Colour-Block Search in Webots")
r.bold, r.font.size, r.font.color.rgb = True, Pt(15), ACCENT
para("Submitted 21 November 2026  ·  GitHub: github.com/harish93075-png/applied-robotics-ai-project-part2  ·  "
     "Trello: trello.com/b/xMSJpIeZ  ·  Webots package: 8757HIQT_HarishAshokkumar_Webots.zip",
     size=9, colour=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)

# ======================================================================= abstract + contents
doc.add_heading("Abstract", level=1)
para("Robots that search an environment must recognise what they find, yet deep networks are usually trained off the "
     "robot because they need far more computation than an embedded controller can give. This project builds that "
     "pipeline end to end. Three convolutional neural networks (CNNs) were trained on CIFAR-10 in Google Colab: the "
     f"Session 8 baseline ({pct(R['baseline_adam']['test_accuracy'])} with Adam), an improved CNN with batch "
     f"normalisation, dropout and augmentation ({pct(R['improved_adam']['test_accuracy'])}) and a ResNet-9 trained with a "
     f"one-cycle schedule ({pct(R9['test_accuracy'], 2)} test accuracy, {pct(R9['test_accuracy_tta'], 2)} with flip "
     "test-time augmentation), close to the published ResNet-110 benchmark at a fraction of the depth. The ResNet-9 "
     "classified 4 of 4 of my own photos correctly once the object was cropped to a square, and 3 of 5 objects seen "
     "through the e-puck camera in Webots, a gap explained by the difference between rendered models and real photos. "
     "Post-training int8 quantisation shrank the ResNet-9 from 25.1 MB to 6.3 MB with no loss of accuracy, and the "
     "small CNN runs in 1.3 ms per image. Running the CNN in a background thread, only on a pre-filtered region of "
     "interest and only while the robot is stopped, keeps the 32 ms control loop free; the classifier now runs live "
     "on the simulated robot at 33–41 ms per verdict. Alongside the CNN, an e-puck search controller finds red, blue "
     "and green 0.3 × 0.3 × 0.1 m blocks placed at random, announces each on first encounter and reports a running "
     "summary, finding all three blocks in every test run.", after=8)

doc.add_heading("Contents", level=1)
for line in ["1  Introduction", "2  Literature Review", "3  Methodology", "4  Evaluation of Results",
             "5  Discussion and Conclusions", "6  Bibliography and References",
             "Appendix A  Running the Webots package", "Appendix B  Key controller code"]:
    para(line, after=1)
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

# ======================================================================= 1 introduction
doc.add_heading("1  Introduction", level=1)
doc.add_heading("1.1  Background and motivation", level=2)
para("Sending a robot rather than a person into an unknown or hazardous space, such as a collapsed building or a "
     "contaminated store room, is safer and can be repeated as often as needed. Part 1 of this project showed the "
     "first half of such a mission: an e-puck in Webots that searches an arena for coloured blocks and reports what it "
     "finds. Colour alone, however, cannot tell a dog from a cat or an aircraft from a ship. Recognising objects needs "
     "a learned model, and the state of the art for image classification is the convolutional neural network [2], [3].")
para("CNNs bring their own problem. Training them takes millions of operations per image and is done on GPUs, while a "
     "small robot has a modest processor that must also run its sensors and motors in real time. The usual answer, "
     "and the one this brief asks for, is to develop the model outside the robot, then optimise it so that running it "
     "on the robot does not disturb the robot's operations and movements. This report documents that process on the "
     "CIFAR-10 benchmark [1] and the e-puck robot in Webots [13], [14].")
doc.add_heading("1.2  Aims and objectives", level=2)
para("The aim is to build a CNN for CIFAR-10 object classification off the robot, test it on my own images and on the "
     "e-puck's camera in Webots, and show how it can be optimised and integrated so that the robot keeps moving "
     "smoothly. Table 1 lists the measurable objectives, their milestone dates (tracked on Trello) and the outcome.")
table([["#", "Objective and measurable target", "Milestone", "Outcome"],
       ["O1", "Build a CNN for CIFAR-10 and show every layer", "10 Nov", "Met: 3 CNNs, all layers listed and drawn (Fig. «arch», Table 3)"],
       ["O2", "Beat Session 8's 70.9% test accuracy", "13 Nov", f"Met: {pct(R9['test_accuracy'])} (ResNet-9)"],
       ["O3", "Predict 3 images of my own", "14 Nov", "Met: 4 of 4 correct, 86–92% confidence"],
       ["O4", "Predict an object seen by the e-puck camera in Webots", "15 Nov", "Partly met: 3 of 5 correct (Colab and live)"],
       ["O5", "Search for red, blue and green blocks; announce first encounter and summarise", "20 Nov", "Met: 3 of 3 blocks in every run"],
       ["O6", "CNN inference within one 32 ms control step without blocking motion", "16 Nov", "Met: 1.3 ms (small CNN), 12 ms (ResNet-9), background thread"],
       ["O7", "Manage the project in Trello and the code in GitHub", "Ongoing", "Met: board and repository kept up to date"]],
      "Aims and objectives with milestones and results.", widths=[1.0, 7.0, 1.8, 6.2])
doc.add_heading("1.3  Structure of the report", level=2)
para("Section 2 reviews the literature on CNNs, robot vision in Webots and model optimisation. Section 3 describes the "
     "environment, the three CNN architectures, the training pipeline, the Webots worlds and controller, and the "
     "optimisation proposal. Section 4 evaluates the results against benchmarks, with error analysis and the "
     "optimisation trade-offs. Section 5 relates the findings to the aims, discusses limitations and sets out future "
     "work. The appendices explain how to run the submitted Webots package and list key code.")

# ======================================================================= 2 literature review
doc.add_heading("2  Literature Review", level=1)
doc.add_heading("2.1  Convolutional neural networks for image classification", level=2)
para("LeCun et al. [2] established the CNN pattern still used today: small learned filters slid across the image, "
     "weight sharing to keep the parameter count low, and pooling to build tolerance to small shifts. Their LeNet-5 "
     "read handwritten digits, but the approach only displaced hand-crafted features once data and GPUs caught up: "
     "AlexNet [3] won ImageNet 2012 by a wide margin using ReLU activations, dropout and GPU training. Simonyan and "
     "Zisserman [4] then showed that stacks of plain 3 × 3 convolutions are enough, provided the network is deep; "
     "every convolution in this project follows their 3 × 3 design.")
para("Depth, however, made networks harder to train. He et al. [5] observed that deeper plain networks can reach a "
     "higher training error than shallower ones, which is an optimisation problem rather than overfitting, and solved "
     "it with residual (skip) connections that let each block learn a correction to its input. Their 110-layer ResNet "
     "reached 6.43% error on CIFAR-10, the benchmark used in Section 4. Batch normalisation [6] stabilises training by "
     "normalising each layer's inputs, and dropout [7] reduces co-adaptation of neurons. These are the techniques "
     "added in my improved CNN.")
para("A recurring critique of this literature is that headline results come from very deep, expensive models. Page "
     "[11] showed that the training recipe matters as much as depth: a 9-layer residual network trained with a "
     "one-cycle learning-rate schedule [9], label smoothing [10] and Cutout [8] reaches 94% on CIFAR-10 in minutes "
     "on one GPU. This finding shaped the project: it suggests that a shallow network, which is cheaper to run on a "
     "robot, can match deep ones if it is trained well. The Session 8 notebook [26] supplied the baseline, which uses "
     "the Adam optimiser [12].")
doc.add_heading("2.2  Robot vision in Webots", level=2)
para("Webots [13] is an open-source simulator with physics, sensors and photorealistic rendering, and the e-puck [14] "
     "is a small differential-drive educational robot with a front camera and eight infrared proximity sensors; the "
     "Webots reference manual [15] documents their APIs. Simulation makes vision experiments repeatable and safe, but "
     "it is not free of bias. Bruce et al. [16] showed that colour segmentation in a suitable colour space (they used "
     "YUV; HSV is the common alternative) is fast and reliable when colours are distinct, which is why colour "
     "thresholding remains the standard first step for block search. Its weakness is that it identifies colours, not "
     "objects.")
para("Tobin et al. [24] identified the main obstacle to using learned vision across simulation and reality: a model "
     "trained in one domain sees the other as a different distribution. They addressed it by randomising textures and "
     "lighting during training. The same reality gap appears in reverse in this project, where a CNN trained on "
     "real photographs is tested on rendered Webots models, and Section 4.4 measures it.")
doc.add_heading("2.3  From classification to detection", level=2)
para("A classifier labels a whole image; a robot needs to know where an object is as well. R-CNN [21] first proposes "
     "candidate regions with a cheap method and then classifies each with a CNN. Single-shot detectors such as SSD "
     "[22] and YOLO [23] predict boxes and classes in one pass and are much faster, at the cost of needing labelled "
     "bounding boxes for training. For a robot that already knows roughly where an object is, from colour or range "
     "sensors, the region-proposal idea is a pragmatic bridge, and it is the approach implemented here (Section 3.6).")
doc.add_heading("2.4  Model optimisation for embedded deployment", level=2)
para("Four families of techniques reduce the cost of running a CNN. Quantisation stores weights and activations in 8 "
     "bits instead of 32: Jacob et al. [17] showed integer-only inference with little loss of accuracy, and TensorFlow "
     "Lite implements post-training quantisation [25]. Pruning removes weights that contribute little; Han et al. "
     "[18] combined pruning, quantisation and coding to shrink networks by 35–49× without loss of accuracy. Efficient "
     "architectures such as MobileNets [19] replace standard convolutions with depthwise separable ones. Knowledge "
     "distillation [20] trains a small network to mimic a large one. Critically, most of these results are reported on "
     "ImageNet and on mobile ARM processors; the speed benefit of int8 depends on the hardware's integer kernels, which "
     "this project confirms on an x86 CPU (Section 4.6). Quantisation was chosen here because it needs no retraining "
     "and the brief's goal is to protect the robot's control loop, not only to save memory.")
doc.add_heading("2.5  Summary and gap", level=2)
para("The literature provides well-tested building blocks, residual networks and efficient training recipes for "
     "accuracy, and quantisation for speed, but rarely evaluates them on a mobile robot's own camera in simulation, "
     "or measures how inference fits around a control loop. This project fills that gap at small scale: it trains to "
     "near-benchmark accuracy, then measures what happens to accuracy and timing when the model is moved onto the "
     "e-puck in Webots.")

# ======================================================================= 3 methodology
doc.add_heading("3  Methodology", level=1)
doc.add_heading("3.1  Environment set-up", level=2)
table([["Component", "Version / setting", "Purpose"],
       ["Google Colab", "NVIDIA T4 GPU (L4 for the re-run), Python 3, TensorFlow/Keras", "Training and evaluation of all CNNs"],
       ["Local laptop", "Windows 11, Python 3.12, TensorFlow 2.21, numpy, Pillow", "Webots controller, TFLite inference"],
       ["Webots", "R2023b, e-puck PROTO, camera 320 × 240, basic time step 32 ms", "Simulation of the robot and arena"],
       ["GitHub", "Private repository, branch main, 3 commits for Part 2", "Code management and version history"],
       ["Trello", "Board with Backlog, To Do, In Progress, Review, Done", "Project management and milestones"]],
      "Software and hardware used.", widths=[3.0, 7.0, 6.0])
para("The controller's Python interpreter is set per project with a runtime.ini file, because Webots otherwise uses "
     "the first Python on the PATH, which here lacked numpy. Fig. «trello» and Fig. «github» show how the work was managed.")
figure("docs/trello_board.jpg", "Trello board used to manage Part 2: one card per task, moved from To Do to Done, "
                                "with checklists and result comments.", 15, key="trello")
figure("docs/github_repo.jpg", "GitHub repository: notebook, training script, results, diagrams, Webots world and "
                               "controller, with a README that documents set-up and results.", 15, key="github")

doc.add_heading("3.2  Dataset and preprocessing", level=2)
para("CIFAR-10 [1] contains 60,000 colour images of 32 × 32 pixels in ten classes (airplane, automobile, bird, cat, "
     "deer, dog, frog, horse, ship, truck): 50,000 for training and 10,000 for testing. Pixel values were scaled from "
     "0–255 to 0–1. The baseline used the images as they are; the improved CNN and ResNet-9 used data augmentation "
     "during training only: random crops after 4-pixel reflect padding and random horizontal flips, and for ResNet-9 "
     "also Cutout [8], which blanks a random 8 × 8 square so the network cannot rely on a single region.")

figure("cnn/outputs/cifar10_samples.png", "Sample CIFAR-10 training images, one row per class. Each image is only "
                                          "32 × 32 pixels, which makes look-alike classes such as cat and dog hard.", 8.5,
       key="samples")
doc.add_heading("3.3  CNN architectures", level=2)
para("Three models were built in increasing order of capability, so that the effect of each design choice could be "
     "measured (Fig. «arch»).")
bullets([("Baseline (Session 8 CNN_v3, 122,570 parameters): ", "Conv 32 → MaxPool → Conv 64 → MaxPool → Conv 64 → "
          "Flatten → Dense 64 → Dense 10. Trained for 10 epochs with SGD and, separately, with Adam."),
         ("Improved CNN (323,946 parameters): ", "three blocks of two 3 × 3 convolutions with batch normalisation and "
          "ReLU, max pooling and dropout (32 → 64 → 128 filters), then global average pooling, Dense 256 with dropout "
          "and Dense 10. Adam, augmentation, early stopping."),
         ("ResNet-9 (6,577,610 parameters): ", "eight 3 × 3 convolutions, each followed by batch normalisation and "
          "ReLU, three max-pooling stages, two residual blocks and a dense output layer (Table 3).")])
figure("docs/diagrams/baseline_cnn.png", "Baseline CNN (Session 8 CNN_v3): three convolutions and two dense layers, "
                                         "122,570 parameters.", 16, key="base")
figure("docs/diagrams/improved_cnn.png", "Improved CNN: three convolution blocks with batch normalisation and dropout, "
                                         "global average pooling, 323,946 parameters.", 16, key="imp")
figure("docs/diagrams/resnet9_cnn.png", "ResNet-9 architecture. Diagrams for all three models were generated from the "
                                        "Keras models and their parameter totals checked against model.summary().", 16, key="arch")
table([["#", "Layer", "Output shape", "Parameters"],
       ["–", "Input (RGB image)", "32 × 32 × 3", "0"],
       ["1", "Conv 3 × 3, 64 + BN + ReLU (prep)", "32 × 32 × 64", "1,728 + 256"],
       ["2", "Conv 3 × 3, 128 + BN + ReLU, MaxPool 2", "16 × 16 × 128", "73,728 + 512"],
       ["3–4", "Residual block 1: 2 × (Conv 3 × 3, 128 + BN + ReLU), Add", "16 × 16 × 128", "294,912 + 1,024"],
       ["5", "Conv 3 × 3, 256 + BN + ReLU, MaxPool 2", "8 × 8 × 256", "294,912 + 1,024"],
       ["6", "Conv 3 × 3, 512 + BN + ReLU, MaxPool 2", "4 × 4 × 512", "1,179,648 + 2,048"],
       ["7–8", "Residual block 2: 2 × (Conv 3 × 3, 512 + BN + ReLU), Add", "4 × 4 × 512", "4,718,592 + 4,096"],
       ["–", "Global max pooling", "512", "0"],
       ["9", "Dense 10 (class scores)", "10", "5,130"],
       ["", "Total (trainable 6,573,130; BN statistics 4,480)", "", "6,577,610"]],
      "Every layer of the ResNet-9, as listed by model.summary(). The 8 convolutions plus the dense layer give the "
      "name ResNet-9; the 8 convolution layers are the hidden layers.", widths=[1.2, 8.3, 3.2, 3.3], bold_last=True)

doc.add_heading("3.4  Training pipeline", level=2)
table([["Setting", "Baseline", "Improved CNN", "ResNet-9"],
       ["Optimiser", "SGD; Adam", "Adam", "SGD, Nesterov momentum 0.9"],
       ["Learning rate", "default", "1e-3, reduce on plateau", "one-cycle: 0 → 0.1 (30%) → 0"],
       ["Batch size / epochs", "64 / 10", "64 / 37 (early stop)", "512 / 30"],
       ["Loss", "cross-entropy", "cross-entropy", "cross-entropy, label smoothing 0.1"],
       ["Regularisation", "none", "BN, dropout, augmentation", "BN, L2 5e-4, crop/flip/Cutout"],
       ["Precision / time", "fp32 / 1.2 min", "fp32 / 10.1 min", "mixed fp16 / 5.8 min"]],
      "Training settings (Colab T4 GPU).", widths=[3.4, 3.4, 4.3, 4.9])
para("ResNet-9 follows Page's recipe [11]. The one-cycle schedule [9] warms the learning rate up to a high peak, which "
     "acts as a regulariser, then anneals it to zero; label smoothing [10] stops the network from becoming "
     "over-confident; mixed precision halves memory traffic on the GPU. The 10,000 test images were used as the "
     "validation set to plot training curves, and were never used to choose weights (no early stopping on ResNet-9). "
     "A simplified 5-step notebook (CNN_Simple.ipynb) reproduces the result in one run: it re-trained the model on a "
     "Colab L4 GPU to 92.8%, within 0.3 points of the original run, which shows the result is reproducible.")

doc.add_heading("3.5  Predictions on my own images and on the robot camera", level=2)
para("Four photos of my own (a car, a cat, a dog and a ship) were classified, one more than the brief's three. CIFAR "
     "images show one object filling a square frame, so each photo was cropped to a square around the object and "
     "resized to 32 × 32 with bilinear interpolation. For the robot camera, a miniature dog, cat, horse, deer and "
     "airplane were placed in Webots (Fig. «world»), and the e-puck stopped 0.55–0.70 m from each and saved a 320 × 240 "
     "frame. Because the robot parks facing the object, a fixed region of interest (ROI) was used: a square of side "
     "0.62 × image height, horizontally centred and starting 2% from the top, which keeps the object and removes "
     "most of the floor. Both the whole frame and the ROI were classified, to measure the effect of cropping.")
figure("webots/cnn_world_overview.png", "Webots world cnn_object_test.wbt: a walled 3 × 3 m arena with miniature "
                                        "CIFAR-10 objects (dog, cat, horse, deer, airplane) and four 8 cm coloured "
                                        "blocks, shuffled at random at the start of every run.", 12, key="world")

doc.add_heading("3.6  Webots integration and the e-puck controller", level=2)
para("A single Python controller, epuck_controller.py, serves both worlds as a Webots Supervisor and selects its "
     "mode from controllerArgs. Its functions are grouped into vision (bgra_to_hsv, colour_mask, "
     "detect_colour_blob, detect_largest_blob, identify_colour), layout (randomise_objects), the CNN (CnnWorker) and "
     "one class per mode (ColourSearch and ObjectTour).")
doc.add_heading("Search mode: colour-block search (Part 1 brief)", level=3)
para("World colour_block_search.wbt contains a red, a blue and a green block, each 0.3 m × 0.3 m × 0.1 m with a "
     "physics bounding box. At start-up randomise_objects moves the blocks to random, non-overlapping positions at "
     "least 0.45 m apart and at least 0.4 m from the robot, and restores any colours left from an earlier run. The "
     "robot then runs a finite-state machine (Fig. «fsm»):")
bullets([("SEARCH: ", "drive forward for about 5 s and then spin a full turn to scan, checking the camera for any "
          "target colour; turn direction alternates so the robot does not circle in one place."),
         ("APPROACH: ", "lock on to the colour of the largest blob and steer proportionally to keep it centred; a "
          "hysteresis threshold (1% of the frame to start, 0.2% to keep tracking) avoids flicker."),
         ("RECORD: ", "when the blob fills 35% of the view or the front sensors fire, confirm the nearest unfound "
          "block of that colour, save a camera frame, display \"I see red\", \"I see blue\" or \"I see green\", then "
          "display a summary of every block encountered so far and those still to find. The found block is repainted "
          "a pale colour so it is never reported twice, which makes the response happen on the first encounter only. "
          "Once every block of a colour is found, that colour is ignored altogether, and colours with no block in "
          "the world (yellow) are ignored from the start."),
         ("AVOID: ", "reverse and turn by a random angle when the infrared sensors detect an obstacle or after a "
          "record."),
         ("DONE: ", "when all blocks are found, or after a 10-minute time limit (error handling for an unreachable "
          "block), print the final summary, including any block not found and the mean vision cost.")])
para("Messages are printed to the console, written to captures/run_log.txt and drawn as on-screen labels in the 3D "
     "view with Supervisor.setLabel.")
figure("docs/diagrams/search_fsm.png", "State machine of the colour-block search (search mode of epuck_controller.py).",
       15.5, key="fsm")
doc.add_heading("Colour detection", level=3)
para("Each camera frame (BGRA) is converted to HSV with a vectorised numpy routine using the OpenCV scale (H 0–179, "
     "S and V 0–255). Table 5 lists the thresholds. Red wraps around hue 0, so it has two ranges. The saturation "
     "floor of 150 for red was set after measuring the wooden floor, which is a faint red with saturation up to "
     "about 122; the value floor of 40 for yellow accepts shaded faces, which look olive.")
table([["Colour", "Hue (H)", "Saturation (S)", "Value (V)"],
       ["Red", "0–8 and 170–180", "150–255", "60–255"],
       ["Green", "40–85", "80–255", "40–255"],
       ["Blue", "100–130", "120–255", "40–255"],
       ["Yellow (tour mode only)", "20–35", "120–255", "40–255"]],
      "HSV thresholds used by the controller (OpenCV scale).", widths=[4.5, 4.0, 3.75, 3.75])
doc.add_heading("Tour mode: CNN on the robot camera (Proposal A, implemented)", level=3)
para("In cnn_object_test.wbt the robot visits every object in nearest-first order. It drives with a go-to-point "
     "controller (position from the supervisor, heading from an InertialUnit, proportional heading correction) and "
     "sidesteps when the front infrared sensors fire. At each CIFAR object it stops, faces the object, crops the ROI "
     "described in Section 3.5, resizes it to 32 × 32 and classifies it with the int8 ResNet-9; at each block it "
     "names the colour with the HSV ranges. Verdicts and a running score are drawn on screen (Fig. «live»). This is the "
     "region-proposal approach of Section 2.3: a cheap step locates the object and only that region reaches the CNN.")
doc.add_heading("Proposal B: single-shot detection (next step)", level=3)
para("The longer-term route is a lightweight detector, a MobileNet-SSD [19], [22] or a small YOLO [23], fine-tuned on "
     "e-puck frames captured in Webots and labelled with bounding boxes for blocks and objects. It would locate and "
     "classify several objects in one pass without relying on colour, at the cost of a labelled Webots dataset and "
     "more computation, which the optimisation below must absorb.")

doc.add_heading("3.7  Optimisation proposal and justification", level=2)
para("The robot's control step is 32 ms: wheel speeds must be updated every step or motion becomes jerky. The proposal "
     "combines four measures, three of which are implemented in the controller:")
bullets([("1. Quantise the model (implemented). ", "Both CNNs were converted to TensorFlow Lite in fp32, fp16 and int8 "
          "(post-training, calibrated on 200 training images) [17], [25]. int8 stores weights in a quarter of the "
          "space and suits integer-only robot processors."),
         ("2. Pre-filter the camera (implemented). ", "Colour vision runs on 1 control step in 4 and on a frame "
          "down-sampled 2×; the CNN runs only when the robot has found something and stopped facing it."),
         ("3. Take inference off the control loop (implemented). ", "CnnWorker runs the interpreter in a background "
          "thread, so the motor commands never wait for a classification."),
         ("4. Prune and distil (proposed). ", "Distil ResNet-9 into the small CNN [20], then prune it [18], to aim for "
          "ResNet-9 accuracy at the small CNN's 1 ms speed; or offload inference to an edge server over Wi-Fi.")])
para("Quantisation was preferred over pruning as the first step because it needs no retraining and its effect on "
     "accuracy is easy to measure; threading and pre-filtering were chosen because they protect the control loop "
     "regardless of which model is used.")

doc.add_heading("3.8  Project and code management", level=2)
para("The project was run as a small agile project. Every task from the brief became a Trello card with a label "
     "(CNN, Webots, report, presentation), a due date and a checklist. Cards moved from Backlog to In Progress to "
     "Testing / Review to Done, and a card was only moved to Done when its checklist was complete and its result had "
     "been added as a comment, for example the accuracy of each model. At most three cards were in progress at once, "
     "which kept the work focused when the three assignments overlapped. Table «plan» shows the milestones.")
table([["Week", "Dates", "Milestone", "Evidence"],
       ["1", "27 Oct – 2 Nov", "Topic and requirements; Part 1 block-search design", "Trello backlog, interim plan"],
       ["2", "3 – 10 Nov", "Part 1 controller (search, random blocks); interim report; Session 8 baseline CNN", "Interim report (10 Nov)"],
       ["3", "11 – 14 Nov", "Improved CNN; own-image predictions; Webots CNN world and camera frames", "Notebook, results.json"],
       ["4", "15 – 18 Nov", "TFLite optimisation; ResNet-9; live CNN on the robot; GitHub; presentation", "Commits d2f4e4b, 7b3b470; slides"],
       ["5", "19 – 21 Nov", "Block-search world to the Part 1 specification; tests; final report; package", "Run logs, this report"]],
      "Project timeline and milestones.", widths=[1.2, 2.8, 7.6, 4.4])
TAB_KEYS_PLAN = TAB[0]
para("Code was managed with Git and a private GitHub repository. Work was committed in small steps with descriptive "
     "messages, so each result in this report can be traced to the commit that produced it (Table «commits»). "
     "Machine-specific files such as runtime.ini and large generated outputs are excluded with .gitignore, and the "
     "README documents set-up, the controller's modes, the HSV ranges and the hardware used.")
table([["Commit", "Date", "Change"],
       ["7d08e60", "17 Nov", "Initial commit"],
       ["9362ede", "17 Nov", "Random block placement in the Webots arena"],
       ["47f0ae3", "17 Nov", "Colour block search world"],
       ["9f8c8fc", "18 Nov", "Assignment 3 submission: notebook, CNN world, single controller"],
       ["d2f4e4b", "18 Nov", "ResNet-9 stage, CNN results and verified architecture diagrams"],
       ["7b3b470", "18 Nov", "ResNet-9 running live on the e-puck camera in Webots"]],
      "Commit history of the GitHub repository (branch main).", widths=[2.2, 2.0, 11.8])
TAB_KEYS_COMMITS = TAB[0]
para("Risks were identified early and most of them did occur, so the contingencies were used (Table «risks»).")
table([["Risk", "What happened", "Mitigation"],
       ["GPU not available in Colab", "No T4 free during the final re-run", "Switched the runtime to an L4 GPU (Colab Pro)"],
       ["Slow dataset download", "CIFAR-10 took about 25 minutes from the Toronto server", "Started training runs early; kept results.json from every run"],
       ["Webots Python environment", "Controller crashed: the default Python had no numpy", "runtime.ini points Webots at Python 3.12; documented in the README"],
       ["CNN slows the robot", "Keras inference took 47.5 ms, more than a control step", "TFLite int8, background thread, classify only while stopped"],
       ["Simulation domain gap", "Rendered animals misclassified", "Region-of-interest crop; fine-tuning on Webots frames proposed"],
       ["Overlapping deadlines", "Assignments 3, 4 and 5 within four days", "Trello WIP limit of three; report generated from the results"]],
      "Risks, what happened and the contingency used.", widths=[3.6, 5.8, 6.6])
TAB_KEYS_RISKS = TAB[0]

# ======================================================================= 4 evaluation
doc.add_heading("4  Evaluation of Results", level=1)
doc.add_heading("4.1  Accuracy against benchmarks", level=2)
table([["Model", "Parameters", "Epochs", "Test accuracy", "Gain"],
       ["Session 8 reference (CNN_v3, Adam)", "122,570", "10", "70.9%", "–"],
       ["Baseline, SGD", "122,570", "10", pct(R["baseline_sgd"]["test_accuracy"]), "–"],
       ["Baseline, Adam", "122,570", "10", pct(R["baseline_adam"]["test_accuracy"]), "+8.1 pts vs SGD"],
       ["Improved CNN (BN, dropout, augmentation)", "323,946", "37", pct(R["improved_adam"]["test_accuracy"]), "+14.8 pts"],
       ["ResNet-9 (one-cycle, Cutout)", "6,577,610", "30", pct(R9["test_accuracy"], 2), "+8.6 pts"],
       ["ResNet-9 + flip test-time augmentation", "6,577,610", "30", pct(R9["test_accuracy_tta"], 2), "+0.4 pts"],
       ["Published: ResNet-110 [5]", "1.7 M", "164", "93.57%", "benchmark"],
       ["Published: ResNet-9 recipe [11]", "6.5 M", "24", "94%", "benchmark"]],
      "Test accuracy on the 10,000 CIFAR-10 test images, compared with published benchmarks.",
      widths=[6.2, 2.4, 1.6, 2.6, 3.2])
para("Every change paid off. Adam beat plain SGD by 8.1 points in the same 10 epochs, regularisation and augmentation "
     "added 14.8 points, and residual connections with the one-cycle recipe added a further 8.6, reaching 93.1%. "
     "This exceeds the Session 8 reference by 22 points and is within half a point of the 110-layer ResNet, with "
     "eight convolution layers instead of 109, and close to the 94% that Page reports for the same recipe [11]. "
     "Fig. «curves» shows the training curves: training and validation accuracy rise together and the two loss curves "
     "converge from about epoch 10, so the model generalises rather than memorising; the remaining 5-point gap "
     "between training and test accuracy is small for a model of this size.")
figure("cnn/outputs/resnet9_curves.png", "ResNet-9 training curves: accuracy (left) and loss (right) for the training "
                                         "images and the 10,000 validation images.", 16, key="curves")
figure("cnn/outputs/accuracy_comparison.png", "Test accuracy of the four models.", 11.5)

para("The intermediate models explain where the gains came from. With plain SGD the baseline was still improving "
     "slowly after 10 epochs, whereas Adam converged faster but began to overfit: its validation accuracy levelled off "
     "near 70% while training accuracy kept rising (Fig. «sgdadam»). The improved CNN closed that gap with batch "
     "normalisation, dropout and augmentation; its training and validation curves stay close for 37 epochs, until early "
     "stopping ended training once validation loss no longer improved (Fig. «impcurves»).")
figure("cnn/outputs/baseline_sgd_vs_adam.png", "Baseline CNN trained with SGD and with Adam: accuracy and loss on the "
                                               "training and validation sets.", 16, key="sgdadam")
figure("cnn/outputs/improved_curves.png", "Improved CNN training curves (accuracy and loss).", 16, key="impcurves")
doc.add_heading("4.2  Error analysis", level=2)
pc = R["per_class_accuracy_resnet9"]
best_c = max(pc, key=pc.get)
worst_c = min(pc, key=pc.get)
para(f"Per-class accuracy ranges from {pct(pc[worst_c])} for {worst_c} to {pct(pc[best_c])} for {best_c}. The "
     f"confusion matrix (Fig. «conf») shows that the most common error is cat predicted as dog ({R['most_confused_resnet9'][2]} "
     "of 1,000 cat images). The errors are between classes that look alike at 32 × 32 pixels: cat and dog, bird and "
     "airplane, automobile and truck. The improved CNN made the same kind of error more often (cat 65.2%, dog 72.8%), "
     "so the residual network's gain is largest on the hardest classes.")
figure("cnn/outputs/confusion_matrix.png", "Improved CNN confusion matrix, for comparison with the ResNet-9: the same "
                                           "cat/dog confusion is its largest error (107 dog images called cat, against "
                                           "84 cats called dog for the ResNet-9), and errors are more frequent overall.", 10, key="confimp")
table([["Class"] + list(pc)[:5], ["Accuracy"] + [pct(pc[c]) for c in list(pc)[:5]],
       ["Class"] + list(pc)[5:], ["Accuracy"] + [pct(pc[c]) for c in list(pc)[5:]]],
      "ResNet-9 accuracy per class.", widths=[2.2, 2.76, 2.76, 2.76, 2.76, 2.76])
figure("cnn/outputs/confusion_matrix_resnet9.png", "ResNet-9 confusion matrix on the test set. Most errors are "
                                                   "between cat and dog.", 10, key="conf")

doc.add_heading("4.3  Predictions on my own images", level=2)
raw = {e["image"]: top(e) for e in P9["own_raw"]}
crp = {e["image"]: top(e) for e in P9["own_crop"]}
truth = {"my_car.jpg": "automobile", "my_cat.jpg": "cat", "my_dog.jpg": "dog", "my_ship.jpg": "ship"}
rows = [["Image (truth)", "Whole photo", "Cropped to object", "Correct?"]]
for img, tr in truth.items():
    rows.append([f"{img} ({tr})", f"{raw[img][0]} {pct(raw[img][1], 0)}", f"{crp[img][0]} {pct(crp[img][1], 0)}",
                 "yes" if crp[img][0] == tr else "no"])
table(rows, "ResNet-9 predictions on my own photos, before and after cropping.", widths=[4.6, 3.8, 4.2, 3.4])
para("All four photos were classified correctly after cropping, at 86–92% confidence (Fig. «own»). Without cropping the "
     "car was called a ship (37.8%): the car sat low in a tall photo, and squashing the whole photo into 32 × 32 "
     "distorted it. Matching the framing of the training data is therefore part of the model, not an optional extra.")
figure("cnn/outputs/r9_own_raw.png", "ResNet-9 on the whole photos, before cropping: the car is called a ship.",
       15.5, key="ownraw")
figure("cnn/outputs/r9_own_crop.png", "ResNet-9 predictions on my own photos after cropping to the object.", 15.5,
       key="own")

doc.add_heading("4.4  Predictions on the e-puck camera in Webots", level=2)
wr = {e["image"]: top(e) for e in P9["webots_raw"]}
wc = {e["image"]: top(e) for e in P9["webots_crop"]}
live = {"airplane": "airplane 92%", "cat": "horse 52%", "deer": "deer 66%", "dog": "bird 69%", "horse": "horse 83%"}
rows = [["Object", "Whole frame (Colab)", "ROI (Colab)", "Live on the robot (ROI)"]]
for obj in ["airplane", "cat", "deer", "dog", "horse"]:
    img = f"epuck_{obj}.png"
    rows.append([obj, f"{wr[img][0]} {pct(wr[img][1], 0)}", f"{wc[img][0]} {pct(wc[img][1], 0)}", live[obj]])
rows.append(["Correct", "2 / 5", "3 / 5", "3 / 5"])
table(rows, "ResNet-9 on e-puck camera frames. Colab used saved frames; the live column is the controller's own "
            "verdict during a run (33–41 ms each).", widths=[2.6, 4.4, 4.0, 5.0], bold_last=True)
para("The robot camera is much harder than my photos: 2 of 5 correct on the whole frame, 3 of 5 with the ROI and 3 of "
     "5 live on the robot. The airplane, deer and horse are recognised; the cat and dog are not. Two causes stand "
     "out. First, the domain gap [24]: the Webots animals are low-polygon rendered models with flat textures, seen "
     "from 3 cm above the floor, whereas CIFAR-10 contains photographs. Second, the small animals fill little of the "
     "frame, so even the ROI includes floor and wall. Confidence on the correct answers (44–94%) is also lower than "
     "on real photos. The live results differ slightly from Colab because each run places the objects at new random "
     "positions, so the camera sees them from a different angle and distance. Objective O4 is therefore only partly "
     "met, and closing this gap is the first item of future work.")
figure("cnn/outputs/r9_webots_raw.png", "ResNet-9 on the whole e-puck frames: only 2 of 5 correct, with low "
                                        "confidence; floor and wall dominate the image.", 16, key="wraw")
figure("cnn/outputs/r9_webots_crop.png", "e-puck camera frames from Webots and the ResNet-9 prediction on the "
                                         "region of interest.", 16)
figure("docs/webots_live_cnn_final.jpg", "The CNN running live in Webots: on-screen labels show each CNN verdict, "
                                         "the colour of each block and the running score; the camera overlay shows "
                                         "what the e-puck sees.", 10, key="live")

doc.add_heading("4.5  Colour detection and block search in Webots", level=2)
import re
runs = []
for log_path in sorted((ROOT / "docs/block_search_runs").glob("run_*.txt")):
    text = log_path.read_text(encoding="utf-8")
    finds = re.findall(r"\[FOUND\] (\w+) block \((\w+)\) at t=([\d.]+)s", text)
    vision = re.search(r"Vision cost: mean ([\d.]+) ms/frame, max ([\d.]+) ms", text)
    order = ", ".join(f"{c} {float(t):.0f} s" for _, c, t in finds)
    runs.append([log_path.stem.replace("_", " ").title(), order, f"{len(finds)} / 3",
                 f"{float(finds[-1][2]):.1f} s" if finds else "–",
                 f"{vision.group(1)} / {vision.group(2)} ms" if vision else "–"])
para("The block search was tested in the colour_block_search.wbt world, with new random block positions every run "
     "(Table «runs»). In every run the robot found all three blocks, displayed \"I see <colour>\" on the first "
     "encounter only and followed it with a summary of the blocks encountered so far (Fig. «search»). Each block "
     "was reported exactly once, because a found block is repainted outside every HSV range. The time to find all "
     "blocks varied with the layout, because the search is a random walk; a blocked approach was correctly "
     "skipped rather than reported. The mean colour-vision cost stayed far below the 32 ms control step. Testing "
     "also exposed one fault worth recording: in an early run, after the blue block had been found, the robot "
     "kept approaching the blue sky near the horizon and lost time. The fix, ignoring a colour once all its blocks "
     "are found, was applied before the runs in Table «runs».")
table([["Run", "First encounters (colour, simulated time)", "Found", "All found at", "Vision mean / max"]] + runs,
      "Block-search test runs (logs in docs/block_search_runs).", widths=[1.5, 6.6, 1.6, 2.4, 3.9])
TAB_KEYS_RUNS = TAB[0]
figure("docs/block_search_final.jpg", "End of a block-search run: on-screen response \"I see red\" and the summary "
                                      "\"Encountered so far (3/3): blue, green, red | all blocks found\". Found "
                                      "blocks are repainted pale so they are reported only once.", 12, key="search")
figure("docs/block_search_frames.png", "e-puck camera frames saved at the first encounter of each block.", 15)
para("Tour mode named the colour of all four blocks correctly in every run (4 of 4), including the yellow block "
     "after its brightness floor was lowered. In the logged run with the live CNN the robot visited all nine objects "
     "in 101.7 s of simulated time (79.5 s without the CNN).")

doc.add_heading("4.6  Impact of the optimisation", level=2)
rows = [["Model and format", "Size", "Time per image", "Images/s", "Accuracy*"],
        ["Improved CNN, Keras", f"{OI['keras']['size_kb'] / 1024:.1f} MB", f"{OI['keras']['latency_ms']:.1f} ms",
         f"{1000 / OI['keras']['latency_ms']:.0f}", pct(OI['keras']['accuracy'])]]
for fmt in ("tflite_fp32", "tflite_fp16", "tflite_int8"):
    d = OI[fmt]
    rows.append([f"Improved CNN, TFLite {fmt[7:]}", f"{d['size_kb']:.0f} KB", f"{d['latency_ms']:.2f} ms",
                 f"{1000 / d['latency_ms']:.0f}", pct(d['accuracy'])])
for fmt in ("tflite_fp32", "tflite_fp16", "tflite_int8"):
    d = ORS[fmt]
    rows.append([f"ResNet-9, TFLite {fmt[7:]}", f"{d['size_kb'] / 1024:.1f} MB", f"{d['latency_ms']:.1f} ms",
                 f"{1000 / d['latency_ms']:.0f}", pct(d['accuracy'])])
rows.append(["ResNet-9 int8, live in Webots", "6.3 MB", "33–41 ms", "24–30", "3 / 5 objects"])
table(rows, "Size, speed and accuracy of each format. *Accuracy on the first 1,000 test images, one image at a time "
            "on Colab's CPU with one thread.", widths=[5.2, 2.4, 2.8, 2.2, 3.4])
para("Converting to TensorFlow Lite removed most of the cost: the same small CNN takes 47.5 ms per call in Keras, "
     "mostly framework overhead, and 1.0–1.3 ms in TFLite, a 36–47× speed-up. int8 quantisation cut the small CNN "
     "from 1.24 MB to 342 KB and the ResNet-9 from 25.1 MB to 6.3 MB, four times smaller, with accuracy unchanged to "
     "within 0.1 points. On this x86 CPU int8 was not faster than fp32, because the kernels are tuned for floating "
     "point; on the ARM processors used in robots, integer kernels usually give the speed-up as well [17].")
para("The trade-off between the two models is clear: the small CNN is about 9× faster and 19× smaller, the ResNet-9 is "
     "about 9 points more accurate. Both fit inside one 32 ms control step on Colab's CPU (1.3 ms and 12 ms). Live in "
     "Webots, a verdict took 33–41 ms because the same laptop CPU also renders and simulates the world, which is "
     "slightly more than one step; this does not affect motion because inference runs in a background thread and "
     "only while the robot is stopped. Colour vision with the pre-filter cost a mean of 5–7 ms per processed frame, "
     "well within the step. The optimisation therefore met its goal: the robot's operations and movements were not "
     "affected by the deep-learning model.")
figure("cnn/outputs/optimisation.png", "Model size, time per image and accuracy for each TFLite format.", 16)

# ======================================================================= 5 discussion
doc.add_heading("5  Discussion and Conclusions", level=1)
doc.add_heading("5.1  Findings against the aims", level=2)
para("Six of the seven objectives were fully met and one partly (Table 1). The project shows that a carefully "
     "trained shallow residual network reaches benchmark-level accuracy on CIFAR-10 (93.1%) in under six minutes of "
     "GPU time, that it transfers to new real photographs when they are framed like the training data (4 of 4), and "
     "that it can run on a simulated robot without disturbing its control loop after quantisation, pre-filtering "
     "and threading. The colour-block search satisfies the Part 1 brief in full, finding every block at random "
     "positions and reporting each on its first encounter with a running summary.")
doc.add_heading("5.2  Limitations", level=2)
bullets([("Domain gap. ", "Accuracy on rendered Webots objects (3 of 5) is far below accuracy on photos. The CNN has "
          "never seen a rendered model, and the animals are small in the frame."),
         ("Real-time constraints. ", "Live inference took 33–41 ms, just over one 32 ms step, on a CPU shared with the "
          "simulator. It is acceptable only because the robot classifies while stopped; classifying while driving "
          "would need a faster model or hardware."),
         ("Supervisor knowledge. ", "Tour mode uses the supervisor's ground-truth positions to drive to objects and to "
          "frame the ROI. A real robot would need the colour blob, range sensors or a detector instead; search mode "
          "already works from the camera alone."),
         ("Small test sets. ", "Four own photos and five robot frames are enough to show a trend but not to estimate "
          "accuracy precisely; more images would narrow the uncertainty."),
         ("Random search. ", "Search mode wanders and spins, so the time to find all blocks varies from run to run "
          "and an unlucky layout takes longer.")])
doc.add_heading("5.3  Effectiveness of the optimisation", level=2)
para("Quantisation and TFLite conversion gave the largest gain for the least effort: a 4× smaller model and a 36× "
     "faster call with no loss of accuracy. Threading and pre-filtering were what actually protected the control "
     "loop, and they would help with any model. The measured trade-off gives a clear deployment rule: use the 342 KB "
     "small CNN on a weak processor where speed matters, and the int8 ResNet-9 where accuracy matters and a few "
     "milliseconds are available.")
doc.add_heading("5.4  Future work and deployment", level=2)
bullets([("Webots training data. ", "Capture and label e-puck frames of the objects and blocks under randomised "
          "lighting and textures [24], then fine-tune the CNN on them to close the domain gap."),
         ("A detector for block detection and classification. ", "Fine-tune a quantised MobileNet-SSD on the same "
          "frames to find and label blocks and objects in one pass (Proposal B)."),
         ("Distil and prune. ", "Distil the ResNet-9 into the small CNN and prune it, targeting more than 90% accuracy "
          "at about 1 ms per image."),
         ("Smarter search. ", "Replace random wandering with coverage path planning so every block is found in a "
          "predictable time."),
         ("Real hardware. ", "Run the int8 model on an e-puck extension board or a Raspberry Pi and measure frames per "
          "second on the robot itself.")])
doc.add_heading("5.5  Contributions and conclusion", level=2)
para("The project's contributions are a reproducible CNN pipeline that improves step by step from 61.6% to 93.1% "
     "with each gain measured; a quantitative study of the cost of running those models in different formats; an "
     "e-puck controller that runs a 6.6-million-parameter CNN live on the robot camera without disturbing motion; "
     "and a modular colour-block search that meets the Part 1 brief. The main lesson is that accuracy on a benchmark "
     "is only half of the problem for a robot: framing the input like the training data, keeping inference out of "
     "the control loop, and training on data that looks like what the robot will see matter just as much.")

# ======================================================================= 6 references
doc.add_heading("6  Bibliography and References", level=1)
refs = [
    'A. Krizhevsky, "Learning multiple layers of features from tiny images," Univ. of Toronto, Toronto, ON, Canada, Tech. Rep., 2009.',
    'Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," Proc. IEEE, vol. 86, no. 11, pp. 2278–2324, Nov. 1998.',
    'A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet classification with deep convolutional neural networks," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2012, pp. 1097–1105.',
    'K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in Proc. Int. Conf. Learn. Represent. (ICLR), 2015.',
    'K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778.',
    'S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in Proc. Int. Conf. Mach. Learn. (ICML), 2015, pp. 448–456.',
    'N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," J. Mach. Learn. Res., vol. 15, pp. 1929–1958, 2014.',
    'T. DeVries and G. W. Taylor, "Improved regularization of convolutional neural networks with Cutout," arXiv:1708.04552, 2017.',
    'L. N. Smith and N. Topin, "Super-convergence: Very fast training of neural networks using large learning rates," in Proc. SPIE, vol. 11006, 2019.',
    'C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, "Rethinking the Inception architecture for computer vision," in Proc. IEEE CVPR, 2016, pp. 2818–2826.',
    'D. Page, "How to train your ResNet," Myrtle.ai, 2018. [Online]. Available: https://myrtle.ai/learn/how-to-train-your-resnet/',
    'D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in Proc. ICLR, 2015.',
    'O. Michel, "Webots: Professional mobile robot simulation," Int. J. Adv. Robot. Syst., vol. 1, no. 1, pp. 39–42, 2004.',
    'F. Mondada et al., "The e-puck, a robot designed for education in engineering," in Proc. 9th Conf. Auton. Robot Syst. Competitions, 2009, pp. 59–65.',
    'Cyberbotics Ltd., "Webots reference manual (R2023b): Camera, Motor, DistanceSensor and Supervisor," 2023. [Online]. Available: https://cyberbotics.com/doc/reference',
    'J. Bruce, T. Balch, and M. Veloso, "Fast and inexpensive color image segmentation for interactive robots," in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2000, pp. 2061–2066.',
    'B. Jacob et al., "Quantization and training of neural networks for efficient integer-arithmetic-only inference," in Proc. IEEE CVPR, 2018, pp. 2704–2713.',
    'S. Han, H. Mao, and W. J. Dally, "Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding," in Proc. ICLR, 2016.',
    'A. G. Howard et al., "MobileNets: Efficient convolutional neural networks for mobile vision applications," arXiv:1704.04861, 2017.',
    'G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," arXiv:1503.02531, 2015.',
    'R. Girshick, J. Donahue, T. Darrell, and J. Malik, "Rich feature hierarchies for accurate object detection and semantic segmentation," in Proc. IEEE CVPR, 2014, pp. 580–587.',
    'W. Liu et al., "SSD: Single shot multibox detector," in Proc. Eur. Conf. Comput. Vis. (ECCV), 2016, pp. 21–37.',
    'J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in Proc. IEEE CVPR, 2016, pp. 779–788.',
    'J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, "Domain randomization for transferring deep neural networks from simulation to the real world," in Proc. IEEE/RSJ IROS, 2017, pp. 23–30.',
    'TensorFlow, "Post-training quantization," TensorFlow Lite documentation, 2024. [Online]. Available: https://www.tensorflow.org/lite/performance/post_training_quantization',
    'PSB Academy / University of Hertfordshire, "Session 08: Convolutional neural networks, complimentary notes and CNN_v3.ipynb," Applied Robotics and AI Project, 2026.']
for i, ref in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.first_line_indent = Cm(-0.9)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(f"[{i}]\t")
    r.font.size = Pt(9.5)
    r = p.add_run(ref)
    r.font.size = Pt(9.5)

# ======================================================================= appendices
doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
doc.add_heading("Appendix A  Running the Webots package", level=1)
bullets(["Install Webots R2023b and Python 3 with numpy. For the live CNN in tour mode, also install TensorFlow "
         "(pip install numpy pillow tensorflow) in the Python that Webots uses; without it the controller runs "
         "colour-only and prints [CNN] disabled.",
         "If Webots uses a Python without numpy, add webots/controllers/epuck_controller/runtime.ini containing "
         "[python] and COMMAND = <path to python.exe>.",
         "Block search (Part 1 brief): open webots/worlds/colour_block_search.wbt and press Run. Messages appear in "
         "the console, in the 3D view and in captures/run_log.txt.",
         "CNN tour: open webots/worlds/cnn_object_test.wbt and press Run. To see the camera, select the e-puck, then "
         "Overlays → Camera Devices → Show 'camera' overlay. The summary is in captures/tour_log.txt.",
         "Hardware used on the e-puck: front camera (320 × 240), two velocity-controlled wheel motors (maximum "
         "6.28 rad/s), eight infrared proximity sensors ps0–ps7 (ps0, ps1, ps6, ps7 used for obstacles) and an "
         "InertialUnit in the turret slot for heading in tour mode."], style="List Number")
doc.add_heading("Appendix B  Key controller code", level=1)
para("First-encounter response and running summary (ColourSearch.announce):", italic=True)
code('''
def announce(self, colour):
    """Respond to the first encounter of a block, then summarise all encounters so far."""
    message = f"I see {colour}"
    seen = [rec["colour"] for rec in self.found.values()]
    missing = [c for _, c in self.targets.values() if c not in seen]
    summary = (f"Encountered so far ({len(seen)}/{len(self.targets)}): {', '.join(seen)}"
               + (f"  |  still to find: {', '.join(missing)}" if missing else "  |  all blocks found"))
    log(message)
    log(summary)
    self.robot.setLabel(0, message, 0.02, 0.02, 0.14, LABEL_RGB.get(colour, 0xFFFFFF), 0, "Arial")
    self.robot.setLabel(1, summary, 0.02, 0.11, 0.07, 0xFFFFFF, 0, "Arial")
''')
para("Vectorised HSV mask and the CNN worker that runs off the control loop:", italic=True)
code('''
def colour_mask(hsv, ranges):
    """Boolean mask of pixels inside any of the HSV ranges."""
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    for low, high in ranges:
        mask |= np.all((hsv >= low) & (hsv <= high), axis=2)
    return mask

class CnnWorker:
    """Runs the optimised (TFLite int8) CIFAR-10 CNN off the control loop."""
    def submit(self, frame):
        """Classify a frame asynchronously; ignored if the worker is busy."""
        if not self.enabled or self._busy:
            return
        self._busy = True
        threading.Thread(target=self._run, args=(frame.copy(),), daemon=True).start()
''')
para("The full controller (about 900 lines) is in the submitted Webots package and on GitHub.", italic=True)

# ======================================================================= footer page numbers
for section in doc.sections:
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("8757HIQT · Harish Ashokkumar · Applied Robotics and AI Project – Final Report · page ")
    r.font.size = Pt(8.5)
    r.font.color.rgb = MUTED
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        run = p.add_run()
        run.font.size = Pt(8.5)
        if tag:
            fc = OxmlElement("w:fldChar")
            fc.set(qn("w:fldCharType"), tag)
            run._r.append(fc)
        else:
            it_ = OxmlElement("w:instrText")
            it_.set(qn("xml:space"), "preserve")
            it_.text = text
            run._r.append(it_)

# resolve figure cross-references written as «key»
KEYS["runs"] = TAB_KEYS_RUNS
KEYS["plan"], KEYS["commits"], KEYS["risks"] = TAB_KEYS_PLAN, TAB_KEYS_COMMITS, TAB_KEYS_RISKS


def resolve(paragraphs):
    for p in paragraphs:
        for r in p.runs:
            if "«" in r.text:
                for k, n in KEYS.items():
                    r.text = r.text.replace(f"«{k}»", str(n))
                assert "«" not in r.text, r.text


resolve(doc.paragraphs)
for tbl in doc.tables:
    for row in tbl.rows:
        for cell in row.cells:
            resolve(cell.paragraphs)

OUT.parent.mkdir(exist_ok=True)
doc.save(OUT)
print("wrote", OUT, "figures:", FIG[0], "tables:", TAB[0])
