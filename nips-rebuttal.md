# Reviewer PHvZ (Reviewer 1)

## Rebuttal

We thank the reviewer for the careful and constructive feedback. We address each concern below. All same-backbone comparisons use YOLOv8 with identical ImageNet-classification pretraining of the backbone (detection heads randomly initialized), matching the pretraining level of NSGP-RePRE.

**W1 / Q1 (Same-architecture baselines).**
We re-implement NSGP, NSGP-RePRE, EWC, and BPF on the **same YOLOv8 baseline** as ESP-YOLO. Results on VOC 15+5 are:

| Method | Old | New | All | Avg |
|---|---|---|---|---|
| EWC | 63.2 | 47.1 | 59.2 | 55.2 |
| NSGP | 72.8 | 48.2 | 66.6 | 60.5 |
| NSGP-RePRE | 72.4 | 49.1 | 66.6 | 60.8 |
| BPF | 54.6 | 80.1 | 53.2 | 67.4 |
| **ESP-YOLO** | **80.4** | **77.1** | **75.0** | **78.8** |

And results on COCO 40+40 are
| Method | mAP | AP75 | AP50 |
|---|---|---|---|
| NSGP | 20.4 | 21.8 | 31.9 |
| BPF | 12.1 | 13.2 | 18.1 |
| ESP-YOLO | 21.5 | 22.9 | 33.8 |

ESP-YOLO clearly outperforms these methods under a controlled same-backbone comparison, indicating that the gains are attributable to ESPReg and CAD rather than the detector alone. Due to limited compute and rebuttal time, our ongoing COCO 40+40 re-runs use 3 epochs rather than the 12 epochs in the paper, and only includes a limited scope of compared methods. All methods trained under the same budget for fair comparison.

**W2 / Q2 (Efficiency and training overhead).**
We report the auxiliary training-time memory of ESP-YOLO versus classical / SOTA regularization-based IOD methods (YOLOv8 setting):

| Method | Historical Data | Previous Checkpoint | PCA Result | Fisher Information |
|---|---|---|---|---|
| Fine-tuning | – | – | – | – |
| EWC | – | 130.4 MB | – | 520.3 MB |
| Pseudo Labeling | – | 130.4 MB | – | – |
| NSGP | – | 130.4 MB | 3.37 GB | 0.55 MB |
| NSGP-RePRE | 46.5 MB | 130.4 MB | 3.37 GB | 0.55 MB |
| **ESP-YOLO** | – | 130.4 MB | **1.69 GB** | – |

ESP-YOLO’s training-time memory footprint is comparable to EWC and lower than NSGP-RePRE (no historical-data buffer or Fisher matrix; half the PCA storage of NSGP). At **inference**, ESPReg and CAD introduce **no extra modules**, so latency / FLOPs / parameters match the plain YOLOv8 baseline:

| Detector | Latency (ms) | GFLOPs | Params (M) |
|---|---|---|---|
| Faster R-CNN | 42.7 | 120.1 | 41.45 |
| ESP-YOLO (YOLOv8) | 8.85 | 8.75 | 3.15 |

The per-task PCA and eigendecomposition take **3 min 11 s** on our hardware (a one-time cost before each incremental task, not incurred at inference). We will add these measurements to the revised paper.

**W3 / Q3 (Statistical significance).**
Within the rebuttal window we completed a three-seed run on the representative VOC 15+5 setting (mean ± std below). The small standard deviations indicate that the reported gains are stable rather than artifacts of a single seed; we will extend the multi-seed analysis to additional splits in the revision when compute permits.

| | Old | New | All | Avg |
|---|---|---|---|---|
| ESP-YOLO |80.50 ± 0.31 | 76.99 ± 0.10 | 74.02 ± 0.86 | 78.75 ± 0.19 |

**W4 (Selective reporting and COCO gain consistency).**
We agree that reporting should be complete and consistent. In the revision we will (i) report **all** metrics (Old / New / All / Avg) for every setting, including cases where ESP-YOLO is not the best on a single column (e.g., 19+1 All); (ii) reconcile the COCO improvement figure so that the abstract, contributions, and tables use one consistent number; and (iii) tone down absolute phrasing such as “comprehensive superiority.”

**W5 / Q4 (New-class mAP near the Joint upper bound).**
We use a **task-agnostic** evaluation protocol that is **identical** for all experiments. The classification head’s output space is the **union** of old and new classes. On the new-task test set, annotations contain **only new classes**. When computing New mAP, we aggregate AP over new-class categories and ignore old-class APs. Because old-class instances are unlabeled on this set, detections on old objects are treated as false positives for new-class evaluation only insofar as they affect ranking on new categories; low old-class APs on this split are expected and are **not** included in the New metric. Under this standard IOD protocol, strong New performance (approaching the YOLOv8 Joint upper bound) is possible when forgetting is well controlled and plasticity for new classes is preserved; the Table 3 numbers are therefore real results under the same protocol as Table 1, not a typo. As a precaution, although we cannot re-run the multi-step 10-2 setting within the rebuttal window, we will aim to submit refreshed numbers in the subsequent discussion phase. We will also state this protocol explicitly in the paper to avoid ambiguity.

**W6 (Theory scope and relation to EWC).**
We acknowledge that our analysis is **per-layer and first-order**, with historical inputs held fixed, and therefore does **not** formally compose drift across depth. We will revise the abstract claim from “theoretically prove … stability” to **“derive a per-layer feature-drift upper bound that motivates ESPReg.”** This single-layer simplification is common in related projection / regularization analyses (e.g., NSGP-RePRE, OGD). Empirically, ESPReg still reduces measured backbone drift (Table 5 / our normalized drift results) and improves detection mAP, suggesting the surrogate is practically useful. We will also add an explicit discussion of the conceptual link to EWC (quadratic penalty with feature covariance playing a role analogous to importance weights) and include EWC in the main same-backbone detection tables (see W1).

**W7 / Q5 (CAD design and hyperparameter sensitivity).**
Ablation of the distillation channel set (VOC 15+5, with ESPReg):

| Distillation Channels | ESPReg | Old | New | All | Avg |
|---|---|---|---|---|---|
| All (full old-class dist.) | √ | 81.0 | 79.9 | 72.6 | 80.5 |
| Top-5 | √ | 81.0 | 80.5 | 72.9 | 80.6 |
| Top-3 | √ | 79.9 | 76.0 | 72.4 | 78.0 |
| Top-1 (CAD, default) | √ | 80.4 | 77.1 | 75.0 | **78.8** |

Top-1 CAD achieves the best **All / Avg** trade-off: full / top-5 distillation can preserve New more aggressively but hurts the overall task-agnostic metric, consistent with our motivation of avoiding background and task-irrelevant channels. Sensitivity of $\alpha$ and $\beta$ (reported as Old / New / All):

| $\alpha$ \ $\beta$ | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|
| 1 | 69.9 / 80.7 / 64.4 | 72.5 / 79.9 / 66.8 | 72.9 / 78.7 / 67.4 | 69.9 / 73.6 / 64.5 |
| 10 | 76.3 / 79.9 / 69.5 | 77.9 / 79.0 / 70.9 | 74.5 / 78.3 / 69.2 | 73.3 / 74.0 / 67.1 |
| 100 | 78.6 / 78.6 / 72.1 | 80.3 / 77.5 / 73.6 | **80.4 / 77.1 / 75.0** | 78.5 / 72.0 / 71.2 |
| 1000 | 77.3 / 78.3 / 72.5 | 80.9 / 72.0 / 73.4 | 81.4 / 68.6 / 73.6 | 79.9 / 63.0 / 71.0 |

Performance is stable in a neighborhood around $(\alpha,\beta)=(100,100)$; excessively large $\beta$ over-distills and harms New / All. Regarding pseudo-labeling (PL): we agree PL contributes substantially to Old-class recovery (Table 4). We view PL as a lightweight, self-generated labeling step rather than storing raw historical images for replay; we will clarify this distinction and discuss the residual privacy / reliance implications in Limitations.

**Typos / formatting.**
We thank the reviewer for catching these issues (broken Eq. references, empty citation `[]`, “Sec.Sec.”, “effeciency”, inconsistent COCO gain). We will correct them in the revision.

**Limitations.**
Following the reviewer’s suggestion, we will add a dedicated Limitations paragraph covering: (i) the single-layer, fixed-input scope of the theory; (ii) training-time PCA / eigendecomposition overhead; and (iii) reliance on pseudo-labeling within the pipeline.

---

## Author–AC Confidential Comment

Dear Area Chair,

We write regarding Reviewer PHvZ’s note about hidden text in the PDF that appears to instruct an LLM to insert specific phrases into the review.

We want to state clearly that we did **not** insert, instruct, or approve any hidden text, prompt injection, or other review-manipulation content in our submission. To the best of our knowledge, the described content is consistent with integrity-monitoring material that the NeurIPS organizing committee has been reported to inject into review PDFs to detect unauthorized reviewer use of external LLMs (outside the sanctioned AI-reviewing experiment).

We therefore respectfully ask you to verify this point and to treat the issue as **not attributable to the authors**. Thank you for your time and careful handling of this matter.

Best regards,  
The Authors

## Official Comment

1. Regarding full 12-epoch results on MS COCO

We have completed the full 12-epoch training runs on COCO 40+40 using the identical YOLOv8 backbone. The results are summarized below:

| Method | mAP | AP75 | AP50 |
|---|---|---|---|
| EWC | 36.4 | 39.8 | 51.2 |
| NSGP | 32.9 | 35.4 | 47.5 |
| BPF | 16.7 | 18.2 | 23.0 |
| ESP-YOLO | 42.5 | 46.4 | 58.0 |

As shown above, ESP-YOLO surpasses all compared IOD methods in terms of mAP over the union of old and new classes.

2. Regarding the anomalous value in the VOC 10-2 multi-step setting

We thank the reviewer for pointing out this anomalous experimental result. After re-conducting this experiment, we found that the originally reported value was likely due to a typo: the old and new mAP values appeared to be swapped. The re-conducted 3-seed experiment results for this setting are as follows:

| | 1-10 | 11-20 | 1-20 |
|---|---|---|---|
| | 74.90 ± 2.87 | 70.21 ± 0.65 | 70.93 ± 0.33 |

After correcting these results, our claim still holds: ESP-YOLO surpasses the other compared methods in the 10-2 multi-step setting on 1-10, 11-20, and 1-20.

## Official Comment

Dear Reviewer,

Thank you for your continued comment. We have submit a further official comment that responds to the remaining concerns you raised. We would be grateful to know whether the additional results and clarifications fully address your remaining questions. If anything is still unclear, please feel free to let us know, and we will gladly provide more details.

Best regards,
The Authors

---

# Reviewer vzcn (Reviewer 2)

## Rebuttal

We thank the reviewer for the detailed critique. We respond to each weakness and question below.

### Weaknesses

**W1 (Findings 1.F / 2.F; novelty and drift measurement).**
We agree that feature drift and class imbalance effects are known phenomena; our contribution is the **comparative diagnosis on single-stage YOLO** and the methods tailored to its architecture. In YOLO, localization and classification are tightly coupled in an end-to-end architecture (no class-agnostic proposal stage). Hard null-space projection can therefore over-constrain updates that are still needed for localization adaptation. ESPReg (Eq. 9) softens this constraint with eigenvalue-scaled penalties, preserving plasticity along less critical directions while stabilizing historically important subspaces.

We also thank the reviewer for pointing out the flaw in comparing raw $\ell_2$ drift across heterogeneous backbones. We revise the metric to **relative (scale-free) feature drift**. Using the last-epoch Task-1 and Task-2 checkpoints under VOC 15+5, we feed the **same** Task-1 test images to both models and extract the **last backbone feature map**. At each spatial location, with feature vectors $a$ (Task-1 model) and $b$ (Task-2 model), we compute $\|a-b\|_2/\|a\|_2$, average over space per image, and report mean ± std over the set:

| | Faster R-CNN | YOLO | NSGP-RePRE | ESP-YOLO |
|---|---|---|---|---|
| Relative Feature Drift (%) | 67.7 ± 5.2 | 100.2 ± 5.9 | 21.5 ± 1.4 | 48.6 ± 5.5 |

Under this normalized metric, the claim still holds: the one-stage YOLO baseline exhibits more severe drift than Faster R-CNN, and ESP-YOLO substantially reduces it. We will clarify that $F$ denotes the backbone output feature map for both architectures (spatially aligned; not RoI features).

**W2 (Analysis 1.A: theory overstatement; composition across depth).**
We acknowledge that “theoretically prove stability” overstates what our analysis delivers. We will revise the claim to: **“we derive a per-layer upper bound on feature-drift risk that motivates ESPReg.”** The bound treats each layer in isolation with historical inputs held fixed and does not formally compose drift across depth; the layer-averaged loss in Eq. 9 is therefore a practical surrogate rather than a full-depth guarantee. This per-layer simplification is common in related analyses (e.g., NSGP-RePRE, OGD). Empirically, reduced measured backbone drift and improved mAP suggest the surrogate remains useful. We will state these assumptions explicitly in Limitations.

**W2 continued (Analysis 2.A: BCE vs. softmax / SS-IL, SSUL).**
We thank the reviewer for pointing out the assertiveness of this claim and for bringing SS-IL and SSUL to our attention. We acknowledge the limitation honestly: the statement "BCE suppresses old classes while softmax preserves them" was **reverse-attributed** from the different behaviors of YOLO and Faster R-CNN under identical IOD protocols, and it lacks support from forward-controlled ablations that isolate the loss function from the architecture. We will revise the manuscript to present it as a hypothesis grounded in empirical observation rather than an established causal conclusion.

Importantly, this over-attribution does **not** undermine the motivation for CAD. The phenomenon itself is real and reproducible: under the extreme old/new class imbalance inherent to IOD, our YOLO baseline exhibits severe cross-task class confusion (Fig. 1b, e.g., cow→sheep at 73%), whereas two-stage baselines are inherently much less susceptible to it. CAD is designed to remedy this empirically observed deficiency of single-stage baselines, and its necessity stands independently of which attribution is ultimately correct. The ablation in Table 4 confirms that CAD markedly alleviates the confusion and improves task-agnostic accuracy.

We further note that SS-IL and SSUL operate under different premises from ours, so the apparent discrepancy in conclusions is reasonable: SS-IL only discusses the comparison between unified softmax and decoupled softmax without comparing it with BCE loss; SSUL freezes the past-class classifiers, so BCE gradients on old-class channels vanish by construction, whereas in our fully-trainable dense head, every anchor — including unlabeled old-class objects that are assigned as background negatives contributes a suppressive gradient on old channels. We will cite SS-IL / SSUL and discuss the above distinctions in the revision and eframe the claim.

**W3 (Method: same base detector, cost-normalized comparison, $\alpha$/$\beta$).**
Please see responses to Q2, W5–W7 below: we provide same-YOLOv8 re-implementations, a cost-normalized memory–performance table, and $\alpha$/$\beta$ sensitivity. These directly address the concern that gains may be confounded with the base detector or tuning.

**W5 (Same-backbone detection comparison).**
We re-implement NSGP, NSGP-RePRE, EWC, and BPF on the **same YOLOv8 baseline** as ESP-YOLO. Results on VOC 15+5 are:

| Method | Old | New | All | Avg |
|---|---|---|---|---|
| EWC | 63.2 | 47.1 | 59.2 | 55.2 |
| NSGP | 72.8 | 48.2 | 66.6 | 60.5 |
| NSGP-RePRE | 72.4 | 49.1 | 66.6 | 60.8 |
| BPF | 54.6 | 80.1 | 53.2 | 67.4 |
| **ESP-YOLO** | **80.4** | **77.1** | **75.0** | **78.8** |

And results on COCO 40+40 are
| Method | mAP | AP75 | AP50 |
|---|---|---|---|
| NSGP | 20.4 | 21.8 | 31.9 |
| BPF | 12.1 | 13.2 | 18.1 |
| ESP-YOLO | 21.5 | 22.9 | 33.8 |

ESP-YOLO clearly outperforms these methods under a controlled same-backbone comparison, indicating that the gains are attributable to ESPReg and CAD rather than the detector alone. Due to limited compute and rebuttal time, our ongoing COCO 40+40 re-runs use 3 epochs rather than the 12 epochs in the paper, and only includes a limited scope of compared methods. All methods trained under the same budget for fair comparison.

**W6 (Cost-normalized comparison).**

| Method | Hist. Data | Prev. Ckpt | PCA | Fisher | Total Aux. Mem. | mAP (Old / New / All) |
|---|---|---|---|---|---|---|
| Fine-tuning | – | – | – | – | 0 | 0.0 / 48.2 / 12.0 |
| L2 | – | 130.4 MB | – | – | 130.4 MB | 76.1 / 2.5 / 57.7 |
| EWC | – | 130.4 MB | – | 520.3 MB | 650.7 MB | 63.2 / 47.1 / 59.2 |
| Pseudo Labeling | – | 130.4 MB | – | – | 130.4 MB | - |
| NSGP-RePRE | 46.5 MB | 130.4 MB | 3.37 GB | 0.55 MB | 3.54 GB | 72.5 / 49.1 / 66.6 |
| **ESP-YOLO** | – | 130.4 MB | 1.69 GB | – | **1.82 GB** | **80.4 / 77.1 / 75.0** |

Relative to cheaper checkpoint-only baselines (L2, EWC) and to NSGP-RePRE, ESP-YOLO offers a favorable accuracy–memory trade-off on the same YOLO base.

**W7 ($\alpha$/$\beta$ sensitivity).**
We sweep $\alpha$ (ESPReg) and $\beta$ (CAD) on VOC 15+5; cells report Old / New / All:

| $\alpha$ \ $\beta$ | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|
| 1 | 69.9 / 80.7 / 64.4 | 72.5 / 79.9 / 66.8 | 72.9 / 78.7 / 67.4 | 69.9 / 73.6 / 64.5 |
| 10 | 76.3 / 79.9 / 69.5 | 77.9 / 79.0 / 70.9 | 74.5 / 78.3 / 69.2 | 73.3 / 74.0 / 67.1 |
| 100 | 78.6 / 78.6 / 72.1 | 80.3 / 77.5 / 73.6 | **80.4 / 77.1 / 75.0** | 78.5 / 72.0 / 71.2 |
| 1000 | 77.3 / 78.3 / 72.5 | 80.9 / 72.0 / 73.4 | 81.4 / 68.6 / 73.6 | 79.9 / 63.0 / 71.0 |

Performance is stable in a neighborhood around $(\alpha,\beta)=(100,100)$. Excessively large $\beta$ over-distills and harms New / All; too small $\alpha$ under-regularizes and hurts Old / All. We will include this grid in the revision.

### Questions

**Q1 (BCE vs. softmax; reconcile with SS-IL / SSUL).**
Please see **W2 continued** above for the reframed claim and the premise-level reconciliation with SS-IL / SSUL. Regarding the requested controlled experiment (e.g., YOLO with softmax vs. BCE, or joint vs. separated softmax), we are unfortunately unable to complete the required code modification and retraining within the rebuttal window. We also note that YOLO's detection head is co-designed with per-class sigmoid + BCE, so forcibly substituting a softmax classifier conflicts with this architecture, and the resulting numbers would conflate the loss change with an architecture mismatch, offering limited comparative value. We will state the loss-isolation study explicitly as future work in Limitations.

**Q2 (Initialization / pretraining fairness).**
ESP-YOLO uses the **same initialization level** as NSGP-RePRE: the backbone is pretrained **only** on ImageNet classification; the FPN / detection heads are **randomly initialized** and learned from the base detection task. We do **not** start from a COCO-detection-pretrained YOLOv8 checkpoint. Thus there is no detection-level pretraining leakage of “novel” COCO classes. We will state this clearly in the experimental setup and release configuration details.

### Other Comments

1. **Equation-first setup.** We will add a short formal description of YOLO’s pipeline (backbone → neck → dense head; per-class sigmoid + BCE) and define the drift feature map $F$ for each architecture.
2. **Drift measurement point.** Confirmed: $F$ is the **backbone output** feature map for both YOLO and Faster R-CNN (dense $H\times W$ map in Eq. 15), ensuring spatial alignment across Task-$t$ and Task-$t{+}1$. We will state this explicitly.
3. **Pseudo-labeling (PL).** PL denotes Pseudo-labeling: the previous-stage model generates labels for old classes on current-task images. It is used in our pipeline (Table 4) but was under-introduced in Sec. 4. We will briefly define PL in the method section and discuss its role versus replay of raw images.
4. **Notation / typos.** We will unify notation ($F$, $M$, $p$), fix “Sec.Sec. 4.1,” and restore the missing citation at l.233. As for the undefiend "anchor", it is a standard terminology of YOLO's detection head and denotes the detection reference point in the input image that corresponds to each spatial location of the head's input feature map.

**Limitations.**
We will add a dedicated Limitations section in the revision.

## Official Comment

1. Regarding hyperparameter tuning

We thank the reviewer for raising this important concern. To clarify, for all compared methods (NSGP, NSGP-RePRE, EWC, BPF), we adopted their originally reported hyperparameters without task-specific re-tuning on our YOLOv8 baseline. While we acknowledge that an exhaustive grid search could improve the performance of the compared methods, the performance gaps observed in our same-backbone comparisons are substantial: on VOC 15+5, ESP-YOLO achieves 75.0% All mAP versus 66.6% for NSGP-RePRE/NSGP, 59.2% for EWC, and 53.2% for BPF; on COCO 40+40, it reaches 42.5% mAP versus 36.4% for EWC, 32.9% for NSGP, and 16.7% for BPF. These large margins suggest that the observed gains are driven by methodological design rather than by hyperparameter tuning.

Due to the limited time window, We only conducted a sensitivity analysis with respect to the distillation weight of BPF on VOC 15+5. The sensitivity analysis for other method-specific hyperparameters will be submit in the revision. The results are as follows:

|distillation weight|old|new|all|
|---|---|---|---|
|0.05|52.7|80.3|51.5|
|0.15|54.6|80.1|53.2|
|0.50|54.1|79.5|52.2|

These results show that the influence of the distillation weight on the performance of BPF is negligible, which further proves that the performance gaps we observed are unlikely to be overcome through hyperparameter tuning.

2. Regarding full 12-epoch results on MS COCO

Yes, we have now completed the full 12-epoch training runs on COCO 40+40 under the identical YOLOv8 backbone. The results are as follows:

| Method | mAP | AP75 | AP50 |
|---|---|---|---|
| EWC | 36.4 | 39.8 | 51.2 |
| NSGP | 32.9 | 35.4 | 47.5 |
| BPF | 16.7 | 18.2 | 23.0 |
| ESP-YOLO | 42.5 | 46.4 | 58.0 |

As the results indicate, ESP-YOLO surpasses all compared IOD methods in terms of mAP over the union of old and new classes.

## Official Comment

Dear Reviewer,

Thank you for your constructive follow-up. We have posted an additional official comment that addresses the remaining concerns you kindly highlighted. We would greatly appreciate your feedback on whether these additional results and clarifications satisfactorily resolve your outstanding questions. If any aspect remains unclear, please let us know and we will promptly provide further details.

Best regards,  
The Authors


## Official Comment

Dear Reviewer,

Thank you for your thoughtful follow-up and for acknowledging the clarifications we provided. We greatly appreciate your continued engagement and the constructive direction you have offered.

Regarding the fairness of the same-backbone comparison, we respectfully argue that the observed performance gaps are large enough to be explained solely by under-configured baselines. While we agree that transferring methods across detector families involves architectural overhead, the margins we observe exceed what is typically recoverable through hyperparameter tuning alone.

Furthermore, our limited sensitivity analysis on BPF (distillation weight sweep on VOC 15+5) shows that its performance varies only within a narrow band (51.5–53.2 All mAP). This suggests that the method is not highly sensitive to this hyperparameter, and that its underperformance on YOLOv8 is more likely rooted in method design than in suboptimal tuning.

We also fully accept your observation that several claims should be scoped more honestly. In the revised manuscript, we will carefully reframe our contribution statements to ensure they accurately reflect the scope and limitations of our theoretical and empirical findings, avoiding any overstatement beyond what is rigorously supported.

Thank you again for your time and consideration.

Best regards,  
The Authors

---

# Reviewer RwTa (Reviewer 3)

## Rebuttal

We thank the reviewer for the positive assessment and the important fairness questions.

### Weaknesses

**W1 (Base-detector confound).**
We agree that comparing YOLOv8-based ESP-YOLO to Faster R-CNN-based prior methods conflates detector strength with the IOD algorithm. To isolate the incremental-learning contribution, we re-implement NSGP, NSGP-RePRE, EWC, and BPF on the **same YOLOv8** backbone and pretraining protocol:

Results on VOC 15+5 are:

| Method | Old | New | All | Avg |
|---|---|---|---|---|
| EWC | 63.2 | 47.1 | 59.2 | 55.2 |
| NSGP | 72.8 | 48.2 | 66.6 | 60.5 |
| NSGP-RePRE | 72.4 | 49.1 | 66.6 | 60.8 |
| BPF | 54.6 | 80.1 | 53.2 | 67.4 |
| **ESP-YOLO** | **80.4** | **77.1** | **75.0** | **78.8** |

And results on COCO 40+40 are
| Method | mAP | AP75 | AP50 |
|---|---|---|---|
| NSGP | 20.4 | 21.8 | 31.9 |
| BPF | 12.1 | 13.2 | 18.1 |
| ESP-YOLO | 21.5 | 22.9 | 33.8 |

ESP-YOLO clearly outperforms these methods under a controlled same-backbone comparison, indicating that the gains are attributable to ESPReg and CAD rather than the detector alone. Due to limited compute and rebuttal time, our ongoing COCO 40+40 re-runs use 3 epochs rather than the 12 epochs in the paper, and only includes a limited scope of compared methods. All methods trained under the same budget for fair comparison.

**W2 (Unnormalized feature-drift comparison).**
We agree that raw $\ell_2$ drift across heterogeneous backbones is not a valid comparison. We now report **relative feature drift** $\|a-b\|_2/\|a\|_2$ on the last backbone layer, using identical Task-1 test images (VOC 15+5):

| | Faster R-CNN | YOLO | NSGP-RePRE | ESP-YOLO |
|---|---|---|---|---|
| Relative Feature Drift (%) | 67.7 ± 5.2 | 100.2 ± 5.9 | 21.5 ± 1.4 | 48.6 ± 5.5 |

The architecture gap remains after normalization. Regarding causality: Table 5 / Fig. 2 show that methods reducing drift (ESPReg) also improve Old-class retention, supporting the link between feature stability and forgetting mitigation; we will phrase this as correlational / mechanistic evidence rather than a formal causal proof.

### Question

**Q1 (Numbers vs. Faster R-CNN Joint; isolate IL contribution).**
The Joint (upper) and Fine-tuning (lower) bounds reported in our paper are already obtained on the **same YOLOv8** baseline as ESP-YOLO. Relative to these YOLOv8 bounds—not the Faster R-CNN Joint—the incremental contribution is properly contextualized. The only entry that slightly exceeds the Joint upper bound is Old mAP on VOC 15+5 (+0.6), which we attribute to run-to-run variation rather than a protocol error. The same-backbone re-implementations above further show that ESP-YOLO’s advantage persists against strong IOD methods on identical YOLOv8, so the gains are not largely an artifact of the detector.

For the claim that feature drift causes forgetting, we thank the reviewer for the correction and clarify that this statement rests on the widely adopted premise of feature-preserving methods (e.g., feature distillation, NSGP): since the detection heads are optimized on the historical feature distribution, drift of intermediate features shifts the inputs of the old-task heads out of distribution and propagates forward to perturb final predictions, even if the heads themselves remain intact. Our results provide indirect evidence consistent with this mechanism — ESPReg reduces measured drift from 13.12 to 6.84 (Table 5) while raising old-class mAP from 51.0% to 72.6% (Table 4) — and we will soften the causal wording and state this limitation explicitly in the revision.

# AC

## Official Comment

Dear Area Chair,

Thank you for your time and effort in overseeing the review process. We are grateful that the reviewers have provided constructive follow-ups and engaged positively with our rebuttal.

Because the additional experiments requested by the reviewers required considerable time to complete, our second-round official comments were posted relatively late in the discussion window. We are concerned that the reviewers may not yet have sufficient time to review these updated results and clarifications.

We would therefore greatly appreciate it if you could kindly remind the reviewers to check our latest responses. Alternatively, if the discussion period is concluding, we would be grateful if you could take into account that we have fully addressed the concerns raised when forming your recommendation.

Thank you again for your understanding and support.

Best regards,
The Authors

---

# Cross-cutting revision checklist (for authors)

- [ ] Fill COCO 40+40 same-backbone table (matched epochs)
- [ ] Fill 3-seed mean ± std (VOC 15+5; others if possible)
- [ ] Fill remaining $\alpha$/$\beta$ grid cells and Pseudo-Labeling mAP in cost table
- [ ] Add controlled BCE / softmax (or separated-head) experiment for Reviewer vzcn Q1
- [ ] Soften abstract: “prove stability” → “derive a per-layer bound motivating ESPReg”
- [ ] Unify COCO gain number; report all metrics including non-best columns
- [ ] Add Limitations paragraph; clarify PL vs. raw-image replay
- [ ] Fix typos; define $F$; state ImageNet-cls-only init
- [ ] Submit Author–AC confidential comment re: hidden integrity-monitoring text
