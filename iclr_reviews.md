# Reviewer 1: u5mT

*Official Review*

**Paper Summary:**

This paper proposes ESP-YOLO to mitigate the problem of catastrophic forgetting in single-stage object detectors during incremental object detection. The authors identify two key challenges that YOLO faces when learning new classes sequentially: intermediate feature drift and cross-task class confusion. To address these, they introduce ESPReg and CAD, respectively. ESPReg applies weighted regularization to parameter updates based on the eigenvalues and eigenvectors of historical feature covariance matrices, while CAD selects class channels according to the prediction confidence of the previous model and performs weighted knowledge distillation. The authors evaluate their method on the PASCAL VOC and MS COCO 2017 datasets under both single-stage and multi-stage incremental learning settings, further validating the approach through ablation studies and an analysis of feature drift. While the paper reports performance improvements over several baselines, the theoretical justification, the fairness of the experimental comparisons, and the consistency of the reported results warrant further scrutiny

**Presentation:** 1: Difficult to follow despite careful reading.

**Informativeness:** 2: I learned something interesting or useful.

**Soundness:** 1: I have significant concerns about the paper’s soundness that do not appear easy to address.

**Critical Strengths:**

This paper addresses the issue of catastrophic forgetting in single-stage incremental object detection, focusing on the stability-plasticity trade-off between preserving features of old classes and learning new ones. Compared to studying continual learning solely within the context of classification tasks, investigating these challenges in the framework of object detectors—such as YOLO—offers clear practical value.

ESPReg specifically addresses the drift in historical features caused by parameter updates, while CAD aims to mitigate prediction bias between old and new classes. The ablation study in Table 4 shows that combining the two yields an Avg mAP of 78.8, which outperforms both ESPReg (74.7) and CAD (71.1) when used individually. While this provides initial evidence for the potential value of the combined mechanism, it does not substitute for theoretical justification or more rigorous comparative experiments.

**Critical Weaknesses:**

The theoretical motivation for ESPReg is derived from an upper bound on feature drift based on the eigenvalues of historical feature covariance matrices. However, the actual regularization objective employs truncated and normalized eigenvalues, and the paper does not rigorously establish how this modified objective relates to the derived bound or guarantees feature stability. Moreover, the theoretical analysis considers individual layer-wise linear mappings, while its extension to the nonlinear convolutional layers and multi-scale feature representations of YOLOv8 remains insufficiently explained. The paper also lacks an overall framework diagram and essential implementation details, including how historical covariance matrices are constructed, stored, and incorporated into the training process. These limitations raise substantial concerns about both the theoretical soundness and reproducibility of the proposed approach.

The experimental evaluation contains notable inconsistencies in the reported metrics. For example, in Table 1, under the VOC 15+5 setting, ESP-YOLO reports 80.4 Old mAP and 77.1 New mAP, but only 75.0 All mAP, whereas their class-count-weighted average would be approximately 79.6 under a unified evaluation protocol. Similar discrepancies appear in Tables 3 and 4. The paper does not clearly explain whether different evaluation protocols are used for these metrics, making the reported improvements difficult to interpret and verify. Furthermore, the comparisons involve different detector architectures without sufficiently controlled experimental settings, and the results lack multiple-run statistics or variance estimates. The effectiveness of CAD is also supported mainly by ablation studies and confusion matrices, without sufficient analysis of its robustness to erroneous high-confidence teacher predictions. Collectively, these issues substantially weaken the credibility of the experimental conclusions and the claimed state-of-the-art performance.

**Questions For Authors:**

The paper lacks an introductory "teaser figure" or a clear, comprehensive flowchart of the ESP-YOLO framework. Although the main text includes some schematic diagrams, readers are still required to mentally synthesize the relationships between ESPReg, CAD, and the YOLOv8 training pipeline. Could the authors provide an overall framework diagram that clearly illustrates the inputs, outputs, and functional placement of each module, as well as their interconnections within the training process, to enhance the clarity and reproducibility of the method?

How does ESPReg compute the historical feature covariance matrix within the convolutional layers and multi-scale feature maps of YOLOv8? Is it necessary to store samples or features from previous tasks? What are the specific time and VRAM overheads associated with PCA, eigen decomposition, and additional regularization? Please provide a comparison of training overheads against baselines such as the original YOLOv8 and NSGP to demonstrate the efficiency advantages of your method in practical deployment scenarios.

Does CAD exacerbate errors when the previous model produces high-confidence predictions for the background or incorrect classes? Have the authors experimented with varying confidence thresholds, full-class distribution distillation, objectness constraints, or other forms of confidence masking? Could you provide class-wise statistics to demonstrate that the gains from CAD are not driven by only a few specific classes?

The magnitude of improvements mentioned in the abstract does not entirely align with the figures presented in the summary of contributions. Please clearly specify the baselines, metric definitions, and calculation methods used for each reported improvement. Furthermore, explain how baselines across different architectures support the claims of achieving state-of-the-art (SOTA) performance. For the primary results, could you report the mean and standard deviation based on multiple independent runs?

Please explicitly detail the class-ordering for each dataset, the data and annotations used at each stage, and the method for generating pseudo-labels for old classes. Additionally, clarify whether the baseline methods utilize the same data, number of training epochs, model initialization, and hyperparameter tuning budgets. These details are crucial for ensuring the comparability of results in incremental object detection.

**Flag For Ethics Review:** No

**Rating:** 1: Clear rejection

**Confidence:** 3: I am familiar with this topic and its relevant literature.

# Reviewer 2: HX4Y

*Official Review*

**Paper Summary:**

This paper proposes ESP-YOLO, a framework for class-incremental object detection based on YOLOv8. The authors identify two major challenges in YOLO-based incremental object detection: intermediate feature drift and cross-task class confusion. To address these challenges, the proposed framework incorporates two components. The first component, Eigenvalue Scaled Projection Regularization (ESPReg), penalizes parameter updates along important directions of historical feature representations, with the regularization strength determined by calibrated eigenvalues of the feature covariance matrix. The second component, Confidence-Aware Distillation (CAD), uses the previous model's classification confidence to selectively distill knowledge from high-confidence predictions, aiming to mitigate cross-task classification confusion. The authors provide a theoretical analysis of feature drift to motivate ESPReg and conduct experiments on PASCAL VOC and MS COCO under multiple single-step and multi-step incremental learning protocols. The reported results demonstrate competitive performance compared with existing incremental object detection methods.

**Presentation:** 2: Generally understandable, but requires substantial effort in places.

**Informativeness:** 2: I learned something interesting or useful.

**Soundness:** 2: I have minor or fixable concerns about the paper’s soundness.

**Critical Strengths:**

The paper investigates incremental object detection using a single-stage detector, which is relevant for practical applications where inference efficiency is important. Compared with the extensive literature on two-stage incremental detectors, exploring effective continual learning strategies for YOLO-based architectures is a worthwhile direction.

The experiments cover multiple class-incremental settings on PASCAL VOC and MS COCO, including multi-step protocols. The reported results are promising, particularly on MS COCO, where ESP-YOLO achieves 42.5% mAP in the 40+40 setting and 44.8% mAP in the 70+10 setting. These results suggest that the proposed framework may be effective for challenging incremental detection scenarios.

**Critical Weaknesses:**

1. The primary concern is that the methodological contributions appear relatively incremental compared with existing continual learning and knowledge distillation approaches. For ESPReg, the underlying principle of protecting important historical feature directions through feature-covariance-based constraints has already been extensively explored in subspace-based continual learning methods. The proposed approach mainly replaces hard gradient projection with an eigenvalue-weighted quadratic regularization term. Similarly, CAD employs teacher confidence to weight the distillation loss and selects the most confident historical class channel for knowledge transfer. Confidence-aware selection and selective knowledge distillation are established ideas in incremental learning and object detection. The ablation studies mainly demonstrate that adding ESPReg and CAD improves performance, rather than establishing the necessity or superiority of their specific formulations. For example, the paper lacks direct comparisons between ESPReg and simpler quadratic subspace regularization schemes, as well as between CAD and confidence-weighted full-channel or foreground-masked distillation.

2. The paper motivates its contributions by arguing that YOLO-based incremental detectors suffer from particularly severe intermediate feature drift and cross-task class confusion. However, both challenges are fundamental problems in incremental object detection and are not unique to single-stage architectures. Feature drift can also arise in two-stage detectors when shared feature representations are updated during incremental training. Similarly, cross-task class confusion can result from class imbalance, incomplete annotations of historical categories, and classifier bias, regardless of the detection architecture. Although Figure 1(a) provides an initial comparison of feature drift between YOLO and Faster R-CNN, this observation alone does not fully establish that the difference is caused by the single-stage detection architecture, as variations in feature representations, optimization, and training configurations may also contribute.

**Questions For Authors:**

1. What are the key methodological differences between ESPReg and existing feature-subspace-based continual learning methods? In particular, can the authors provide comparisons with simpler soft quadratic regularization schemes and alternative eigenvalue-weighting strategies to demonstrate the necessity of the proposed formulation?

2. What is the motivation for distilling only the highest-confidence historical class channel at each anchor? How does this strategy compare with confidence-weighted distillation over all historical class channels or conventional foreground-masked distillation? Such comparisons would help establish the distinctive contribution of CAD.

3. Can the authors provide stronger evidence that feature drift and cross-task class confusion are substantially more severe in YOLO-based detectors than in two-stage incremental detectors under controlled experimental settings? Which specific aspects of ESPReg and CAD are designed to exploit or address YOLO-specific architectural characteristics?

4. Could the authors clarify the definitions and computation procedures for Old, New, and All mAP in Tables 1 and 4? In particular, why is All mAP lower than both Old and New mAP in several cases? Are these metrics computed using different inference or evaluation protocols, or do the tables contain reporting errors?

**Flag For Ethics Review:** No

**Rating:** 2: Weak rejection

**Confidence:** 4: I have substantial direct research experience on this topic.

# Reviewer 3: tYEC

*Official Review*

**Paper Summary:**

This paper studies class-incremental object detection with YOLOv8 and proposes two components. ESPReg penalizes parameter changes along historical feature directions using calibrated covariance eigenvalues. CAD distills the teacher’s most confident old-class channel at each prediction location, weighted by its confidence. Experiments on Pascal VOC and MS COCO report improvements over several incremental detection baselines, together with analyses of feature drift and cross-task class confusion.

**Presentation:** 2: Generally understandable, but requires substantial effort in places.

**Informativeness:** 2: I learned something interesting or useful.

**Soundness:** 2: I have minor or fixable concerns about the paper’s soundness.

**Critical Strengths:**

1. Studying incremental learning with single-stage detectors is a worthwhile direction. Both components can be incorporated into the training objective without introducing an additional inference module, which makes the approach practically appealing;

2. The evaluation covers two standard benchmarks and several incremental protocols, including multi-step learning on VOC. This gives a broader picture of the method than a single base-to-new transition;

3. The component ablation and feature-drift analysis are useful. Tab. 4 suggests that ESPReg and CAD are complementary, while the within-YOLO comparison in Table 5 provides additional evidence that ESPReg reduces representation changes.

**Critical Weaknesses:**

1. I think the derivation gives a reasonable local motivation for ESPReg, but its scope needs to be stated more carefully. Eq. 3 to Eq. 6 consider a single layer while keeping its input fixed. During incremental training, upstream layers also change, so the input to that layer will generally change as well. Furthermore, Eq. 8 replaces the raw eigenvalues in the derived bound with clipped, normalized weights, assigning the same weight to all dominant directions. The resulting penalty is therefore different from the quadratic quantity originally derived. It would help to explain what connection to the drift bound remains after these changes, and how this supports the claimed stability of the full feature extractor;

2. CAD preserves one selected old-class probability at each location. I think the paper needs to explain more clearly why this choice should improve competition between old and new classes. An old-class score can remain close to the teacher’s prediction while a new-class score becomes larger, so preserving that one channel does not directly preserve the classification margin. The evidence in Fig. 1 also needs clarification: its caption describes average post-softmax confidence scores, whereas Sec 6.3 interprets the change from 73% to 42% as a reduction in misclassification rate. These are different measurements. A confusion matrix based on matched detections would provide more direct evidence that CAD reduces old-to-new classification errors;

3. Meanwhile, I have a concern about the result. Under the standard class-wise mAP definition, if old, new, and all use the same predictions and evaluation settings, All should be their class-count-weighted average. In Tab. 1, ESP-YOLO reports 80.4 for Old and 77.1 for New in the 15 + 5 setting, which would give approximately 79.6 overall, rather than the reported 75.0. Similar discrepancies appear in Tab. 3 and Tab. 4. One possible explanation is that Old/New are task-aware scores while All is task-agnostic, as the discussion in Sec. 6 suggests. If so, this distinction needs to be explicit, and the comparison with published baselines needs to use the same protocol. Reporting per-class AP from one shared task-agnostic evaluation would resolve much of this uncertainty.

**Questions For Authors:**

1. How does the single-layer drift analysis extend to changing upstream representations, and what relationship to the bound remains after eigenvalue clipping?

2. Are the values in Figure 1 confidence averages or detection error rates? Can the authors report old-to-new confusion using matched detections?

3. Are Old/New evaluated with task information while All is evaluated without it? Are the published baseline scores obtained under the same settings?

4. Do all YOLO baselines share the same initialization and pseudo-labeling procedure? How do ESPReg and CAD compare with the two simpler controls described above?

Above all, if the author can solve all of my questions. I would like to consider to rise my scores.

**Flag For Ethics Review:** No

**Rating:** 2: Weak rejection

**Confidence:** 2: I have broad familiarity with this field but little or no direct research experience on this topic.
