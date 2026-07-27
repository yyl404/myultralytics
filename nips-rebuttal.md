# reviewer 1
1. 
For VOC 15+5:
| method | old | new | all | avg |
|---|---|---|---|---|
|EWC|63.2|47.1|59.2|55.2|
|NSGP|72.8|48.2|66.6|60.5|
|NSGP-RePRE|72.4|49.1|66.6|60.8|
|BPF|54.6|80.1|53.2|67.4|
|ESP-YOLO|80.4|77.1|75.0|78.8|

For coco 40+40:
| method | mAP | AP75 | AP50 |
|---|---|---|---|
|EWC|
|NSGP|
|NSGP-RePRE|
|BPF|
|ESP-YOLO|

2. 

The training overhead between different IOD methods
||Historical Data|Previous Checkpoint|PCA Result|Fisher Information|
|---|---|---|---|---|
|finetuning|-|-|-|-|
|EWC|-|?|-|?|
|Pseudo Labeling|-|?|-|-|
|NSGP-RePRE|?|?|?|-|
|NSGP-RePRE(YOLOv8)|?|?|?|-|
|ESP-YOLO|-|?|?|-|

Inference Latency and FLOPs against two-stage baselines
||Latency|FLOPs|
|---|---|---|
|Faster R-CNN|
|NSGP-RePRE|
|ESP-YOLO|

3. Mean and standard deviation over 3 seeds for VOC 15+5:

||old|new|all|avg|
|---|---|---|---|---|
|ESP-YOLO||||

4. 


5. 
The ablation of $\alpha$ and $\beta$
|$\alpha$ \ $\beta$|1 (old/new/all)|10 (old/new/all)|100 (old/new/all)|1000 (old/new/all)|
|---|---|---|---|---|
|1|/ /|/ /|/ /|/ /|
|10|/ /|/ /|74.5 / 78.3 / 69.2|/ /|
|100|/ /|80.3 / 77.5 / 73.6|80.4 / 77.1 / 75.0|/ /|
|1000|77.3 / 78.3 / 72.5|80.9 / 72.0 / 73.4| 81.4 / 68.6 / 73.6 |79.9 / 63.0 / 71.0|

The ablation of different distillation methods
|Distillation Channels|ESPReg|old|new|all|avg|
|---|---|---|---|---|---|
|All|√|
|Top-5|√|
|Top-3|√|
|Top-1|√|

**回复AC关于隐藏文本的事**

解释空引用

# reviewer 2

1. YOLO IOD的特点：定位+Proposal耦合，ESPReg(eq. 9)保持了localization微调的能力

2. The normalized feature drift analysis

||Faster R-CNN|YOLO|YOLO-ESP|
|---|---|---|---|
|Feature Drift||1.00 $\pm$ 0.06| 0.49 $\pm$ 0.03|

3. 在实验误差范围内可接受

4. 读一下论文

5. For VOC 15+5:

| method | old | new | all | avg |
|---|---|---|---|---|
|EWC|
|NSGP|
|NSGP-RePRE|
|BPF|
|ESP-YOLO|80.4|77.1|75.0|78.8|

For coco 40+40:
| method | mAP | AP75 | AP50 |
|---|---|---|---|
|EWC|
|NSGP|
|NSGP-RePRE|
|BPF|
|ESP-YOLO|

6. The training overhead between different IOD methods

||Historical Data|Previous Checkpoint|PCA Result|Fisher Information|
|---|---|---|---|---|
|finetuning|-|-|-|-|
|L2-Normalized|-|?|-|-|
|EWC|-|?|-|?|
|Pseudo Labeling|-|?|-|-|
|NSGP-RePRE|?|?|?|-|
|NSGP-RePRE(YOLOv8)|?|?|?|-|
|ESP-YOLO|-|?|?|-|

7. The ablation of $\alpha$ and $\beta$

|$\alpha$ \ $\beta$|1 (old/new/all)|10 (old/new/all)|100 (old/new/all)|1000 (old/new/all)|
|---|---|---|---|---|
|1|/ /|/ /|/ /|/ /|
|10|/ /|/ /|74.5 / 78.3 / 69.2|/ /|
|100|/ /|80.3 / 77.5 / 73.6|80.4 / 77.1 / 75.0|/ /|
|1000|77.3 / 78.3 / 72.5|80.9 / 72.0 / 73.4| 81.4 / 68.6 / 73.6 |79.9 / 63.0 / 71.0|

# reviewer 3

1. For VOC 15+5:

| method | old | new | all | avg |
|---|---|---|---|---|
|EWC|
|NSGP|
|NSGP-RePRE|
|BPF|
|ESP-YOLO|80.4|77.1|75.0|78.8|

For coco 40+40:
| method | mAP | AP75 | AP50 |
|---|---|---|---|
|EWC|
|NSGP|
|NSGP-RePRE|
|BPF|
|ESP-YOLO|

2. The normalized feature drift analysis

||Faster R-CNN|YOLO|YOLO-ESP|
|---|---|---|---|
|Feature Drift|||

3. The upper bound is indeed from YOLOv8 baseline.