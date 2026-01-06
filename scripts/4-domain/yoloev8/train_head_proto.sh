python tools/train_head_proto.py \
    --model runs/yolov8l_4-domain_pretrained-yoloe_proto_rp/task-2/best.pt \
    --prototypes runs/yolov8l_4-domain_pretrained-yoloe_proto_rp/task-2/task-1-prototypes-converted.pt \
    --output runs/yolov8l_4-domain_pretrained-yoloe_proto_rp/task-2/best-train-head-proto-1.pt \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001

python tools/vis_prototypes_det.py \
    --model runs/yolov8l_4-domain_pretrained-yoloe_proto_rp/task-2/best-train-head-proto-1.pt \
    --prototypes runs/yolov8l_4-domain_pretrained-yoloe_proto_rp/task-2/prototypes.pt \
    --output runs/yolov8l_4-domain_pretrained-yoloe_proto_rp/task-2/prototypes-pred-results-train-head-proto-1

python tools/eval.py \
    --model /root/myultralytics/runs/yolov8l_4-domain_pretrained-yoloe_proto_rp/task-2/best-train-head-proto-1.pt \
    --data /root/myultralytics/runs/yolov8l_4-domain_pretrained-yoloe_proto_rp/evaluation_results/task_2_task_1_converted/dataset.yaml