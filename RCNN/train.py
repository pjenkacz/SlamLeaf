from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
import os

def main():
    register_coco_instances("my_train", {}, os.path.join("annotations_train.json"), os.path.join("images","train"))
    register_coco_instances("my_val",   {}, os.path.join("annotations_val.json"),   os.path.join("images","val"))

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # From-scratch + GN
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.RESNETS.NORM = "GN"
    cfg.MODEL.FPN.NORM = "GN"
    cfg.MODEL.ROI_BOX_HEAD.NORM = "GN"


    # Wariant A: gęstsze (8/pozycję)
    #cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64,96],[128,192],[256,320],[384,448],[512,640]]
    #cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5,1.0,2.0,3.0]] * 5

    # # Wariant B: lżejsze (3/pozycję)
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64],[128],[256],[384],[640]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5,1.0,2.0]] * 5

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 30
    cfg.DATASETS.TRAIN = ("my_train",)
    cfg.DATASETS.TEST  = ("my_val",)
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.INPUT.MIN_SIZE_TRAIN = (640, 704, 768, 832, 896)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST  = 800
    cfg.INPUT.MAX_SIZE_TEST  = 1333

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 24000
    cfg.SOLVER.STEPS = (18000, 22000)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WEIGHT_DECAY = 1e-4
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.LOGGING_INTERVAL = 50
    cfg.TEST.EVAL_PERIOD = 2000
    cfg.OUTPUT_DIR = "./output_frcnn_scratch_anchors"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator("my_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "my_val")
    print("EVAL:", evaluator.evaluate(trainer.model, val_loader))

if __name__ == "__main__":
    main()