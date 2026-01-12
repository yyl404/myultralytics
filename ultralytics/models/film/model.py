from ultralytics.engine.model import Model
from ultralytics.models.film.train import FiLMTrainer
from ultralytics.models.film.val import FiLMValidator
from ultralytics.nn.tasks_film import DetectionModelFiLM

class YOLOFiLM(Model):
    """
    YOLO-FiLM 接口类。
    """
    @property
    def task_map(self):
        """映射到自定义的 Trainer 和 Validator"""
        return {
            "detect": {
                "model": DetectionModelFiLM,
                "trainer": FiLMTrainer,
                "validator": FiLMValidator,
            }
        }