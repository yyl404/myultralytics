from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import VSPRegDetectionTrainer
import os


def test_vspreg_trainer():
    """测试VSPRegDetectionTrainer在不同条件下的表现"""
    
    # 基础配置
    base_config = {
        "data": "/root/data/datasets/VOC_inc_15_5/task_2_cls_5/dataconfig.yaml",
        "epochs": 1,
        "batch": 16,
        "workers": 8,
        "device": 0,
        "trainer": VSPRegDetectionTrainer,
    }
    
    # 样本数据路径
    sample_images = "/root/data/datasets/VOC_inc_15_5/task_1_cls_15/images/test"
    sample_labels = "/root/data/datasets/VOC_inc_15_5/task_1_cls_15/labels/test"
    
    # PCA缓存路径配置
    cache_dir = "runs/detect"
    os.makedirs(cache_dir, exist_ok=True)
    
    # 测试用例列表
    test_cases = [
        # {
        #     "name": "Test 1: 既没有指定sample_images也没有指定sample_labels",
        #     "config": {**base_config, "model": "yolov8m.pt"}
        # },
        # {
        #     "name": "Test 2: 指定了sample_images而没有指定sample_labels",
        #     "config": {
        #         **base_config, 
        #         "model": "yolov8m.pt",
        #         "sample_images": sample_images
        #     }
        # },
        # {
        #     "name": "Test 3: 指定了sample_labels而没有指定sample_images",
        #     "config": {
        #         **base_config, 
        #         "model": "yolov8m.pt",
        #         "sample_labels": sample_labels
        #     }
        # },
        # {
        #     "name": "Test 4: 基于Test 3，指定freeze=[1, 2, 3]",
        #     "config": {
        #         **base_config, 
        #         "model": "yolov8m.pt",
        #         "sample_images": sample_images,
        #         "sample_labels": sample_labels,
        #         "freeze": [1, 2, 3]
        #     }
        # },
        # {
        #     "name": "Test 5: 基于Test 3，指定freeze=1",
        #     "config": {
        #         **base_config, 
        #         "model": "yolov8m.pt",
        #         "sample_images": sample_images,
        #         "sample_labels": sample_labels,
        #         "freeze": 1
        #     }
        # },
        # {
        #     "name": "Test 6: 基于Test 3，指定projection_layers=[1, 2, 3]",
        #     "config": {
        #         **base_config, 
        #         "model": "yolov8m.pt",
        #         "sample_images": sample_images,
        #         "sample_labels": sample_labels,
        #         "projection_layers": [1, 2, 3]
        #     }
        # },
        {
            "name": "Test 7: 基于Test 3，指定pca_cache_save_path",
            "config": {
                **base_config, 
                "model": "yolov8m.pt",
                "sample_images": sample_images,
                "sample_labels": sample_labels,
                "pca_cache_save_path": f"{cache_dir}/test_7_pca_cache.joblib"
            }
        },
        {
            "name": "Test 8: 基于Test 3，指定pca_cache_load_path（使用Test 7的缓存）",
            "config": {
                **base_config, 
                "model": "yolov8m.pt",
                "sample_images": sample_images,
                "sample_labels": sample_labels,
                "pca_cache_load_path": f"{cache_dir}/test_7_pca_cache.joblib"
            }
        },
        {
            "name": "Test 9: 基于Test 3，同时指定pca_cache_load_path和pca_cache_save_path",
            "config": {
                **base_config, 
                "model": "yolov8m.pt",
                "sample_images": sample_images,
                "sample_labels": sample_labels,
                "pca_cache_load_path": f"{cache_dir}/test_7_pca_cache.joblib",
                "pca_cache_save_path": f"{cache_dir}/test_9_pca_cache.joblib"
            }
        }
    ]
    
    print("开始VSPRegDetectionTrainer测试...")
    print("=" * 80)
    
    # 用于跟踪缓存文件的存在性
    cache_files_created = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{test_case['name']}")
        print("-" * 60)
        
        try:
            # 创建模型
            model = YOLO(test_case['config']['model'])
            
            # 准备训练配置
            train_config = {k: v for k, v in test_case['config'].items() if k != 'model'}
            
            print(f"训练配置: {train_config}")
            
            # 显示缓存路径信息
            if 'pca_cache_load_path' in train_config:
                print(f"📁 将加载缓存文件: {train_config['pca_cache_load_path']}")
            
            if 'pca_cache_save_path' in train_config:
                print(f"💾 将保存缓存文件: {train_config['pca_cache_save_path']}")
            
            # 开始训练
            results = model.train(**train_config)
            
            # 检查缓存文件是否被创建（对于save_path测试）
            if 'pca_cache_save_path' in train_config:
                cache_path = train_config['pca_cache_save_path']
                if os.path.exists(cache_path):
                    print(f"✅ 缓存文件已创建: {cache_path}")
                    cache_files_created.append(cache_path)
                else:
                    print(f"⚠️  缓存文件未创建: {cache_path}")
            
            print(f"✅ Test {i} 完成成功")
            
        except Exception as e:
            print(f"❌ Test {i} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("-" * 60)
    
    print("\n" + "=" * 80)
    print("所有测试完成!")
    
    # 显示创建的缓存文件
    if cache_files_created:
        print(f"\n创建的PCA缓存文件:")
        for cache_file in cache_files_created:
            if os.path.exists(cache_file):
                file_size = os.path.getsize(cache_file)
                print(f"  - {cache_file} ({file_size} bytes)")
            else:
                print(f"  - {cache_file} (文件不存在)")
    else:
        print("\n没有创建任何PCA缓存文件")
    
    # 显示runs/detect目录下的所有缓存文件
    print(f"\nruns/detect目录下的所有文件:")
    if os.path.exists(cache_dir):
        for item in os.listdir(cache_dir):
            item_path = os.path.join(cache_dir, item)
            if os.path.isfile(item_path):
                file_size = os.path.getsize(item_path)
                print(f"  - {item} ({file_size} bytes)")
            else:
                print(f"  - {item}/ (目录)")
    else:
        print(f"  {cache_dir} 目录不存在")


if __name__ == "__main__":
    test_vspreg_trainer()