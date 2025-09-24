# import pickle
# from mmengine.config import Config
# from mmengine.runner import Runner

# def main():
#     config_file = 'configs/drad_mvxnet/drad_mvxnet_fpn_custom.py'
#     checkpoint_file = 'work_dirs/drad_mvxnet_fpn_custom_9_23_3/best_Kitti metric_pred_instances_3d_KITTI_Overall_3D_AP40_moderate_epoch_80.pth'
#     out_file = 'work_dirs/my_test_results/results.pkl'

#     # 读取 config
#     cfg = Config.fromfile(config_file)
#     cfg.load_from = checkpoint_file
#     cfg.work_dir = './work_dirs/tmp_test'

#     # 构建 runner
#     runner = Runner.from_cfg(cfg)

#     # 执行 test，得到结果
#     results = runner.test()
#      # 保存 raw predictions
#     out_file = 'work_dirs/my_test_results/raw_results.pkl'
#     with open(out_file, 'wb') as f:
#         pickle.dump(results, f)
#     print(f'✅ Raw results saved to {out_file}')

#     # 保存为 pkl
#     with open(out_file, 'wb') as f:
#         pickle.dump(results, f)

#     print(f'Results saved to {out_file}')

# if __name__ == '__main__':
#     main()
import mmcv
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet3d.apis import init_model, inference_detector
import pickle
import os

def main():
    cfg_file = 'configs/drad_mvxnet/drad_mvxnet_fpn_custom.py'
    checkpoint_file = 'work_dirs/drad_mvxnet_fpn_custom_9_23_3/best_Kitti metric_pred_instances_3d_KITTI_Overall_3D_AP40_moderate_epoch_80.pth'
    out_file = 'work_dirs/my_test_results/pred_instances.pkl'

    # 加载配置和模型
    cfg = Config.fromfile(cfg_file)
    model = init_model(cfg, checkpoint_file, device='cuda:0')

    # 构建 dataloader
    data_loader = Runner.build_dataloader(cfg.test_dataloader)

    results = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            preds = model.test_step(data)
            results.extend(preds)  # preds 是 list

    # 保存成 pkl
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"预测结果已保存到 {out_file}")

if __name__ == '__main__':
    main()

