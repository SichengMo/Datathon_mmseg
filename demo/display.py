from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = 'configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K.py'
checkpoint_file = 'checkpoints/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/iter_160000.pth'

# 通过配置文件和模型权重文件构建模型
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# 对单张图片进行推理并展示结果
img = 'data/dataset/images/train/000238.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_segmentor(model, img)
# 在新窗口中可视化推理结果
model.show_result(img, result, show=True)
# 或将可视化结果存储在文件中
# 你可以修改 opacity 在(0,1]之间的取值来改变绘制好的分割图的透明度
model.show_result(img, result, out_file='result.jpg', opacity=0.5)

