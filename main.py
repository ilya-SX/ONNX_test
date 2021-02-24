import torchvision
import skimage.io
import skimage.transform
import torch
import numpy as np
import detect_utils

# parameters
bs_list = np.arange(1, 11)              # list of batch sizes
iters = 30                              # average time for how much iterations
pass_first = 10                         # number of iterations for caching
save_onnx_models = False                # convert to ONNX and save all models (for each batch size)
run_on_cpu = True                       # cpu computation
in_image_path = '/home/ilya.tcenov/Downloads/sheep.jpg' # use input image
out_image_path = '/home/ilya.tcenov/image.png'          # detection results saving path

# initialize
times = np.zeros_like(bs_list, dtype='float')
if run_on_cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model
pytorch_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
pytorch_model.eval()
pytorch_model.to(device)

# image
im = skimage.io.imread(in_image_path)
im = skimage.transform.resize(im, (600, 1200))

# for each batch size
for i, bs in enumerate(bs_list):
    # pred
    boxes, classes, labels, avg_time = detect_utils.predict(im, pytorch_model, device, bs=bs, iters=iters,
                                                            pass_iters=pass_first, detection_threshold=0.8,
                                                            save_models=save_onnx_models, ind=0)
    # batch timing
    times[i] = avg_time
    print('bs: ' + str(bs) + ' Time for forward pass: ' + str(round(avg_time, 4)) + ' [s]')
    # draw results
    image = detect_utils.draw_boxes(boxes, classes, labels, im)
    # save image
    skimage.io.imsave(out_image_path, image)

# print results
print(times)

