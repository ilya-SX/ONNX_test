import onnx
import onnxruntime
import torch
import numpy as np
import time
import skimage.io
import skimage.transform
import torchvision.transforms as transforms
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names
import detect_utils

# parameters
bs_list = np.arange(1, 11)                  # list of batch sizes
iters = 30                                  # average time for how much iterations
pass_first = 10                             # number of iterations for caching
onnx_simplified = True                      # save simplified ONNX as well
run_on_cpu = True                           # cpu computation
in_image_path = '/home/ilya.tcenov/Downloads/sheep.jpg' # use input image
out_image_path = '/home/ilya.tcenov/image_onnx.png'     # ONNX detection results saving path
simp_out_image_path = '/home/ilya.tcenov/image_onnx_simp.png'     # simplified ONNX detection results saving path

detection_threshold=0.8                     # draw results with confidence over threshold
show_ind = 0                                # image index to save from stack


def to_numpy(tensor):
    # convert to numpy
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# initialize
transform = transforms.Compose([
        transforms.ToTensor(),
    ])
if run_on_cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
times = np.zeros_like(bs_list, dtype='float')

# for each batch size
for i, bs in enumerate(bs_list):
    # select model
    if onnx_simplified:
        model_str = 'model_bs' + str(bs) + '_simp' + '.onnx'
    else:
        model_str = 'model_bs' + str(bs) + '.onnx'

    # image
    im0 = skimage.io.imread(in_image_path)
    im0 = skimage.transform.resize(im0, (600, 1200))
    im = transform(im0).float().to(device)
    im = torch.unsqueeze(im, 0)

    # concat batch
    image_stack = im
    for k in range(bs - 1):
        image_stack = torch.cat((image_stack, im), 0)

    # check model
    onnx_model = onnx.load(model_str)
    onnx.checker.check_model(onnx_model)

    # session and inputs definition
    ort_session = onnxruntime.InferenceSession(model_str)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image_stack)}

    for j in range(pass_first):
        ort_outs = ort_session.run(None, ort_inputs)    # get the predictions on the image (caching)
    st = time.time()
    for j in range(iters):
        outputs = ort_session.run(None, ort_inputs)     # get the predictions on the image
    en = time.time()
    avg_time = (en - st) / iters

    if show_ind == 0:
        boxes_ind = 0
        labels_ind = 1
        scores_ind = 2

    classes = [coco_names[o] for o in outputs[labels_ind]]
    # get score for all the predicted objects
    pred_scores = outputs[scores_ind]
    # get all the predicted bounding boxes
    pred_bboxes = outputs[boxes_ind]
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)

    # draw results
    image = detect_utils.draw_boxes(boxes, classes, outputs[labels_ind], im0)
    # save image
    if onnx_simplified:
        skimage.io.imsave(simp_out_image_path, image)
    else:
        skimage.io.imsave(out_image_path, image)

    # batch timing
    print('bs: ' + str(bs) + ' Time for forward pass: ' + str(round(avg_time, 4)) + ' [s]')
    times[i] = avg_time

# print results
print(times)


