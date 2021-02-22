import torchvision.transforms as transforms
import cv2
import torch
import timeit
import time
import numpy as np
import onnx
from onnxsim import simplify
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

COLORS = np.random.uniform(0, 1, size=(len(coco_names), 3))
transform = transforms.Compose([
    transforms.ToTensor(),
])


def save_onnx_model(model, inp, bs, dynamic_ax = False, simplify_models=True):
    # convert model to ONNX and save
    model = model.to(torch.device('cpu'))
    inp = inp.cpu()

    model_str = 'model_bs' + str(bs) + '.onnx'
    if dynamic_ax:
        torch.onnx.export(model,                        # model being run
                          inp,                          # model input (or a tuple for multiple inputs)
                          model_str,                    # where to save the model (can be a file or file-like object)
                          export_params=True,           # store the trained parameter weights inside the model file
                          opset_version=12,             # the ONNX version to export the model to
                          do_constant_folding=True,     # whether to execute constant folding for optimization
                          input_names=['input'],        # the model's input names
                          output_names=['output'],      # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})
    else:
        torch.onnx.export(model,                        # model being run
                          inp,                          # model input (or a tuple for multiple inputs)
                          model_str,                    # where to save the model (can be a file or file-like object)
                          export_params=True,           # store the trained parameter weights inside the model file
                          opset_version=12,             # the ONNX version to export the model to
                          do_constant_folding=True,     # whether to execute constant folding for optimization
                          input_names=['input'],        # the model's input names
                          output_names=['output']),     # the model's output names


    if simplify_models:
        model_simp_str = 'model_bs' + str(bs) + '_simp.onnx'
        onnx_model = onnx.load(model_str)
        onnx.checker.check_model(onnx_model)

        # convert model
        model_simp, check = simplify(onnx_model, input_shapes={"input": inp.shape})
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, model_simp_str)


def predict(image, model, device, bs, iters, pass_iters, detection_threshold, save_models, ind=0):
    # transform the image to tensor
    image = transform(image).float().to(device)
    image = torch.unsqueeze(image, 0)

    # concat batch
    image_stack = image
    for i in range(bs - 1):
        image_stack = torch.cat((image_stack, image), 0)

    # forward pass
    with torch.no_grad():
        for j in range(pass_iters):
            outputs = model(image_stack)  # get the predictions on the image (caching)

        st = time.time()
        for j in range(iters):
            outputs = model(image_stack)  # get the predictions on the image
        nd = time.time()
    avg_time = (nd - st) / iters

    # save onnx
    if save_models:
        save_onnx_model(model, image_stack, bs)

    pred_classes = [coco_names[i] for i in outputs[ind]['labels'].cpu().numpy()]
    # get score for all the predicted objects
    pred_scores = outputs[ind]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[ind]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    return boxes, pred_classes, outputs[ind]['labels'], avg_time


def draw_boxes(boxes, classes, labels, image):
    # draw detection results
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image