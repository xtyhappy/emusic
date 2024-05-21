import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import numpy as np
import cv2
from PIL import Image


# modify the Batch Normalization layer
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module


# prepare all the labels
def load_labels():
    # scene category relevant
    file_name_category = 'categories_places365.txt'
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    # file_name_IO = 'IO_places365.txt'
    # with open(file_name_IO) as f:
    #     lines = f.readlines()
    #     labels_IO = []
    #     for line in lines:
    #         items = line.rstrip().split()
    #         labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    # labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    W_attribute = np.load(file_name_W)

    # Load new attribute words
    new_attribute_file = 'scene_attribute.txt'
    with open(new_attribute_file) as new_attr_file:
        new_attribute_words = [line.strip() for line in new_attr_file]

    # return classes, labels_IO, labels_attribute, W_attribute, new_attribute_words
    return classes, labels_attribute, W_attribute, new_attribute_words


# feature outputs from the middle layer
def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))


# load the image transformer
def returnTF():
    # a converter for image preprocessing
    tf = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


# load the model
def load_model():
    model_file = 'whole_wideresnet18_places365_python36.pth.tar'
    import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint.state_dict().items()}
    model.load_state_dict(state_dict)

    # hacky way to deal with the upgraded batchnorm2D and avgpool layers
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

    # model.eval()

    model.eval()

    # hook the feature extractor
    features_names = ['layer4', 'avgpool']  # last conv layer
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model


if __name__ == "__main__":
    # load the labels and model
    classes, labels_attribute, W_attribute, new_attribute_words = load_labels()
    # classes, labels_IO, labels_attribute, W_attribute, new_attribute_words = load_labels()

    features_blobs = []
    model = load_model()

    # image transformer
    tf = returnTF()

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax < 0] = 0

    # image for test
    img = Image.open('input.jpg')
    input_img = V(tf(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # # output the scene categories
    # print('--SCENE CATEGORIES:')
    # for i in range(0, 5):
    #     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # # output the IO prediction
    # io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
    # if io_image < 0.5:
    #     print('--TYPE OF ENVIRONMENT: indoor')
    # else:
    #     print('--TYPE OF ENVIRONMENT: outdoor')


    # output the scene attributes
    responses_attribute = W_attribute.dot(features_blobs[1])
    idx_a = np.argsort(responses_attribute)
    print('--SCENE ATTRIBUTES:')
    filtered_attributes = [labels_attribute[idx_a[i]] for i in range(-1, -20, -1) if labels_attribute[idx_a[i]] in new_attribute_words]

    # take the first five elements
    filtered_attributes = filtered_attributes[:5]
    scene_words = ', '.join(filtered_attributes)

    if filtered_attributes:
        print(scene_words)
    else:
        print('No matching attributes found.')

    with open("scene_words.txt", "w") as file:
        file.write(scene_words)


# def part2_main():
#     # load the labels and model
#     classes, labels_attribute, W_attribute, new_attribute_words = load_labels()
#     # classes, labels_IO, labels_attribute, W_attribute, new_attribute_words = load_labels()
#
#     features_blobs = []
#     model = load_model()
#
#     # image transformer
#     tf = returnTF()
#
#     # get the softmax weight
#     params = list(model.parameters())
#     weight_softmax = params[-2].data.numpy()
#     weight_softmax[weight_softmax < 0] = 0
#
#     # image for test
#     img = Image.open('input.jpg')
#     input_img = V(tf(img).unsqueeze(0))
#
#     # forward pass
#     logit = model.forward(input_img)
#     h_x = F.softmax(logit, 1).data.squeeze()
#     probs, idx = h_x.sort(0, True)
#     probs = probs.numpy()
#     idx = idx.numpy()
#
#     # # output the scene categories
#     # print('--SCENE CATEGORIES:')
#     # for i in range(0, 5):
#     #     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
#
#     # # output the IO prediction
#     # io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
#     # if io_image < 0.5:
#     #     print('--TYPE OF ENVIRONMENT: indoor')
#     # else:
#     #     print('--TYPE OF ENVIRONMENT: outdoor')
#
#     # output the scene attributes
#     responses_attribute = W_attribute.dot(features_blobs[1])
#     idx_a = np.argsort(responses_attribute)
#     print('--SCENE ATTRIBUTES:')
#     filtered_attributes = [labels_attribute[idx_a[i]] for i in range(-1, -20, -1) if
#                            labels_attribute[idx_a[i]] in new_attribute_words]
#
#     # take the first five elements
#     filtered_attributes = filtered_attributes[:5]
#     return filtered_attributes
