import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from models.resnet import Model
from thop import profile


def pre_process(image_path):
    """
    # use the same preprocess to handle images
    :param image_path: the english path of the inference image
    :return:
    """
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data = transform(img)
    # inference in torch.nn need 4 dimensional, the first is the batch, add 1
    data = torch.unsqueeze(data, dim=0)
    return img, data


def inference(image_path, weights_path):
    """

    :param image_path:  the path of image
    :param weights_path: the model path
    :return:
    """
    img, data = pre_process(image_path)
    data = torch.autograd.Variable(data.cuda())
    model = Model()
    model.load_state_dict(torch.load(weights_path))
    output = model(data)
    _, prediction = torch.max(output, 1)
    prediction = prediction.cpu().item()
    fig, ax = plt.subplots(1, 2)
    ax[0, 0].imshow(img)
    ax[0, 0].set_title('origin')
    ax[0, 1].imshow(data)
    ax[0, 1].set_title('prediction:' + str(prediction))
    plt.show()

    return prediction


def cal_params(model):
    """
    calculate the Flops and Params of the model
    :param model:
    :return:
    """
    dummy_input = torch.rand(1, 3, 69, 69).cpu()  # dummy input is a fake input
    flops, params = profile(model, inputs=(dummy_input,))
    print('-' * 50)
    print(f'Flops = {str(flops / 1000 ** 3)}G')
    print(f'Params = {str(params / 1000 ** 2)}M')


if __name__ == "__main__":
    img_path = None
    weights_path = None
    # inference(image_path=img_path, weights_path=weights_path)

    # model = models.resnet50(num_classes=10, pretrained=False).cpu()
    # cal_params(model)


