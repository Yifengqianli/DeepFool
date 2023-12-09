import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
# from torch.autograd.gradcheck import zero_gradients

def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: source_images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, encoded_images estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")


    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:               # 当前标签仍然没有改变并且没有到最大寻找次数

        pert = np.inf                                       # 先将扰动设置为一个很大的值
        fs[0, I[0]].backward(retain_graph=True)             # 反向传播，计算当前梯度，连续执行两次backward，并且保留backward后的中间参数
        grad_orig = x.grad.data.cpu().numpy().copy()        # 获取原始梯度

        for k in range(1, num_classes):                     # 遍历十个类
            # zero_gradients(x)
            if x.grad is not None:                          # 将梯度清空
                x.grad.zero_()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()     # 计算当前梯度

            # set encoded_images w_k and encoded_images f_k
            w_k = cur_grad - grad_orig                      # 计算新的weight
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            # 计算第k个标签需要的扰动
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)                     # 计算累计扰动

        # 将扰动添加到对抗样本中
        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)        # 更新x
        fs = net.forward(x)                                 # 根据对抗样本获取网络输出
        k_i = np.argmax(fs.data.cpu().numpy().flatten())    # 对抗样本的标签

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image