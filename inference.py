from networks import FaceBox
from encoderl import DataEncoder

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import time


def detect(im, cuda):
    h, w, _ = im.shape
    #dim_diff = np.abs(h - w)
    #pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    #pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    #input_img = np.pad(im, pad, 'constant', constant_values=128)
    input_img = cv2.resize(im,(1024,1024))
    input_img = input_img/255.
    t0 = time.time()

    im_tensor = torch.from_numpy(input_img.transpose((2,0,1))).float()

    if cuda:
        im_tensor = im_tensor.cuda()

    loc, conf = net(im_tensor.unsqueeze(0))

    if cuda:
        loc.data = loc.data.cpu()
        conf = conf.cpu()

    boxes, labels, probs = data_encoder.decode(loc.data.squeeze(0),
                                                F.softmax(conf.squeeze(0), dim=1))
    t1 = time.time();
    total = t1-t0
    print("Run took %s\n" % total) 
    if probs[0] != 0:
        boxes = boxes.numpy()
        probs = probs.detach().numpy()
        if h <= w:
            boxes[:,1] = boxes[:,1]*h#-pad1
            boxes[:,3] = boxes[:,3]*h#-pad1
            boxes[:,0] = boxes[:,0]*w
            boxes[:,2] = boxes[:,2]*w
        else:
            boxes[:,1] = boxes[:,1]*w
            boxes[:,3] = boxes[:,3]*w
            boxes[:,0] = boxes[:,0]*h#-pad1
            boxes[:,2] = boxes[:,2]*h#-pad1

    return boxes, probs

def testIm(im, cuda):
    if im is None:
        print("can not open image:")
        return
    h,w,_ = im.shape
    boxes, probs = detect(im, cuda)

    if probs[0] == 0:
        print('There is no face in the image')
        exit()
    for i, (box) in enumerate(boxes):
        print('i=', i, 'box=', box)
        if probs[i] > 0.8:
            x1 = int(box[0])
            x2 = int(box[2])
            y1 = int(box[1])
            y2 = int(box[3])
            cv2.rectangle(im,(x1,y1+4),(x2,y2),(0,255,0),2)
            cv2.putText(im, str(probs[i]), (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,255,0))
    cv2.imshow('photo', im)
    cv2.imwrite('picture/1111.jpg', im)
    cv2.waitKey(0)


if __name__ == '__main__':
    cuda = 1
    net = FaceBox()
    net.load_state_dict(torch.load('weight/faceboxes.pt', map_location=lambda storage, loc:storage), strict=False) 
    net.eval()
    if cuda:
        net.cuda()
    data_encoder = DataEncoder()

    # given image path, predict and show
    root_path = "/home/sacha/dev/ifaceengine/test_images/"
    picture = 'The-cast-of-Les-Miserables.jpg'#Face NOT detected.JPG'
    im = cv2.imread(root_path + picture)
    for i in range(0, 100):
      im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
      testIm(im, cuda)

