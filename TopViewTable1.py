# coding: utf-8

import imageio
import numpy as np
from PIL import Image


frame_f = imageio.imread("front_undistort.jpg")
frame_b = imageio.imread("rear_undistort.jpg")
frame_l = imageio.imread("left_undistort.jpg")
frame_r = imageio.imread("right_undistort.jpg")
im = Image.fromarray(frame_r)

####forward
H_f = np.array([[7.16484249e-001, -1.21329653e+000, 3.95333954e+002],
                [6.18596748e-002, -1.09690875e-001, 1.50242676e+002],
                [2.14985557e-005, -1.84673560e-003, 1]])
####rear(back)
H_b = np.array([[5.67742646e-001, -9.36438203e-001, 4.62838440e+002],
                [4.23971266e-002, -4.45890613e-002, -9.08954544e+001],
                [5.94135827e-006, -1.42084842e-003, 1]])

#####left
H_l = np.array([[-1.82098770e+000, -1.05679226e+000, 1.32148901e+003],
                [-9.12221596e-002, -6.53561205e-002, 2.80831299e+002],
                [-2.73178820e-003, 3.40395927e-005, 1]])
####right
H_r = np.array([[-1.94364822e+000, -1.13981271e+000, 1.39862231e+003],
                [-1.12904012e-001, -7.77252018e-002, -1.34969070e+002],
                [-2.94386805e-003, 3.57765421e-005, 1]])


f_l_max_v = 465
f_l_max_u = 265
f_l_min_v = 0
f_l_min_u = 0

b_l_max_v = 1280
b_l_max_u = 265
b_l_min_v = 815
b_l_min_u = 0

f_r_max_v = 465
f_r_max_u = 720
f_r_min_v = 0
f_r_min_u = 455

b_r_max_v = 1280
b_r_max_u = 720
b_r_min_v = 815
b_r_min_u = 455


f_min_v = 0
f_max_v = 465
f_min_u = 0
f_max_u = 720

b_min_v = 815
b_max_v = 1280
b_min_u = 0
b_max_u = 720

l_min_v = 0
l_max_v = 1280
l_min_u = 0
l_max_u = 265

r_min_v = 0
r_max_v = 1280
r_min_u = 455
r_max_u = 720

'' \
'frame:inputs(image)' \
'' \
'' \
'' \
'' \
'' \
''
def get_cross_trans_fb_four(frame, H, min_u, max_u, min_v, max_v, cross_min_u1,
                            cross_max_u1, center0, center1, center2, num):
    weights = []
    out_put_pxiels = []
    in_put_pxiels = []
    for u in range(min_u, max_u):
        for v in range(min_v, max_v):
            if u <= cross_min_u1:
                rate = get_rate(center0, center1, u, v)[0]
            #                 rate = get_rate_1(center1, u, v)[0]
            elif u >= cross_max_u1:
                rate = get_rate(center0, center2, u, v)[0]
            #                 rate = get_rate_1(center2, u, v)[0]
            else:
                rate = 1
            out_put = np.array([u, v, 1])
            in_put = np.dot(H, out_put)
            frame_x, frame_y = in_put[1] / in_put[2], in_put[0] / in_put[2]
            if 0 <= round(frame_x) < frame.shape[0] and 0 <= round(frame_y) < frame.shape[1]:
                weight = get_cross_weight_four(frame, frame_x, frame_y, rate, num)
            else:
                weight = (num, 0, 0, 0, 0)
            weights.append(weight)
            out_put_pxiel = (v, u)
            out_put_pxiels.append(out_put_pxiel)
            in_put_pxiel = (frame_x, frame_y)
            in_put_pxiels.append(in_put_pxiel)
    return weights, out_put_pxiels, in_put_pxiels



def get_gray_level_four(x, y, weight):
    z = np.array([0, 0, 0, 0])
    w = np.array([weight[1], weight[2], weight[3], weight[4]])
    gray_levels = []
    if (w == z).all():
        gray_levels = [0, 0, 0]
    else:
        if weight[0] == 0:
            frame = frame_f
        if weight[0] == 1:
            frame = frame_l
        if weight[0] == 2:
            frame = frame_b
        if weight[0] == 3:
            frame = frame_r
        w = w.reshape(2, 2)
        for i in range(0, 3):
            f = frame[int(x):int(x) + 2, int(y):int(y) + 2, i]
            gray_level = np.sum(np.multiply(w, f))
            gray_levels.append(gray_level)
    return np.array(gray_levels)



def get_cross_weight_four(frame, in_x, in_y, rate, num):
    intSx, intSy = int(in_x), int(in_y)
    frame_width, frame_height = frame.shape[0], frame.shape[1]
    if 0 <= intSx < frame_width - 2 and 0 <= intSy < frame_height - 2:
        x1, x2 = intSx, intSx + 1
        y1, y2 = intSy, intSy + 1
        a11 = (x2 - in_x) * (y2 - in_y) * rate
        a12 = (x2 - in_x) * (in_y - y1) * rate
        a21 = (in_x - x1) * (y2 - in_y) * rate
        a22 = (in_x - x1) * (in_y - y1) * rate
        weight = (num, a11, a12, a21, a22)
    else:
        weight = (num, 1, 0, 0, 0)
    return weight



def remap_four(weights, out_put_pxiels, in_put_pxiels):
    out_put_img = np.zeros((1280, 720, frame_f.shape[2]), dtype=frame_f.dtype)
    for weight, out_put, in_put in zip(weights, out_put_pxiels, in_put_pxiels):
        out_x, out_y = out_put[0], out_put[1]
        #         print (in_put)
        in_x, in_y = in_put[0], in_put[1]

        out_put_img[out_x, out_y] = out_put_img[out_x, out_y] + get_gray_level_four(in_x, in_y, weight)
    return out_put_img



'''
获取交叉区域的权重(基于欧式距离)
center1 图像1中心点
center2 图像2中心点
x       图像1/2交叉区域点x
y       图像1/2交叉区域点y

'''
def get_rate(center1, center2, x, y):
    dis1 = np.sqrt(pow(center1[0] - x, 2) + pow(center1[1] - y, 2))
    dis2 = np.sqrt(pow(center2[0] - x, 2) + pow(center2[1] - y, 2))
    rate1 = dis1 / (dis1 + dis2)
    rate2 = dis2 / (dis1 + dis2)
    cross_weight = (rate1, rate2)
    return cross_weight



def get_rate_1(center1, x, y):
    rate1 = abs(center1[0] - x) / center1[1]
    rate2 = 1 - rate1
    cross_weight = (rate1, rate2)
    return cross_weight



centerf = [360, 465]
centerb = [360, 815]
centerl = [265, 640]
centerr = [455, 640]
center1 = [265, 265]
center2 = [455, 265]


def get_cross_trans_rl_four(frame, H, min_u, max_u, min_v, max_v, cross_min_v1,
                            cross_max_v1, center0, center1, center2, num):
    weights = []
    out_put_pxiels = []
    in_put_pxiels = []
    for u in range(min_u, max_u):
        for v in range(min_v, max_v):
            if v <= cross_min_v1:
                rate = get_rate(center0, center1, u, v)[0]
            #                 rate = get_rate_1( center1, u, v)[1]
            elif v >= cross_max_v1:
                rate = get_rate(center0, center2, u, v)[0]
            #                 rate = get_rate_1( center2, u, v)[1]
            else:
                rate = 1.1
            out_put = np.array([u, v, 1])
            in_put = np.dot(H, out_put)
            frame_x, frame_y = in_put[1] / in_put[2], in_put[0] / in_put[2]
            if 0 <= round(frame_x) < frame.shape[0] and 0 <= round(frame_y) < frame.shape[1]:
                weight = get_cross_weight_four(frame, frame_x, frame_y, rate, num)
            else:
                weight = (num, 0, 0, 0, 0)
            weights.append(weight)
            out_put_pxiel = (v, u)
            out_put_pxiels.append(out_put_pxiel)
            in_put_pxiel = (frame_x, frame_y)
            in_put_pxiels.append(in_put_pxiel)
    return weights, out_put_pxiels, in_put_pxiels


weights_b1, out_put_pxiels_b1, in_put_pxiels_b1 = get_cross_trans_fb_four(frame_b, H_b, b_min_u, b_max_u, b_min_v,
                                                                          b_max_v, b_l_max_u, b_r_min_u, centerb,
                                                                          centerl, centerr, 2)
out_put_img_b1 = remap_four(weights_b1, out_put_pxiels_b1, in_put_pxiels_b1)
bird = Image.fromarray(out_put_img_b1)
bird.show()
bird.save("birdb.jpg")


weights_f1, out_put_pxiels_f1, in_put_pxiels_f1 = get_cross_trans_fb_four(frame_f, H_f, f_min_u, f_max_u, f_min_v,
                                                                          f_max_v, f_l_max_u, f_r_min_u, centerb,
                                                                          centerl, centerr, 0)
out_put_img_f1 = remap_four(weights_f1, out_put_pxiels_f1, in_put_pxiels_f1)
bird = Image.fromarray(out_put_img_f1)
bird.show()
bird.save("birdf.jpg")


weights_r1, out_put_pxiels_r1, in_put_pxiels_r1 = get_cross_trans_rl_four(frame_r, H_r, r_min_u, r_max_u, r_min_v,
                                                                          r_max_v, f_r_max_v, b_r_min_v, centerr,
                                                                          centerf, centerb, 3)
out_put_img_r1 = remap_four(weights_r1, out_put_pxiels_r1, in_put_pxiels_r1)
bird = Image.fromarray(out_put_img_r1)
bird.show()
bird.save("birdr.jpg")


weights_l1, out_put_pxiels_l1, in_put_pxiels_l1 = get_cross_trans_rl_four(frame_l, H_l, l_min_u, l_max_u, l_min_v,
                                                                          l_max_v, f_l_max_v, b_l_min_v, centerl,
                                                                          centerf, centerb, 1)
out_put_img_l1 = remap_four(weights_l1, out_put_pxiels_l1, in_put_pxiels_l1)
bird = Image.fromarray(out_put_img_l1)
bird.show()
bird.save("birdl.jpg")


weights = weights_f1 + weights_l1 + weights_b1 + weights_r1
in_put_pxiels = in_put_pxiels_f1 + in_put_pxiels_l1 + in_put_pxiels_b1 + in_put_pxiels_r1
out_put_pxiels = out_put_pxiels_f1 + out_put_pxiels_l1 + out_put_pxiels_b1 + out_put_pxiels_r1



out_put_img_1 = out_put_img_l1 + out_put_img_f1 + out_put_img_r1 + out_put_img_b1
bird = Image.fromarray(out_put_img_1)
bird.show()


bird.save("bird_overall.jpg")


weight_dict = {}
cor_dict = {}


for out_put, in_put, weight in zip(out_put_pxiels, in_put_pxiels, weights):
    if out_put not in weight_dict.keys():
        weight_dict[out_put] = []
        cor_dict[out_put] = []
        weight_dict[out_put].append(weight)
        cor_dict[out_put].append(in_put)

    else:
        weight_dict[out_put].append(weight)
        cor_dict[out_put].append(in_put)
#         print (out_put)



# print (len(weight_dict[0,0]))
print(cor_dict[0, 0][0] + cor_dict[0, 0][1])
print(len(cor_dict[(0, 0)]))
# for i in cor_dict.keys():
#     print (len(cor_dict[i]))



dt = np.dtype((np.float64, (1, 5)))
dt1 = np.dtype((np.float64, (1, 2)))
weight_table1 = np.ndarray(shape=(1280, 720), dtype=dt)
cor_table1 = np.ndarray(shape=(1280, 720), dtype=dt1)
# for out_put, in_put, weight in zip(out_put_pxiels, in_put_pxiels, weights):
#     weight_table[out_put[0],out_put[1]] = weight
#     cor_table[out_put[0],out_put[1]] = in_put
for out_put in cor_dict.keys():

    if len(cor_dict[out_put]) == 1:
        weight_table1[out_put[0], out_put[1]] = weight_dict[out_put]
        cor_table1[out_put[0], out_put[1]] = cor_dict[out_put]
    else:
        weight_table1[out_put[0], out_put[1]] = weight_dict[out_put][0]  # + weight_dict[out_put][1]
        cor_table1[out_put[0], out_put[1]] = cor_dict[out_put][0]  # + cor_dict[out_put][1]


# print (weight_table)
print(cor_table)


def remap(weights, out_put_pxiels, in_put_pxiels):
    out_put_img = np.zeros((1280, 720, frame_f.shape[2]), dtype=frame_f.dtype)
    for weight, out_put, in_put in zip(weights, out_put_pxiels, in_put_pxiels):
        out_x, out_y = out_put[0], out_put[1]
        #         print (in_put)
        in_x, in_y = in_put[0], in_put[1]
        if (out_put_img[out_x, out_y] == np.array([0, 0, 0])).all():
            out_put_img[out_x, out_y] = get_gray_level_four(in_x, in_y, weight)
        else:
            #             print (out_x, out_y)
            out_put_img[out_x, out_y] = out_put_img[out_x, out_y] + get_gray_level_four(in_x, in_y, weight)
    #         out_put_img[out_x, out_y] = out_put_img[out_x, out_y] + get_gray_level_four( in_x, in_y, weight)
    return out_put_img



weights_b2, out_put_pxiels_b2, in_put_pxiels_b2 = get_cross_trans_fb_four(frame_b, H_b, b_min_u, b_max_u, b_min_v,
                                                                          b_max_v, b_l_max_u, b_r_min_u, centerb,
                                                                          center1, center2, 2)
out_put_img_b2 = remap_four(weights_b2, out_put_pxiels_b2, in_put_pxiels_b2)
bird = Image.fromarray(out_put_img_b2)
bird.show()


weights_f2, out_put_pxiels_f2, in_put_pxiels_f2 = get_cross_trans_fb_four(frame_f, H_f, f_min_u, f_max_u, f_min_v,
                                                                          f_max_v, f_l_max_u, f_r_min_u, centerb,
                                                                          center1, center2, 0)
out_put_img_f2 = remap_four(weights_f2, out_put_pxiels_f2, in_put_pxiels_f2)
bird = Image.fromarray(out_put_img_f2)
bird.show()


weights_r2, out_put_pxiels_r2, in_put_pxiels_r2 = get_cross_trans_rl_four(frame_r, H_r, r_min_u, r_max_u, r_min_v,
                                                                          r_max_v, f_r_max_v, b_r_min_v, centerr,
                                                                          center1, center2, 3)
out_put_img_r2 = remap_four(weights_r2, out_put_pxiels_r2, in_put_pxiels_r2)
bird = Image.fromarray(out_put_img_r2)
bird.show()


weights_l2, out_put_pxiels_l2, in_put_pxiels_l2 = get_cross_trans_rl_four(frame_l, H_l, l_min_u, l_max_u, l_min_v,
                                                                          l_max_v, f_l_max_v, b_l_min_v, centerl,
                                                                          center1, center2, 1)
out_put_img_l2 = remap_four(weights_l2, out_put_pxiels_l2, in_put_pxiels_l2)
bird = Image.fromarray(out_put_img_l1)
bird.show()


weights = weights_f2 + weights_l2 + weights_b2 + weights_r2
in_put_pxiels = in_put_pxiels_f2 + in_put_pxiels_l2 + in_put_pxiels_b2 + in_put_pxiels_r2
out_put_pxiels = out_put_pxiels_f2 + out_put_pxiels_l2 + out_put_pxiels_b2 + out_put_pxiels_r2


out_put_img_2 = out_put_img_l2 + out_put_img_f2 + out_put_img_r2 + out_put_img_b2
bird = Image.fromarray(out_put_img_2)
bird.show()


import time

start = time.time()
out_put_img = remap(weights, out_put_pxiels, in_put_pxiels)
end = time.time()
print(end - start)
bird = Image.fromarray(out_put_img)
bird.show()
