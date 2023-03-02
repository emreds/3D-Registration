import cv2
import numpy as np
from os import path

idx = 2

img = cv2.imread(path.join("out", "RGB", "frame_{:04d}.png".format(idx)))
uv_map = cv2.imread(path.join("data", "2_uv_map.png"))

coords = np.loadtxt(path.join("out", "Image_Coordinates", "image_coordinates_{:04d}.txt".format(idx)),dtype=int, delimiter=',')
uv_coords = np.loadtxt(path.join("out", "UV_Coordinates", "uv_coordinates_{:04d}.txt".format(idx)),dtype=int, delimiter=',')

h,w,c = img.shape


assert len(uv_coords) == len(coords)
# for kp, uv in zip(coords, uv_coords):
#     # img = cv2.circle(img, ((h-1) -kp[0], kp[1]), radius=0, color=(0, 0, 255), thickness=-1)
#     # x = kp[0] + w
#     # y = kp[1]
#     img = cv2.circle(img, (x, y), radius=1, color=(0, 0, 255), thickness=-1)
#     uv_map = cv2.circle(uv_map, (uv[0], uv[1]), radius=1, color=(0, 0, 255), thickness=-1)

combined_img = np.hstack((cv2.copyMakeBorder(uv_map, 208, 0, 768, 0, cv2.BORDER_CONSTANT), img))
for i, (kp, uv) in enumerate(zip(coords, uv_coords)):
    
    color = list(np.random.random(size=3) * 256)
    x = kp[0] + w
    y = kp[1]
    u = uv[0] + w - 512
    v = uv[1] + h - 512
    combined_img = cv2.circle(combined_img, (x, y), radius=2, color=color, thickness=-1)
    combined_img = cv2.circle(combined_img, (u, v), radius=2, color=color, thickness=-1)
    if np.random.rand() < 0.01: 
        cv2.line(combined_img, (u,v), (x,y), color)


cv2.imshow("", combined_img)

cv2.imwrite("img.png", combined_img)
# cv2.imshow("1",img)
# cv2.imshow("2",uv_map)
cv2.waitKey()