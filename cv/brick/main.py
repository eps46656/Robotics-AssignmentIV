
import cv2 as cv
import math
import numpy as np
import os

DIR = os.path.dirname(__file__).replace("\\", "/")

def AnnoObjWithColorSim(img, obj_colors):
    # img[H, W, 3]
    # obj_colors[N, 3]

    assert len(img.shape) == 3
    assert img.shape[2] == 3

    assert len(obj_colors.shape) == 2
    assert obj_colors.shape[1] == 3

    H, W = img.shape[:2]
    N = obj_colors.shape[0]

    img = cv.medianBlur(img, 11)

    THRESHOLD = 60

    dists = [np.full((H, W), THRESHOLD)]

    for obj_color in obj_colors:
        dists.append(np.abs(img - obj_color).max(axis=-1))

    mark_img = np.stack(dists, axis=-1).argmin(axis=-1)
    mark_img, trans = SepImage(mark_img)

    assigned_obj_idx = set()

    for mark, count in sorted(zip(*np.unique(mark_img, return_counts=True)),
                              key=lambda p: p[1],
                              reverse=True):
        obj_idx = trans[mark]

        if obj_idx in assigned_obj_idx:
            trans[mark] = 0
        else:
            assigned_obj_idx.add(obj_idx)

    return np.vectorize(trans.get, otypes=[np.int32])(mark_img)

def SepImage(img):
    # img[H, W]

    assert len(img.shape) == 2

    H, W = img.shape

    m = np.zeros((H, W), dtype=np.int32)
    obj_i = 0

    q = list()

    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]

    trans = {0: 0}

    for i in range(H):
        for j in range(W):
            if img[i, j] == 0 or 0 < m[i, j]:
                continue

            obj_i += 1
            m[i, j] = obj_i
            q.append((i, j))

            cur_val = img[i, j]

            trans[obj_i] = cur_val

            while 0 < len(q):
                cur_i, cur_j = q[-1]
                q.pop()

                for d in range(4):
                    nxt_i, nxt_j = cur_i + dx[d], cur_j + dy[d]

                    if 0 <= nxt_i and nxt_i < H and \
                    0 <= nxt_j and nxt_j < W and \
                    m[nxt_i, nxt_j] == 0 and \
                    cur_val == img[nxt_i, nxt_j]:
                        m[nxt_i, nxt_j] = obj_i
                        q.append((nxt_i, nxt_j))

    return m, trans

def FindCL(points):
    # points[N, 2]

    N = points.shape[0]

    assert points.shape == (N, 2)

    center = points.mean(axis=0)
    norm_points = points - center

    u11 = (norm_points[:, 0] * norm_points[:, 1]).sum()
    u20 = (norm_points[:, 0]**2).sum()
    u02 = (norm_points[:, 1]**2).sum()

    phi = 0.5 * math.atan2(2 * u11, u20 - u02)

    return center, phi

def DrawLine(img, center, phi, color):
    assert len(img.shape) == 3
    assert img.shape[2] == 3

    c = math.cos(phi)
    s = math.sin(phi)

    H, W = img.shape[:2]

    def ok(p):
        return 0 <= p[0] and p[0] < H and 0 <= p[1] and p[1] < W

    ps = [[int((  0-center[0]) / c * s + center[1]),   0],
          [int((H-1-center[0]) / c * s + center[1]), H-1],
          [0  , int((  0-center[1]) / s * c + center[0])],
          [W-1, int((W-1-center[1]) / s * c + center[0])],]

    k = []

    for p in ps:
        if 0 <= p[0] and p[0] < W and 0 <= p[1] and p[1] < H:
            k.append(p)

    cv.line(img, k[0], k[1], color, 1)

def ReadCameraParam(filename):
    camera_params = np.load(filename, allow_pickle=True).item()
    camera_mat = camera_params["camera_mat"]
    camera_distort = camera_params["camera_distort"]

    return camera_mat, camera_distort

def GetCos(u, v):
    return (u*v).sum() / np.linalg.norm(u) / np.linalg.norm(v)

def GetAng(u, v):
    return np.arccos(GetCos(u, v))

def main():
    img = cv.cvtColor(cv.imread(f"{DIR}/images/img-730.png"), cv.COLOR_BGR2RGB)

    print(img.shape)

    obj_colors = np.array([
        [254,224, 38], # yellow
        [ 97,113,145], # blue
        [181,210,124], # green
    ])

    obj_anno = AnnoObjWithColorSim(img, obj_colors)

    H, W = img.shape[:2]

    anno_colors = np.array([
        [0, 0, 0],
        obj_colors[0, :],
        obj_colors[1, :],
        obj_colors[2, :],
    ], dtype=np.uint8)

    img_ = anno_colors[obj_anno]

    for obj_i, obj_color in enumerate(obj_colors):
        obj_points = np.stack(np.where(obj_anno == obj_i + 1), axis=-1)

        center, phi = FindCL(obj_points)

        print(f"obj {obj_i}: center = {center} phi = {phi}")

        DrawLine(img_, center, phi, [255, 255, 255])

        cv.circle(img_,
                  (int(round(center[1])), int(round(center[0]))),
                   3, [127, 127, 127], -1)

    cv.imshow("img", cv.cvtColor(img_, cv.COLOR_RGB2BGR))
    cv.waitKey(0)

def main2():
    img = cv.cvtColor(cv.imread(f"{DIR}/images/img-430.png"), cv.COLOR_BGR2RGB)

    obj_colors = np.array([
        [254,224, 38], # yellow
        [ 97,113,145], # blue
        [181,210,124], # green
    ])

    N = len(obj_colors)

    obj_anno = AnnoObjWithColorSim(img, obj_colors)

    H, W = img.shape[:2]

    camera_mat, camera_distort = \
        ReadCameraParam(f"{DIR}/../camera_calib/camera_params.npy")

    inv_camera_mat = np.linalg.inv(camera_mat)

    obj_areas = np.empty((N,))
    obj_centers = np.empty((N, 2))
    obj_phis = np.empty((N,))

    for obj_idx in range(N):
        obj_points = np.stack(np.where(obj_anno == obj_idx + 1), axis=-1)

        obj_area = float(obj_points.shape[0])
        obj_center, obj_phi = FindCL(obj_points)

        obj_areas[obj_idx] = obj_area
        obj_centers[obj_idx] = obj_center
        obj_phis[obj_idx] = obj_phi

    obj_area_coeffs = list()

    for obj_idx in range(N):
        obj_center = obj_centers[obj_idx]
        obj_center = np.array([[obj_center[0]], [obj_center[1]], [1]])

        obj_area_coeffs.append(
            GetCos(inv_camera_mat @ obj_center,
                   np.array([[0], [0], [1]])))

    # print(obj_area_coeffs)
    print(obj_areas**0.5)

def UndistortImage(img, camera_mat, camera_distort):
    H, W = img.shape[:2]

    new_camera_mat, roi = cv.getOptimalNewCameraMatrix(
        camera_mat, camera_distort, (W, H), 1, (W, H))

    undistorted_img = cv.undistort(
        img, camera_mat, camera_distort, None, new_camera_mat)

    x, y, w, h = roi

    undistorted_img = undistorted_img[y:y+h, x:x+w]

    return undistorted_img

def main3():
    '''
    model:
        proj_area = K / distance^2

    F:
        z: the z param of robot arm

        proj_area = K / (C + M * z)^2

        merge K

        C + M * z = sqrt(1 / proj_area)

        denote xi = sqrt(1 / proj_area_i)
        denote yi = z

        modify the model for robustness

        yi = c0 + c1 * xi + c2 * xi^2 + c3 * xi^3

        linear regression to get c

    when height estimating
        est_z = (sqrt(real_area / proj_area) - C) / M

        cur_z: the current z param of robot arm

        height = cur_z - est_z
    '''

    camera_mat, camera_distort = \
        ReadCameraParam(f"{DIR}/../camera_calib/camera_params.npy")

    inv_camera_mat = np.linalg.inv(camera_mat)

    img_filenames = [
        f"{DIR}/images/img-430.png",
        f"{DIR}/images/img-530.png",
        f"{DIR}/images/img-630.png",
        f"{DIR}/images/img-730.png",
    ]

    IMG_NUM = len(img_filenames)

    zs = np.array([430, 530, 630, 730])

    obj_colors = np.array([
        [ 97,113,145], # blue
        [254,224, 38], # yellow
        [181,210,124], # green
    ])

    anno_colors = np.concatenate([[[0, 0, 0]], obj_colors],
                                 axis=0).astype(np.uint8)

    OBJ_NUM = obj_colors.shape[0]

    anno_to_obj_color_r = np.vectorize(
        lambda anno: anno_colors[anno][0], otypes=[np.uint8])
    anno_to_obj_color_g = np.vectorize(
        lambda anno: anno_colors[anno][1], otypes=[np.uint8])
    anno_to_obj_color_b = np.vectorize(
        lambda anno: anno_colors[anno][2], otypes=[np.uint8])

    obj_areas = np.empty((IMG_NUM, OBJ_NUM))

    for img_idx in range(IMG_NUM):
        print(f"img_idx = {img_idx}")

        img_filename = img_filenames[img_idx]

        print(f"img_filename = {img_filename}")

        img = cv.cvtColor(cv.imread(img_filenames[img_idx]),
                          cv.COLOR_BGR2RGB)

        img = UndistortImage(img, camera_mat, camera_distort)

        assert len(img.shape) == 3
        assert img.shape[2] == 3

        obj_anno = AnnoObjWithColorSim(img, obj_colors)

        anno_img = np.stack([
            anno_to_obj_color_r(obj_anno),
            anno_to_obj_color_g(obj_anno),
            anno_to_obj_color_b(obj_anno),
        ], axis=2)

        for obj_idx in range(OBJ_NUM):
            obj_points = np.stack(np.where(obj_anno == obj_idx + 1), axis=-1)

            area = float(obj_points.shape[0])

            center, phi = FindCL(obj_points)

            obj_areas[img_idx, obj_idx] = area

            print(f"obj_idx {obj_idx}: center = {center} phi = {phi} area = {area}")

            DrawLine(anno_img, center, phi, [255, 255, 255])

            cv.circle(anno_img,
                      (int(round(center[1])), int(round(center[0]))),
                      3, [255, 0, 0], -1)

        '''
        cv.imshow(f"anno {img_filename}",
                  cv.cvtColor(anno_img, cv.COLOR_RGB2BGR))
        cv.waitKey(0)
        '''

    for obj_idx in range(OBJ_NUM):
        print(f"obj_idx = {obj_idx}")

        xs = obj_areas[:, obj_idx]**-0.5
        ys = zs.reshape((IMG_NUM, 1))

        K = len(xs)

        A = list()

        A.append(np.ones((K,)))
        A.append(xs)

        A = np.stack(A, axis=-1)

        cs, res = np.linalg.lstsq(A, ys)[:2]

        res = ((A @ cs - ys)**2).sum()

        print(f"cs =\n{cs}")
        print(f"res = {res}")
        print(f"rms = {(res / K)**0.5}")

if __name__ == "__main__":
    main3()
