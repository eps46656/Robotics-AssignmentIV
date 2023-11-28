
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

def UndistortImage(img, camera_mat, camera_distort):
    H, W = img.shape[:2]

    new_camera_mat, roi = cv.getOptimalNewCameraMatrix(
        camera_mat, camera_distort, (W, H), 1, (W, H))

    undistorted_img = cv.undistort(
        img, camera_mat, camera_distort, None, new_camera_mat)

    x, y, w, h = roi

    undistorted_img = undistorted_img[y:y+h, x:x+w]

    return undistorted_img

def GetNormalizedMat(points, center, dist):
    # points[P, N]
    # center[P-1]
    # dist

    center = np.array(center)

    assert len(points.shape) == 2
    assert points.shape[0] - 1 == center.shape[0]

    P = points.shape[0]

    origin = points.mean(1)[:-1]

    odist = (((points[:-1, :] - origin.reshape((P-1, 1)))**2).sum(0)**0.5).mean()

    k = dist / odist

    ret = np.zeros([P, P])

    for i in range(P-1):
        ret[i, i] = k

    ret[-1, -1] = 1

    ret[:-1, -1] = center - origin * k

    return ret

def FindHomographic(src, dst):
    # src[P, N]
    # dst[Q, N]

    assert len(src.shape) == 2
    assert len(dst.shape) == 2
    assert src.shape[1] == dst.shape[1]

    P, N = src.shape
    Q, _ = dst.shape

    A = np.zeros([N*(Q-1), P*Q])

    for i in range(N):
        for j in range(Q-1):
            A[(Q-1)*i+j, j*P:j*P+P] = src[:, i]
            A[(Q-1)*i+j, -P:] = src[:, i] * -dst[j, i]

    _, _, Vh = np.linalg.svd(A)

    return Vh[-1, :].reshape((Q, P))

def NormalizedDLT(points1, points2, normalized):
    # points1[P, N]
    # points2[Q, N]

    assert len(points1.shape) == 2
    assert len(points2.shape) == 2

    P = points1.shape[0]
    Q = points2.shape[0]

    T1 = np.identity(P)
    T2 = np.identity(Q)

    if normalized:
        T1 = GetNormalizedMat(points1, np.zeros([P-1]), np.sqrt(P-1))
        T2 = GetNormalizedMat(points2, np.zeros([Q-1]), np.sqrt(Q-1))

    rep_points1 = T1 @ points1
    rep_points2 = T2 @ points2

    H = FindHomographic(rep_points1, rep_points2) # [Q, P]

    return np.linalg.inv(T2) @ H @ T1

def GetObjHomography(obj_locs, obj_centers, obj_areas):
    N = obj_locs.shape[0]

    assert obj_locs.shape == (N, 3)
    assert obj_centers.shape == (N, 2)
    assert obj_areas.shape == (N,)

    xs = np.empty((4, N))
    ys = np.empty((4, N))

    xs[0, :] = obj_centers[:, 0]
    xs[1, :] = obj_centers[:, 1]
    xs[2, :] = obj_areas**-0.5
    xs[3, :] = 1

    ys[0, :] = obj_locs[:, 0]
    ys[1, :] = obj_locs[:, 1]
    ys[2, :] = obj_locs[:, 2]
    ys[3, :] = 1

    H = NormalizedDLT(xs, ys, True) # [4, 4]

    re_ys = H @ xs # [4, N]
    re_ys /= re_ys[3, :]

    err = (re_ys - ys)**2 # [4, N]
    err = err.sum(axis=0) # [N]
    err = err**0.5 # [N]
    err = err.sum() # []

    return H, err

'''

estimate the translation between current loc to obj loc

'''

def main():
    camera_mat, camera_distort = \
        ReadCameraParam(f"{DIR}/../camera_calib/camera_params.npy")

    obj_colors = np.array([
        [ 97,113,145], # blue
        [254,224, 38], # yellow
        [181,210,124], # green
    ])

    IMG_NUM = 12
    H = 1280 # TODO: this parameter should be modified
    W = 960 # TODO: this parameter should be modified

    OBJ_NUM = obj_colors.shape[0]

    img_locs = np.empty((IMG_NUM, 3))
    # TODO: locations denote where these image are captured

    imgs = np.empty((IMG_NUM, H, W, 3))
    # TODO: images in rgb

    obj_locs = np.empty((IMG_NUM, OBJ_NUM, 3))
    # TODO: obj position for each images, each objects

    imgs = np.stack([
        UndistortImage(img, camera_mat, camera_distort) for img in imgs
    ], axis=0)

    obj_centers = np.empty((IMG_NUM, OBJ_NUM, 2))
    obj_phis = np.empty((IMG_NUM, OBJ_NUM, 2))
    obj_areas = np.empty((IMG_NUM, OBJ_NUM))

    for img_idx in range(IMG_NUM):
        img = imgs[img_idx]

        obj_anno = AnnoObjWithColorSim(img, obj_colors)

        for obj_idx in range(OBJ_NUM):
            obj_points = np.stack(np.where(obj_anno == obj_idx + 1), axis=-1)

            center, phi = FindCL(obj_points)
            area = float(obj_points.shape[0])

            # center[2]
            # phi[]

            obj_centers[img_idx, obj_idx] = center
            obj_phis[img_idx, obj_idx] = phi
            obj_areas[img_idx, obj_idx] = area

    # x = [obj_center_x,
    #      obj_center_y,
    #      obj_area**-0.5,
    #      1]^T

    # y = [obj_pos_x - img_loc_x,
    #      obj_pos_y - img_loc_y,
    #      obj_pos_z - img_loc_z,
    #      1]^T

    for obj_idx in range(OBJ_NUM):
        print(f"obj_idx = {obj_idx}")

        H, err = GetObjHomography(
            obj_locs[:, obj_idx, :] - img_locs, # [N, 3]
            obj_centers[:, obj_idx, :], # [N, 2]
            obj_areas[:, obj_idx], # [N]
        )

        print(f"H =\n{H}")
        print(f"err = {err}")

def main2():
    N = 5

    H, err = GetObjHomography(
        np.random.rand(N, 3), # locs[N, 3]
        np.random.rand(N, 2), # centers[N, 2]
        np.random.rand(N), # areas[N]
    )

    print(f"H =\n{H}")
    print(f"err = {err}")

if __name__ == "__main__":
    main2()
