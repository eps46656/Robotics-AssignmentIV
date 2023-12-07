
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

def FindObjs(img, obj_colors):
    # obj_colors[OBJ_NUM, 3]

    assert len(obj_colors.shape) == 2

    OBJ_NUM = obj_colors.shape[0]

    obj_founds = np.empty((OBJ_NUM,), dtype=np.bool_)
    obj_centers = np.empty((OBJ_NUM, 2))
    obj_phis = np.empty((OBJ_NUM, 2))
    obj_areas = np.empty((OBJ_NUM,))

    obj_anno = AnnoObjWithColorSim(img, obj_colors)

    for obj_idx in range(OBJ_NUM):
        obj_points = np.stack(np.where(obj_anno == obj_idx + 1), axis=-1)

        center, phi = FindCL(obj_points) # [2], []
        area = float(obj_points.shape[0]) # [1]

        if area == 0:
            obj_founds[obj_idx] = False
            continue

        obj_founds[obj_idx] = True
        obj_centers[obj_idx] = center
        obj_phis[obj_idx] = phi
        obj_areas[obj_idx] = area

    return obj_founds, obj_centers, obj_phis, obj_areas

def GetNormalizeMat(points, center, radius):
    # points[N, P]
    # center[P]
    # radius[]

    center = np.array(center)
    radius = np.array(radius)

    assert len(points.shape) == 2
    assert center.shape == (points.shape[1],)
    assert len(radius.shape) == 0

    origin_center = points.mean(axis=0) # [P]

    tmp = points - origin_center # [N, P]
    tmp = tmp**2 # [N, P]
    tmp = tmp.sum(axis=1) # [N]
    tmp = tmp**0.5 # [N]
    origin_radius = tmp.mean() # []

    ratio = radius / origin_radius

    return \
        np.array([
            [1, 0, 0, origin_center[0]],
            [0, 1, 0, origin_center[1]],
            [0, 0, 1, origin_center[2]],
            [0, 0, 0,                1],]) @ \
        np.array([
            [ratio,     0,     0, 0],
            [    0, ratio,     0, 0],
            [    0,     0, ratio, 0],
            [    0,     0,     0, 1],]) @ \
        np.array([
            [1, 0, 0, -center[0]],
            [0, 1, 0, -center[1]],
            [0, 0, 1, -center[2]],
            [0, 0, 0,          1],])

def SolveM(points1, points2):
    # points1[P, N]
    # points2[Q, N]
    # M @ points1 = points2

    assert len(points1.shape) == 2
    assert len(points2.shape) == 2

    P, N = points1.shape
    Q = points2.shape[0]

    assert points2.shape == (Q, N)

    points1 = points1.transpose()

    A = np.empty((Q*N, Q*P))

    for i in range(Q):
        for j in range(Q):
            A[N*i:N*(i+1), P*j:P*(j+1)] = points1 if i == j else 0

    M, res = np.linalg.lstsq(A, points2.reshape((Q*N, 1)), rcond=None)[:2]

    M = M.reshape((Q, P))

    return M

def GetObjM(T_base_to_gripper, obj_base_locs, obj_centers, obj_areas):
    # T_base_to_gripper[N, 4, 4]
    # obj_base_locs[N, 3]
    # obj_centers[N, 2]
    # obj_areas[N]

    N = obj_base_locs.shape[0]

    assert T_base_to_gripper.shape == (N, 4, 4)
    assert obj_base_locs.shape == (N, 3)
    assert obj_centers.shape == (N, 2)
    assert obj_areas.shape == (N,)

    tmp = np.empty((N, 4, 1))
    tmp[:, :3, 0] = obj_base_locs
    tmp[:, 3, 0] = 1

    obj_gripper_locs = T_base_to_gripper @ tmp # [N, 4, 1]
    obj_gripper_locs = obj_gripper_locs.reshape((N, 4)).transpose() # [4, N]

    rsqrt_obj_areas = obj_areas**-0.5 # [N]

    obj_camera_locs = np.empty((4,N))
    obj_camera_locs[0, :] = obj_centers[:, 1] * rsqrt_obj_areas
    obj_camera_locs[1, :] = obj_centers[:, 0] * rsqrt_obj_areas
    obj_camera_locs[2, :] = rsqrt_obj_areas
    obj_camera_locs[3, :] = 1

    M = SolveM(obj_camera_locs, obj_gripper_locs) # [4, 4]

    return M

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

    Ts_base_to_gripper = np.empty((IMG_NUM, 4, 4))
    # TODO: transformations from base to gripper for each image

    imgs = np.empty((IMG_NUM, H, W, 3))
    # TODO: images in rgb

    obj_locs = np.empty((IMG_NUM, OBJ_NUM, 3))
    # TODO: obj position for each images, each objects

    obj_centers = np.empty((IMG_NUM, OBJ_NUM, 2))
    obj_phis = np.empty((IMG_NUM, OBJ_NUM, 2))
    obj_areas = np.empty((IMG_NUM, OBJ_NUM))

    for img_idx in range(IMG_NUM):
        img = UndistortImage(imgs[img_idx], camera_mat, camera_distort)

        obj_anno = AnnoObjWithColorSim(img, obj_colors)

        for obj_idx in range(OBJ_NUM):
            obj_points = np.stack(np.where(obj_anno == obj_idx + 1), axis=-1)

            center, phi = FindCL(obj_points) # [2], []
            area = float(obj_points.shape[0]) # [1]

            obj_centers[img_idx, obj_idx] = center
            obj_phis[img_idx, obj_idx] = phi
            obj_areas[img_idx, obj_idx] = area

    obj_Ms = np.empty((OBJ_NUM, 4, 4))

    for obj_idx in range(OBJ_NUM):
        print(f"obj_idx = {obj_idx}")

        obj_M = GetObjM(
            Ts_base_to_gripper, # [IMG_NUM, 4, 4]
            obj_locs[:, obj_idx, :], # [IMG_NUM, 3]
            obj_centers[:, obj_idx, :], # [IMG_NUM, 2]
            obj_areas[:, obj_idx], # [IMG_NUM]
        )

        assert obj_M.shape == (4, 4)

        obj_Ms[obj_idx] = obj_M

        print(f"obj_M =\n{obj_M}")

def Regress():
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

    Ts_base_to_gripper = np.empty((IMG_NUM, 4, 4))
    # TODO: transformations from base to gripper for each image

    imgs = np.empty((IMG_NUM, H, W, 3))
    # TODO: images in rgb

    obj_locs = np.empty((IMG_NUM, OBJ_NUM, 3))
    # TODO: obj position for each images, each objects

    obj_centers = np.empty((IMG_NUM, OBJ_NUM, 2))
    obj_phis = np.empty((IMG_NUM, OBJ_NUM, 2))
    obj_areas = np.empty((IMG_NUM, OBJ_NUM))

    for img_idx in range(IMG_NUM):
        cur_obj_founds, cur_obj_centers, cur_obj_phis, cur_obj_areas = \
            FindObjs(imgs[img_idx], obj_colors)
        # cur_obj_founds[OBJ_NUM] boolean
        # cur_obj_centers[OBJ_NUM, 2] float
        # cur_obj_phis[OBJ_NUM] float
        # cur_obj_areas[OBJ_NUM] float

        cur_obj_centers = cv.undistortPoints(
            cur_obj_centers, camera_mat, camera_distort, None, camera_mat) \
            .reshape((OBJ_NUM, 2))

        obj_centers[img_idx] = cur_obj_centers
        obj_phis[img_idx] = cur_obj_phis
        obj_areas[img_idx] = cur_obj_areas

    obj_Ms = np.empty((OBJ_NUM, 4, 4))

    for obj_idx in range(OBJ_NUM):
        print(f"obj_idx = {obj_idx}")

        obj_M = GetObjM(
            Ts_base_to_gripper, # [IMG_NUM, 4, 4]
            obj_locs[:, obj_idx, :], # [IMG_NUM, 3]
            obj_centers[:, obj_idx, :], # [IMG_NUM, 2]
            obj_areas[:, obj_idx], # [IMG_NUM]
        )

        assert obj_M.shape == (4, 4)

        obj_Ms[obj_idx] = obj_M

        print(f"obj_M =\n{obj_M}")

def Inference(img):
    OBJ_NUM = 3

    camera_mat, camera_distort = \
        ReadCameraParam(f"{DIR}/../camera_calib/camera_params.npy")

    obj_colors = np.array([
        [ 97,113,145], # blue
        [254,224, 38], # yellow
        [181,210,124], # green
    ])

    obj_Ms = np.empty((OBJ_NUM, 4, 4))

    T_base_to_gripper = np.empty((4, 4))

    obj_founds, obj_centers, obj_phis, obj_areas = FindObjs(img, obj_colors)
    # obj_founds[OBJ_NUM] boolean
    # obj_centers[OBJ_NUM, 2] float
    # obj_phis[OBJ_NUM] float
    # obj_areas[OBJ_NUM] float

    obj_centers = cv.undistortPoints(
        obj_centers, camera_mat, camera_distort, None, camera_mat) \
        .reshape((OBJ_NUM, 2))

    rsqrt_obj_areas = obj_areas**-0.5

    obj_camera_locs = np.empty((OBJ_NUM, 4, 1))
    obj_camera_locs[:, 0, 0] = obj_centers[:, 1] * rsqrt_obj_areas
    obj_camera_locs[:, 1, 0] = obj_centers[:, 0] * rsqrt_obj_areas
    obj_camera_locs[:, 2, 0] = rsqrt_obj_areas
    obj_camera_locs[:, 3, 0] = 1

    tmp = obj_Ms @ obj_camera_locs # [OBJ_NUM, 4, 1]
    tmp = np.linalg.inv(T_base_to_gripper) @ tmp # [OBJ_NUM, 4, 1]

    obj_locs = tmp[:, :3, 0] # [OBJ_NUM, 3]

    print("obj_locs =\n")
    print(obj_locs)

def TestSolveM():
    P = 7
    Q = 5
    N = 128
    sigma = 2

    M = np.random.rand(Q, P) * 64

    points1 = np.random.rand(P, N) * 64
    points2 = M @ points1 + sigma * np.random.rand(Q, N)

    pr_M = SolveM(points1, points2)

    assert pr_M.shape == M.shape

    pr_points2 = pr_M @ points1

    err = (pr_points2 - points2)**2 # [Q, N]
    err = err.sum(axis=0) # [N]
    err = err**0.5 # [N]
    err = err.mean() # []

    print(f"err = {err}")

if __name__ == "__main__":
    TestSolveM()
