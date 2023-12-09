import cv2 as cv
import math
import numpy as np
import os
import glob

DIR = os.path.dirname(__file__).replace("\\", "/")

RAD_TO_DEG = 180 / math.pi
DEG_TO_RAD = math.pi / 180

def WriteImg(path, img):
    cv.imwrite(path, cv.cvtColor(img, cv.COLOR_RGB2BGR))

def ShowImg(title, img):
    cv.imshow(title, cv.cvtColor(img, cv.COLOR_RGB2BGR))

    while True:
        if cv.waitKey(50) & 0xff == ord("q"):
            break

    cv.destroyWindow(title)

def GetRotMat(x, y, z, theta):
    k = 1 / np.sqrt(x**2 + y**2 + z**2)

    x *= k
    y *= k
    z *= k

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c+(1-c)*x**2 , (1-c)*x*y-s*z, (1-c)*x*z+s*y],
        [(1-c)*y*x+s*z, c+(1-c)*y**2 , (1-c)*y*z-s*x],
        [(1-c)*z*x-s*y, (1-c)*z*y+s*x, c+(1-c)*z**2]
    ])

def GetHomoTransMat(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ])

def GetHomoRotMat(x, y, z, theta):
    ret = np.identity(4)
    ret[:3, :3] = GetRotMat(x, y, z, theta)
    return ret

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

    # cv.imshow("blur_frame", frame)

    WriteImg(f"{DIR}/blur_img.png", img)

    THRESHOLD = 80

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

    color = [int(x) for x in color]

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

def NPSave(filename, data):
    np.save(filename, data)

def NPLoad(filename):
    return np.load(filename, allow_pickle=True).item()

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
    obj_phis = np.empty((OBJ_NUM,))
    obj_areas = np.empty((OBJ_NUM,))

    obj_anno_map = AnnoObjWithColorSim(img, obj_colors)

    if True:
        anno_colors = np.concatenate([[[0, 0, 0]], obj_colors],
                                    axis=0).astype(np.uint8)

        anno_to_obj_color_r = np.vectorize(
            lambda anno: anno_colors[anno][0], otypes=[np.uint8])
        anno_to_obj_color_g = np.vectorize(
            lambda anno: anno_colors[anno][1], otypes=[np.uint8])
        anno_to_obj_color_b = np.vectorize(
            lambda anno: anno_colors[anno][2], otypes=[np.uint8])

        anno_img = np.stack([
            anno_to_obj_color_r(obj_anno_map),
            anno_to_obj_color_g(obj_anno_map),
            anno_to_obj_color_b(obj_anno_map),
        ], axis=2)

        WriteImg(f"{DIR}/anno_frame.png", anno_img)

        '''
        while True:
            if cv.waitKey(50) & 0xff == ord("q"):
                break
        '''

    for obj_idx in range(OBJ_NUM):
        obj_points = np.stack(np.where(obj_anno_map == obj_idx + 1), axis=-1)

        center, phi = FindCL(obj_points) # [2], []
        area = float(obj_points.shape[0]) # [1]

        if area == 0:
            obj_founds[obj_idx] = False
            continue

        obj_founds[obj_idx] = True
        obj_centers[obj_idx] = center
        obj_phis[obj_idx] = phi
        obj_areas[obj_idx] = area

    return obj_anno_map, obj_founds, obj_centers, obj_phis, obj_areas

def ColorAnnoObjs(anno_colors, obj_anno_map):
    # anno_colors[OBJ_NUM + 1, 3] uint8
    # img[H, W]

    anno_to_obj_color_r = np.vectorize(
        lambda anno: anno_colors[anno][0], otypes=[np.uint8])
    anno_to_obj_color_g = np.vectorize(
        lambda anno: anno_colors[anno][1], otypes=[np.uint8])
    anno_to_obj_color_b = np.vectorize(
        lambda anno: anno_colors[anno][2], otypes=[np.uint8])

    obj_anno_img = np.stack([
        anno_to_obj_color_r(obj_anno_map),
        anno_to_obj_color_g(obj_anno_map),
        anno_to_obj_color_b(obj_anno_map),
    ], axis=2)

    return obj_anno_img

def DrawCenterPhi(obj_color, img, center, phi):
    DrawLine(img, center, phi, obj_color)

    circle_color = [int(255 - x) for x in obj_color]

    cv.circle(img,
              (int(round(center[1])), int(round(center[0]))),
              3, circle_color, 1)

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

    N = T_base_to_gripper.shape[0]

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

def Regress():
    camera_param = NPLoad(f"{DIR}/camera_params.npy")
    camera_mat = camera_param["camera_mat"]
    camera_distort = camera_param["camera_distort"]

    print(f"camera_mat")
    print(camera_mat)

    print(f"camera_distort")
    print(camera_distort)

    img_filenames = glob.glob(f"{DIR}/images/img-*.png")
    img_filenames.sort()

    IMG_NUM = len(img_filenames)
    H = 960
    W = 1280

    imgs = np.empty((IMG_NUM, H, W, 3), dtype=np.uint8)

    for img_idx in range(IMG_NUM):
        img = cv.cvtColor(cv.imread(img_filenames[img_idx]), cv.COLOR_BGR2RGB)
        imgs[img_idx] = img

    Ts_base_to_gripper = list()

    stride = 10

    x_start = 200
    y_start = 250

    above_z = 600

    for x in range(x_start, x_start + stride * 4, stride):
        for y in range(y_start, y_start + stride * 4, stride):
            for z in [above_z - stride, above_z]:
                x_angle = 180
                y_angle = 0
                z_angle = 135

                T = np.identity(4)
                T = GetHomoRotMat(1, 0, 0, x_angle * DEG_TO_RAD) @ T
                T = GetHomoRotMat(0, 1, 0, y_angle * DEG_TO_RAD) @ T
                T = GetHomoRotMat(0, 0, 1, z_angle * DEG_TO_RAD) @ T
                T = GetHomoTransMat(x, y, z) @ T

                Ts_base_to_gripper.append(np.linalg.inv(T))

    Ts_base_to_gripper = np.stack(Ts_base_to_gripper, axis=0)

    obj_colors = np.array([
        [247, 189, 170], # red
        [253, 254, 151], # yellow
        [118, 156, 208], # blue
    ])

    obj_area_factors = np.array([1, 2, 3])

    OBJ_NUM = obj_colors.shape[0]

    obj_locs = np.empty((IMG_NUM, OBJ_NUM, 3))
    obj_locs[:, 0, :] = [348, 158, 110] # red
    obj_locs[:, 1, :] = [171, 374, 110] # yellow
    obj_locs[:, 2, :] = [360, 319, 135] # blue

    anno_colors = np.concatenate(
        [[[0, 0, 0]], obj_colors], axis=0).astype(np.uint8)

    obj_loc_cases = list()
    obj_img_cases = list()

    for img_idx in range(IMG_NUM):
        print(f"regress img_idx {img_idx}")

        img = UndistortImage(imgs[img_idx], camera_mat, camera_distort)

        obj_anno_map, cur_obj_founds, \
        cur_obj_centers, cur_obj_phis, cur_obj_areas = FindObjs(img, obj_colors)
        # cur_obj_founds[OBJ_NUM] boolean
        # cur_obj_centers[OBJ_NUM, 2] float
        # cur_obj_phis[OBJ_NUM] float
        # cur_obj_areas[OBJ_NUM] float

        obj_anno_img = ColorAnnoObjs(anno_colors, obj_anno_map)

        for obj_idx in range(OBJ_NUM):
            DrawCenterPhi(obj_colors[obj_idx], obj_anno_img,
                          cur_obj_centers[obj_idx], cur_obj_phis[obj_idx])

        # ShowImg(f"obj_anno_img {img_idx}", obj_anno_img)
        WriteImg(f"{DIR}/obj_anno_img/obj_anno_img_{img_idx}.png", obj_anno_img)

        for obj_idx in range(OBJ_NUM):
            obj_loc = obj_locs[img_idx, obj_idx]

            obj_loc = np.array([
                [obj_loc[0]],
                [obj_loc[1]],
                [obj_loc[2]],
                [         1],
            ])

            obj_loc_cases.append(Ts_base_to_gripper[img_idx] @ obj_loc)

            k = (cur_obj_areas[obj_idx] / obj_area_factors[obj_idx])**-0.5

            obj_img_cases.append(
                np.array([[cur_obj_centers[obj_idx, 1] * k],
                          [cur_obj_centers[obj_idx, 0] * k],
                          [                              k],
                          [                              1],]))

    # obj_loc_cases[N][4, 1]
    # obj_img_cases[N][4, 1]

    obj_loc_cases = np.stack(obj_loc_cases, axis=-1) # [4, 1, N]
    obj_loc_cases = obj_loc_cases.reshape((4, -1))
    # [4, N]

    obj_img_cases = np.stack(obj_img_cases, axis=-1) # [4, 1, N]
    obj_img_cases = obj_img_cases.reshape((4, -1))
    # [4, N]

    M_camera_to_gripper = SolveM(obj_img_cases, obj_loc_cases)

    print(f"M_camera_to_gripper\n{M_camera_to_gripper}")

    NPSave(f"{DIR}/M_camera_to_gripper.npy",
           {"M_camera_to_gripper": M_camera_to_gripper})

def Inference(img, cur_pose):
    camera_param = NPLoad(f"{DIR}/camera_params.npy")
    camera_mat = camera_param["camera_mat"]
    camera_distort = camera_param["camera_distort"]

    obj_colors = np.array([
        [247, 189, 170], # red
        [253, 254, 151], # yellow
        [118, 156, 208], # blue
    ])

    obj_area_factors = np.array([1, 2, 3])

    OBJ_NUM = obj_colors.shape[0]

    M_camera_to_gripper = NPLoad(f"{DIR}/M_camera_to_gripper.npy") \
        ["M_camera_to_gripper"] # [4, 4]

    T = np.identity(4)
    T = GetHomoRotMat(1, 0, 0, cur_pose[3] * DEG_TO_RAD) @ T
    T = GetHomoRotMat(0, 1, 0, cur_pose[4] * DEG_TO_RAD) @ T
    T = GetHomoRotMat(0, 0, 1, cur_pose[5] * DEG_TO_RAD) @ T
    T = GetHomoTransMat(cur_pose[0], cur_pose[1], cur_pose[2]) @ T

    T_base_to_gripper = np.linalg.inv(T)

    img = UndistortImage(img, camera_mat, camera_distort)

    obj_anno_map, obj_founds, \
    obj_centers, obj_phis, obj_areas = FindObjs(img, obj_colors)
    # obj_founds[OBJ_NUM] boolean
    # obj_centers[OBJ_NUM, 2] float
    # obj_phis[OBJ_NUM] float
    # obj_areas[OBJ_NUM] float

    ks = (obj_areas / obj_area_factors)**-0.5

    obj_camera_locs = np.empty((OBJ_NUM, 4, 1))
    obj_camera_locs[:, 0, 0] = obj_centers[:, 1] * ks
    obj_camera_locs[:, 1, 0] = obj_centers[:, 0] * ks
    obj_camera_locs[:, 2, 0] = ks
    obj_camera_locs[:, 3, 0] = 1

    tmp = M_camera_to_gripper @ obj_camera_locs # [OBJ_NUM, 4, 1]
    tmp = np.linalg.inv(T_base_to_gripper) @ tmp # [OBJ_NUM, 4, 1]

    obj_locs = tmp[:, :3, 0] # [OBJ_NUM, 3]

    obj_poses = list()

    for obj_idx in range(OBJ_NUM):
        obj_ang = obj_phis[obj_idx] * RAD_TO_DEG

        if obj_founds[obj_idx]:
            while obj_ang < 0:
                obj_ang += 360

            while obj_ang < -90:
                obj_ang += 180

            while 90 < obj_ang:
                obj_ang -= 180

        obj_ang += 135

        obj_pose = [
            float(obj_locs[obj_idx, 0]),
            float(obj_locs[obj_idx, 1]),
            float(obj_locs[obj_idx, 2]),
            float(180),
            float(0),
            float(obj_ang),
        ]

        obj_poses.append(obj_pose)

    return obj_poses

def Test():
    camera_param = NPLoad(f"{DIR}/camera_params.npy")
    camera_mat = camera_param["camera_mat"]
    camera_distort = camera_param["camera_distort"]

    print(f"camera_mat")
    print(camera_mat)

    print(f"camera_distort")
    print(camera_distort)

    img_filenames = glob.glob(f"{DIR}/images/img-*.png")
    img_filenames.sort()

    IMG_NUM = len(img_filenames)
    H = 960
    W = 1280

    imgs = np.empty((IMG_NUM, H, W, 3), dtype=np.uint8)

    for img_idx in range(IMG_NUM):
        img = cv.cvtColor(cv.imread(img_filenames[img_idx]), cv.COLOR_BGR2RGB)
        imgs[img_idx] = img

    stride = 10

    x_start = 200
    y_start = 250

    above_z = 600

    cur_poses = list()

    for x in range(x_start, x_start + stride * 4, stride):
        for y in range(y_start, y_start + stride * 4, stride):
            for z in [above_z - stride, above_z]:
                x_angle = 180
                y_angle = 0
                z_angle = 135

                cur_poses.append([x, y, z, x_angle, y_angle, z_angle])

    obj_colors = np.array([
        [247, 189, 170], # red
        [253, 254, 151], # yellow
        [118, 156, 208], # blue
    ])

    OBJ_NUM = obj_colors.shape[0]

    obj_locs = np.empty((IMG_NUM, OBJ_NUM, 3))
    obj_locs[:, 0, :] = [348, 158, 110] # red
    obj_locs[:, 1, :] = [171, 374, 110] # yellow
    obj_locs[:, 2, :] = [360, 319, 135] # blue

    loc_errs = list()

    for img_idx in range(IMG_NUM):
        obj_poses = Inference(imgs[img_idx], cur_poses[img_idx])

        for obj_idx in range(OBJ_NUM):
            obj_pose = obj_poses[obj_idx]

            obj_pose = np.array(obj_pose)

            err = obj_pose[:3] - obj_locs[img_idx, obj_idx]
            err = err[:2]

            loc_err = (err**2).sum()

            loc_errs.append(loc_err)

            print(f"loc_err {img_idx} {obj_idx} = {loc_err}")

    for loc_err in loc_errs:
        print(f"{loc_err}")

if __name__ == "__main__":
    Regress()
    Test()
