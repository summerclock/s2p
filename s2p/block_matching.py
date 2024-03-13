# Copyright (C) 2015, Carlo de Franchis <carlo.de-franchis@ens-cachan.fr>
# Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
# Copyright (C) 2015, Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>
# Copyright (C) 2015, Julien Michel <julien.michel@cnes.fr>

import os
import numpy as np
import rasterio
import math

from s2p import common
from s2p.config import cfg


class MaxDisparityRangeError(Exception):
    pass


def create_rejection_mask(disp, im1, im2, mask):
    """
    Create rejection mask (0 means rejected, 1 means accepted)
    Keep only the points that are matched and present in both input images

    Args:
        disp: path to the input disparity map
        im1, im2: rectified stereo pair
        mask: path to the output rejection mask
    """
    tmp1 = common.tmpfile('.tif')
    tmp2 = common.tmpfile('.tif')
    common.run(["plambda", disp, "x 0 join", "-o", tmp1])
    common.run(["backflow", tmp1, im2, tmp2])
    common.run(["plambda", disp, im1, tmp2, "x isfinite y isfinite z isfinite and and vmul", "-o", mask])


def compute_disparity_map(im1, im2, disp, mask, algo, disp_min=None,
                          disp_max=None, timeout=600, max_disp_range=None,
                          extra_params=''):
    """
    Runs a block-matching binary on a pair of stereo-rectified images.

    Args:
        im1, im2: rectified stereo pair
        disp: path to the output diparity map
        mask: path to the output rejection mask
        algo: string used to indicate the desired binary. Currently it can be
            one among 'hirschmuller02', 'hirschmuller08',
            'hirschmuller08_laplacian', 'hirschmuller08_cauchy', 'sgbm',
            'msmw', 'tvl1', 'mgm', 'mgm_multi' and 'micmac'
        disp_min: smallest disparity to consider
        disp_max: biggest disparity to consider
        timeout: time in seconds after which the disparity command will
            raise an error if it hasn't returned.
            Only applies to `mgm*` algorithms.
        extra_params: optional string with algorithm-dependent parameters

    Raises:
        MaxDisparityRangeError: if max_disp_range is defined,
            and if the [disp_min, disp_max] range is greater
            than max_disp_range, to avoid endless computation.
    """
    # limit disparity bounds
    if disp_min is not None and disp_max is not None:
        with rasterio.open(im1, "r") as f:
            width = f.width
        if disp_max - disp_min > width:
            center = 0.5 * (disp_min + disp_max)
            disp_min = int(center - 0.5 * width)
            disp_max = int(center + 0.5 * width)

    # round disparity bounds
    if disp_min is not None:
        disp_min = int(np.floor(disp_min))
    if disp_max is not None:
        disp_max = int(np.ceil(disp_max))

    if (
        max_disp_range is not None
        and disp_max - disp_min > max_disp_range
    ):
        raise MaxDisparityRangeError(
            'Disparity range [{}, {}] greater than {}'.format(
                disp_min, disp_max, max_disp_range
            )
        )

    # define environment variables
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(cfg['omp_num_threads'])

    # call the block_matching binary
    if algo == 'hirschmuller02':
        bm_binary = 'subpix.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
        # extra_params: LoG(0) regionRadius(3)
        #    LoG: Laplacian of Gaussian preprocess 1:enabled 0:disabled
        #    regionRadius: radius of the window

    if algo == 'hirschmuller08':
        bm_binary = 'callSGBM.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
        # extra_params: regionRadius(3) P1(default) P2(default) LRdiff(1)
        #    regionRadius: radius of the window
        #    P1, P2 : regularization parameters
        #    LRdiff: maximum difference between left and right disparity maps

    if algo == 'hirschmuller08_laplacian':
        bm_binary = 'callSGBM_lap.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
    if algo == 'hirschmuller08_cauchy':
        bm_binary = 'callSGBM_cauchy.sh'
        common.run('{0} {1} {2} {3} {4} {5} {6} {7}'.format(bm_binary, im1, im2, disp, mask, disp_min,
                                                            disp_max, extra_params))
    if algo == 'sgbm':
        # opencv sgbm function implements a modified version of Hirschmuller's
        # Semi-Global Matching (SGM) algorithm described in "Stereo Processing
        # by Semiglobal Matching and Mutual Information", PAMI, 2008

        p1 = 8  # penalizes disparity changes of 1 between neighbor pixels
        p2 = 32  # penalizes disparity changes of more than 1
        # it is required that p2 > p1. The larger p1, p2, the smoother the disparity

        win = 3  # matched block size. It must be a positive odd number
        lr = 1  # maximum difference allowed in the left-right disparity check
        cost = common.tmpfile('.tif')
        common.run('sgbm {} {} {} {} {} {} {} {} {} {}'.format(im1, im2,
                                                               disp, cost,
                                                               disp_min,
                                                               disp_max,
                                                               win, p1, p2, lr))

        create_rejection_mask(disp, im1, im2, mask)

    if algo == 'tvl1':
        tvl1 = 'callTVL1.sh'
        common.run('{0} {1} {2} {3} {4}'.format(tvl1, im1, im2, disp, mask),
                   env)

    if algo == 'msmw':
        bm_binary = 'iip_stereo_correlation_multi_win2'
        common.run('{0} -i 1 -n 4 -p 4 -W 5 -x 9 -y 9 -r 1 -d 1 -t -1 -s 0 -b 0 -o 0.25 -f 0 -P 32 -m {1} -M {2} {3} {4} {5} {6}'.format(bm_binary, disp_min, disp_max, im1, im2, disp, mask))

    if algo == 'msmw2':
        bm_binary = 'iip_stereo_correlation_multi_win2_newversion'
        common.run('{0} -i 1 -n 4 -p 4 -W 5 -x 9 -y 9 -r 1 -d 1 -t -1 -s 0 -b 0 -o -0.25 -f 0 -P 32 -D 0 -O 25 -c 0 -m {1} -M {2} {3} {4} {5} {6}'.format(
                bm_binary, disp_min, disp_max, im1, im2, disp, mask), env)

    if algo == 'msmw3':
        bm_binary = 'msmw'
        common.run('{0} -m {1} -M {2} -il {3} -ir {4} -dl {5} -kl {6}'.format(
                bm_binary, disp_min, disp_max, im1, im2, disp, mask))

    if algo == 'mgm':
        env['MEDIAN'] = '1'
        env['CENSUS_NCC_WIN'] = str(cfg['census_ncc_win'])
        env['TSGM'] = '3'
        env['TESTLRRL']       = str(cfg['mgm_leftright_control'])
        env['TESTLRRL_TAU']   = str(cfg['mgm_leftright_threshold'])
        env['MINDIFF']        = str(cfg['mgm_mindiff_control'])

        nb_dir = cfg['mgm_nb_directions']

        conf = '{}_confidence.tif'.format(os.path.splitext(disp)[0])

        common.run(
            '{executable} '
            '-r {disp_min} -R {disp_max} '
            '-s vfit '
            '-t census '
            '-O {nb_dir} '
            '-confidence_consensusL {conf} '
            '{im1} {im2} {disp}'.format(
                executable='mgm',
                disp_min=disp_min,
                disp_max=disp_max,
                nb_dir=nb_dir,
                conf=conf,
                im1=im1,
                im2=im2,
                disp=disp,
            ),
            env=env,
            timeout=timeout,
        )

        create_rejection_mask(disp, im1, im2, mask)


    if algo == 'mgm_multi_lsd':
        ref = im1
        sec = im2
        wref = common.tmpfile('.tif')
        wsec = common.tmpfile('.tif')
        # TODO TUNE LSD PARAMETERS TO HANDLE DIRECTLY 12 bits images?
        # image dependent weights based on lsd segments
        with rasterio.open(ref, "r") as f:
            width = f.width
            height = f.height
        #TODO refactor this command to not use shell=True
        common.run('qauto %s | \
                   lsd  -  - | \
                   cut -d\' \' -f1,2,3,4   | \
                   pview segments %d %d | \
                   plambda -  "255 x - 255 / 2 pow 0.1 fmax" -o %s' % (ref, width, height, wref),
                   shell=True)
        # image dependent weights based on lsd segments
        with rasterio.open(sec, "r") as f:
            width = f.width
            height = f.height
        #TODO refactor this command to not use shell=True
        common.run('qauto %s | \
                   lsd  -  - | \
                   cut -d\' \' -f1,2,3,4   | \
                   pview segments %d %d | \
                   plambda -  "255 x - 255 / 2 pow 0.1 fmax" -o %s' % (sec, width, height, wsec),
                   shell=True)


        env['REMOVESMALLCC'] = str(cfg['stereo_speckle_filter'])
        env['SUBPIX'] = '2'
        env['MEDIAN'] = '1'
        env['CENSUS_NCC_WIN'] = str(cfg['census_ncc_win'])
        env['MINDIFF']        = str(cfg['mgm_mindiff_control'])
        env['TESTLRRL']       = str(cfg['mgm_leftright_control'])
        env['TESTLRRL_TAU']   = str(cfg['mgm_leftright_threshold'])
        # it is required that p2 > p1. The larger p1, p2, the smoother the disparity
        regularity_multiplier = cfg['stereo_regularity_multiplier']

        nb_dir = cfg['mgm_nb_directions']

        # increasing these numbers compensates the loss of regularity after incorporating LSD weights
        P1 = 12*regularity_multiplier   # penalizes disparity changes of 1 between neighbor pixels
        P2 = 48*regularity_multiplier  # penalizes disparity changes of more than 1
        conf = disp+'.confidence.tif'

        common.run(
            '{executable} '
            '-r {disp_min} -R {disp_max} '
            '-S 6 '
            '-s vfit '
            '-t census '
            '-O {nb_dir} '
            '-wl {wref} -wr {wsec} '
            '-P1 {P1} -P2 {P2} '
            '-confidence_consensusL {conf} '
            '{im1} {im2} {disp}'.format(
                executable='mgm_multi',
                disp_min=disp_min,
                disp_max=disp_max,
                nb_dir=nb_dir,
                wref=wref,
                wsec=wsec,
                P1=P1,
                P2=P2,
                conf=conf,
                im1=im1,
                im2=im2,
                disp=disp,
            ),
            env=env,
            timeout=timeout,
        )

        create_rejection_mask(disp, im1, im2, mask)


    if algo == 'mgm_multi':
        env['REMOVESMALLCC']  = str(cfg['stereo_speckle_filter'])
        env['MINDIFF']        = str(cfg['mgm_mindiff_control'])
        env['TESTLRRL']       = str(cfg['mgm_leftright_control'])
        env['TESTLRRL_TAU']   = str(cfg['mgm_leftright_threshold'])
        env['CENSUS_NCC_WIN'] = str(cfg['census_ncc_win'])
        env['SUBPIX'] = '2'
        # it is required that p2 > p1. The larger p1, p2, the smoother the disparity
        regularity_multiplier = cfg['stereo_regularity_multiplier']

        nb_dir = cfg['mgm_nb_directions']

        P1 = 8*regularity_multiplier   # penalizes disparity changes of 1 between neighbor pixels
        P2 = 32*regularity_multiplier  # penalizes disparity changes of more than 1
        conf = '{}_confidence.tif'.format(os.path.splitext(disp)[0])

        common.run(
            '{executable} '
            '-r {disp_min} -R {disp_max} '
            '-S 6 '
            '-s vfit '
            '-t census '
            '-O {nb_dir} '
            '-P1 {P1} -P2 {P2} '
            '-confidence_consensusL {conf} '
            '{im1} {im2} {disp}'.format(
                executable='mgm_multi',
                disp_min=disp_min,
                disp_max=disp_max,
                nb_dir=nb_dir,
                P1=P1,
                P2=P2,
                conf=conf,
                im1=im1,
                im2=im2,
                disp=disp,
            ),
            env=env,
            timeout=timeout,
        )

        create_rejection_mask(disp, im1, im2, mask)

    if (algo == 'micmac'):
        # add micmac binaries to the PATH environment variable
        s2p_dir = os.path.dirname(os.path.dirname(os.path.realpath(os.path.abspath(__file__))))
        micmac_bin = os.path.join(s2p_dir, 'bin', 'micmac', 'bin')
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + micmac_bin

        # prepare micmac xml params file
        micmac_params = os.path.join(s2p_dir, '3rdparty', 'micmac_params.xml')
        work_dir = os.path.dirname(os.path.abspath(im1))
        common.run('cp {0} {1}'.format(micmac_params, work_dir))

        # run MICMAC
        common.run('MICMAC {0:s}'.format(os.path.join(work_dir, 'micmac_params.xml')))

        # copy output disp map
        micmac_disp = os.path.join(work_dir, 'MEC-EPI',
                                   'Px1_Num6_DeZoom1_LeChantier.tif')
        disp = os.path.join(work_dir, 'rectified_disp.tif')
        common.run('cp {0} {1}'.format(micmac_disp, disp))

        # compute mask by rejecting the 10% of pixels with lowest correlation score
        micmac_cost = os.path.join(work_dir, 'MEC-EPI',
                                   'Correl_LeChantier_Num_5.tif')
        mask = os.path.join(work_dir, 'rectified_mask.png')
        common.run(["plambda", micmac_cost, "x x%q10 < 0 255 if", "-o", mask])

from scipy.ndimage import gaussian_filter

def insertDepth32f(disp,fill_disp):
    # 使用rasterio库读取输入的深度地图文件
    with rasterio.open(disp) as src:
        depth = src.read(1)
        
    # 获取深度地图的高度和宽度
    height, width = depth.shape

    # 初始化两个全零矩阵，大小与深度地图相同
    integralMap = np.zeros((height, width), dtype = np.float64)
    ptsMap = np.zeros((height, width), dtype = np.int32)

    # 创建一个掩码，标记出深度地图中有效的位置（即值不为NaN的位置）
    validMask = ~np.isnan(depth)

    # 在有效的位置上，将深度地图的值赋给integralMap，对应的ptsMap设为1
    integralMap[validMask] = depth[validMask]
    ptsMap[validMask] = 1

    # 计算integralMap和ptsMap的垂直积分图
    for i in range(1, height):
        integralMap[i, :] += integralMap[i-1, :]
        ptsMap[i, :] += ptsMap[i-1, :]

    # 计算integralMap和ptsMap的水平积分图
    for j in range(1, width):
        integralMap[:, j] += integralMap[:, j-1]
        ptsMap[:, j] += ptsMap[:, j-1]

    # 定义滑动窗口的大小
    dWnd = 8

    # 当滑动窗口的大小大于1时，循环执行以下操作
    while dWnd > 1:
        # 定义当前窗口的大小
        wnd = int(dWnd)
        # 窗口大小减半
        dWnd /= 2

        # 滑动窗口遍历图像中的每个像素
        for i in range(height):
            for j in range(width):
                # 定义窗口的边界
                left = max(0, j - wnd - 1)
                right = min(j + wnd, width - 1)
                top = max(0, i - wnd - 1)
                bot = min(i + wnd, height - 1)
                
                # 计算窗口内的有效像素点数量和灰度值的总和
                ptsCnt = ptsMap[bot, right] + ptsMap[top, left] - (ptsMap[top, right] + ptsMap[bot, left])#有效像素数
                sumGray = integralMap[bot, right] + integralMap[top, left] - (integralMap[top, right] + integralMap[bot, left])

                # 如果当前窗口内没有有效的像素，跳过此轮循环
                if ptsCnt <= 0:
                    continue
                
                # 将深度图的相应位置设为窗口内的平均灰度值
                depth[i, j] = sumGray / ptsCnt

        # 对深度图进行高斯模糊处理
        # s = wnd if wnd%2 == 1 else wnd+1
        # depth = gaussian_filter(depth, sigma=s)
    with rasterio.open(fill_disp, 'w', driver='GTiff', 
                   height=depth.shape[0], 
                   width=depth.shape[1], 
                   count=1, 
                   dtype=str(depth.dtype),
                   crs=src.crs, 
                   transform=src.transform) as dst:
        dst.write(depth, 1)
   
def insertDepth32fMedian(disp, fill_disp):
    
    # 使用rasterio库读取输入的深度地图文件
    with rasterio.open(disp) as src:
        depth = src.read(1)
        
    # 获取深度地图的高度和宽度
    height, width = depth.shape
    
    # 创建一个mask，nan值以外的地方不需要滤波处理
    nan_mask = np.isnan(depth)
    
    # 初始化存储像素值的列表
    pixelValues = []

    # 定义滑动窗口的大小
    dWnd = 6

    # 当滑动窗口的大小大于1时，循环执行以下操作
    while dWnd > 1:
        # 定义当前窗口的大小
        wnd = int(dWnd)
        # 窗口大小减半
        dWnd /= 2

        # 滑动窗口遍历图像中的每个像素
        for i in range(height):
            for j in range(width):
                 
                # 如果当前深度图像素点不是空的，则跳过
                if not np.isnan(depth[i, j]):
                    continue
                
                # 定义窗口的边界
                left = max(0, j - wnd - 1)
                right = min(j + wnd, width - 1)
                top = max(0, i - wnd - 1)
                bot = min(i + wnd, height - 1)

                # 从窗口内收集有效像素值
                for x in range(left, right+1):
                    for y in range(top, bot+1):
                        if not np.isnan(depth[y, x]):
                            pixelValues.append(depth[y, x])
                            
                            
                # 如果当前窗口内有有效的像素，则取它们的中值
                if pixelValues:
                    depth[i, j] = np.median(pixelValues)

                # 清空列表，以便下次使用
                pixelValues.clear()
        # 对深度图进行高斯模糊处理
        s = wnd if wnd%2 == 1 else wnd+1
        # depth = gaussian_filter(depth, sigma=s)
        depth[nan_mask] = gaussian_filter(depth[nan_mask], sigma=s)
    with rasterio.open(fill_disp, 'w', driver='GTiff', 
                       height=depth.shape[0], 
                       width=depth.shape[1], 
                       count=1, 
                       dtype=str(depth.dtype),
                       crs=src.crs, 
                       transform=src.transform) as dst:
        dst.write(depth, 1)   

    # 使用rasterio库读取输入的深度地图文件
    with rasterio.open(disp) as src:
        depth = src.read(1)
        
    # 获取深度地图的高度和宽度
    height, width = depth.shape

    # 初始化存储像素值的列表
    pixelValues = []

    # 定义滑动窗口的大小
    dWnd = 8

    # 创建和输入深度图相同的掩码，先进行初始化为False
    mask = np.zeros_like(depth, dtype=bool)

    # 当滑动窗口的大小大于1时，循环执行以下操作
    while dWnd > 1:
        # 定义当前窗口的大小
        wnd = int(dWnd)
        # 窗口大小减半
        dWnd /= 2

        # 滑动窗口遍历图像中的每个像素
        for i in range(height):
            for j in range(width):
                 
                # 如果当前深度图像素点不是空的，则跳过
                if not np.isnan(depth[i, j]):
                    continue

                # 此位置需要应用高斯模糊，更新mask
                mask[i, j] = True
                
                # 定义窗口的边界
                left = max(0, j - wnd - 1)
                right = min(j + wnd, width - 1)
                top = max(0, i - wnd - 1)
                bot = min(i + wnd, height - 1)

                # 从窗口内收集有效像素值
                for x in range(left, right+1):
                    for y in range(top, bot+1):
                        if not np.isnan(depth[y, x]):
                            pixelValues.append(depth[y, x])
                            
                # 如果当前窗口内有有效的像素，则取它们的中值
                if pixelValues:
                    depth[i, j] = np.median(pixelValues)

                # 清空列表，以便下次使用
                pixelValues.clear()

        # 对深度图进行高斯模糊处理
        s = wnd if wnd%2 == 1 else wnd+1
        blurred = gaussian_filter(depth, sigma=s)

        # 只在mask为True的地方应用高斯模糊
        depth[mask] = blurred[mask]

    with rasterio.open(fill_disp, 'w', driver='GTiff', 
                       height=depth.shape[0], 
                       width=depth.shape[1], 
                       count=1, 
                       dtype=str(depth.dtype),
                       crs=src.crs, 
                       transform=src.transform) as dst:
        dst.write(depth, 1)