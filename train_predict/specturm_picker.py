import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib as mpl
import argparse
import filter_utils

plt.rcParams['font.family'] = ['Helvetica Neue', 'Microsoft YaHei']


def main(args):
    data = np.fromfile(args.data, np.float32).reshape(args.shape)
    x, y = filter_utils.fftNd(data, args.dt, fmax=180)

    # # 假设您已有原始的频谱数据 y(x)
    # xy = np.load('zjfreq_raw.npy')
    # x = xy[:, 0]
    # y = xy[:, 1]

    # xy2 = np.load('zjfreq_sr_raw.npy')
    # x2 = xy2[:, 0]
    # y2 = xy2[:, 1]

    # 提示用户在图上选择点
    print("请在弹出的图上点击以选择点。完成后关闭图窗口。")

    # 再次绘制以供选择
    plt.figure()
    plt.plot(x, y, label='原始频谱')
    # plt.plot(x2, y2, label='sr', alpha=0.5)
    plt.plot([x[0], x[-1]], [-15, -15], '--', c='black')
    plt.title('请点击选择点')
    plt.grid(True, linestyle='--', lw=0.8, alpha=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.ylim(y.min(), 0)
    plt.ylim(-20, 0)
    plt.xlim(0, x.max())

    # 使用 ginput 收集用户点击的点
    selected_points = plt.ginput(n=-1, timeout=0)
    plt.close()

    # 提取选取的 x 和 y 坐标
    selected_x, selected_y = zip(*selected_points)
    selected_x = np.array(selected_x)
    selected_y = np.array(selected_y)

    # 按 x 坐标排序
    sorted_indices = np.argsort(selected_x)
    selected_x = selected_x[sorted_indices]
    selected_y = selected_y[sorted_indices]

    unique_x, unique_indices = np.unique(selected_x, return_index=True)
    unique_y = selected_y[unique_indices]

    # 根据选取的点进行插值拟合
    interp_func = interp1d(unique_x,
                           unique_y,
                           kind='linear',
                           bounds_error=False,
                           fill_value='extrapolate')

    # 按照原始 x 进行采样，得到新的频谱曲线
    new_x = np.arange(200)
    new_y = interp_func(new_x)
    new_y[0] = y[0]

    np.save(args.output, np.c_[new_x, new_y])
    # new_y[-1] = y[-1]

    # 绘制新的频谱曲线
    plt.figure()
    plt.plot(x, y, label='原始频谱')
    # plt.plot(x2, y2, label='sr', alpha=0.5)
    plt.plot(new_x, new_y, label='拟合频谱', linestyle='--')
    plt.plot([x[0], x[-1]], [-15, -15], '--', c='black')
    plt.scatter(selected_x, selected_y, color='red', label='选取的点')
    plt.title('拟合后的频谱曲线')
    plt.grid(True, linestyle='--', lw=0.8, alpha=0.8)
    plt.xlabel('x')
    plt.ylim(y.min(), 0)
    plt.xlim(0, x.max())
    plt.ylabel('y')
    plt.legend()
    plt.show()


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

if __name__ == '__main__':
    script_name = sys.argv[0] if sys.argv[0] else "specturm_picker.py"
    examples = f'''
示例:
    python {script_name} --data data/subd_3_1801_400.dat --shape 3 1801 400 --dt 0.002 --output zjfreq_ref.npy
'''
    parser = argparse.ArgumentParser(
        description='参考频谱制作工具',
        epilog=examples,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--data', type=str, required=True, help="(必须)数据路径, 格式要求为二进制的bin/dat格式或者npy格式,时间维度(nt)必须在最后")
    parser.add_argument('--shape', type=int, nargs='+', required=True, help="(必须)数据维度, 二维为 nw nt, 三维为 ni nx nt")
    parser.add_argument('--dt', type=float, default=0.002, help="时间采样间隔, 单位 秒(s)")
    parser.add_argument('--output', type=str, default='ref.npy', help="输出文件路径")
    args = parser.parse_args()
    main(args)
