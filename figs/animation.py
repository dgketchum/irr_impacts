import os
from subprocess import check_call

from pandas import Series, to_datetime, DatetimeIndex, read_csv, date_range, isna
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib import animation

VIRIDIS = np.loadtxt('vir.txt').astype(int)


def build_et_gif(_dir, jpeg, gif, background=None, overwrite=False, freq='monthly', out_series=None):
    l = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.tif')]
    l = sorted(l, key=lambda n: (int(os.path.basename(n)[8:12]),
                                 int(os.path.basename(n).split('.')[0][13:])),
               reverse=False)
    years = [int(os.path.basename(n)[8:12]) for n in l]
    months = [int(os.path.basename(n).split('.')[0][13:]) for n in l]

    jp_l = []
    max_annual = 1206.
    durations = []

    if out_series:
        dtr = [to_datetime('{}-{}-{}'.format(years[i], months[i], 1)) for i in range(len(years))]
        idx = DatetimeIndex(dtr)
        data = [0. for _ in idx]
        series = Series(index=idx, data=data)

    with rasterio.open(background, 'r') as bck:
        back = bck.read()
        back = np.moveaxis(back, 0, -1)

    for f, y, m in zip(l, years, months):

        if m == 10:
            durations.append(200)
        else:
            durations.append(80)

        out_j = os.path.join(jpeg, os.path.basename(f).replace('.tif', '.png'))

        if os.path.exists(out_j) and not overwrite:
            jp_l.append(out_j)
            continue

        with rasterio.open(f, 'r') as src:
            if m == 4:
                data = src.read()
                max_annual_depth = np.ones_like(data) * max_annual
            else:
                tif = src.read()
                data = np.append(data, tif, axis=0)

            data[np.isnan(data)] = 0.0
            data[data < 0] = 0.0

            if out_series:
                series.loc['{}-{}-01'.format(y, m)] = data.sum()

            if m == 10 or freq == 'monthly':
                rl_sum = data.sum(axis=0)
                arr = rl_sum / max_annual_depth

                cm = plt.get_cmap('gist_rainbow')
                colored_image = cm(arr)[0, :, :, :]
                rgba = (colored_image * 255).astype(np.uint8)

                zeros = arr == 0
                zeros = np.repeat(zeros[0, :, :, np.newaxis], 3, axis=2)
                rgba[:, :, :3] = np.where(zeros, back[:, :, :3], rgba[:, :, :3])
                img = Image.fromarray(rgba)

                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("arial.ttf", 60, encoding="unic")
                draw.text((data.shape[2] * 0.1, data.shape[1] * 0.85), u"{}".format(y), font=font)
                img.save(out_j)
                jp_l.append(out_j)
                print(os.path.basename(out_j))

    print('{} seconds'.format(sum(durations) / 1000.))
    if out_series:
        series.to_csv(out_series)

    def gen_frame(path):
        im = Image.open(path)
        return im

    first, frames = True, []
    for f in jp_l:
        if first:
            im1 = gen_frame(f)
            first = False
        else:
            frames.append(gen_frame(f))
    im1.save(gif, save_all=True, append_images=frames, duration=durations)


def build_irr_gif(_dir, jpeg, gif, theme='cummulative', background=None, overwrite=False):
    l = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.tif')]
    l = sorted(l, key=lambda n: int(os.path.basename(n).split('.')[0].split('_')[2]), reverse=False)
    jp_l = []
    max_cummulative = 36.
    first = True
    years = [y for y in range(1986, 2022)]
    for f, yr in zip(l, years):
        out_j = os.path.join(jpeg, os.path.basename(f).replace('.tif', '.png'))
        if os.path.exists(out_j) and not overwrite:
            jp_l.append(out_j)
            continue
        with rasterio.open(f, 'r') as src:
            if first:
                meta = src.meta
                meta['dtype'] = rasterio.uint8
                data = src.read()
                data[data > 0] = 1
                data = data.astype(np.uint8)
                max_cummulative = np.ones_like(data) * max_cummulative

                if background:
                    with rasterio.open(background, 'r') as bck:
                        back = bck.read()
                        back = np.moveaxis(back, 0, -1)

                first = False
            else:
                tif = src.read()
                tif[tif > 0] = 1
                tif = tif.astype(np.uint8)
                if theme == 'cummulative':
                    data = np.append(data, tif, axis=0)

            data[np.isnan(data)] = 0.0
            data[data < 0] = 0.0

            zeros = np.array(data[-1, :, :] == 0)
            if theme == 'cummulative':
                rl_sum = data.sum(axis=0)
                arr = rl_sum / max_cummulative

            cm = plt.get_cmap('jet_r')
            colored_image = cm(arr)[0, :, :, :]
            rgba = (colored_image * 255).astype(np.uint8)

            if background:
                zeros = np.repeat(zeros[:, :, np.newaxis], 3, axis=2)
                rgba[:, :, :3] = np.where(zeros, back[:, :, :3], rgba[:, :, :3])
                img = Image.fromarray(rgba)
            else:
                rgba[:, :, 3] = np.where(zeros, zeros, rgba[:, :, 3])
                img = Image.fromarray(rgba)
                img.info['transparency'] = 0

            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("arial.ttf", 60, encoding="unic")
            draw.text((data.shape[2] * 0.1, data.shape[1] * 0.85), u"{}".format(yr), font=font)
            img.save(out_j)
            jp_l.append(out_j)
            print('{} to png'.format(os.path.basename(out_j)))

    def gen_frame(path):
        im = Image.open(path)
        if not background:
            im.info['transparency'] = 0
        return im

    first, frames = True, []
    for f in jp_l:
        if first:
            im1 = gen_frame(f)
            first = False
        else:
            frames.append(gen_frame(f))
    durations = [300 for _ in range(len(frames))] + [7000]
    im1.save(gif, save_all=True, append_images=frames, loop=5, duration=durations)


def et_time_series(csv, gif):
    series = read_csv(csv, index_col=0, parse_dates=True, infer_datetime_format=True)
    idx = date_range(series.index[0], series.index[-1], freq='D')
    series = series.reindex(idx)
    series = series.interpolate('linear', limit=32)
    series = series.fillna(0.0)
    x = [i for i in range(len(series.index))]
    y = series.values
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    def animate(i):
        ax.clear()
        ax.plot(x[:i], y[:i])
        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([1.1 * np.min(y), 1.1 * np.max(y)])

    anim = animation.FuncAnimation(fig, animate, frames=len(x) + 1, interval=200, blit=False)
    plt.show()


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages/gridmet_analysis'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages/gridmet_analysis'

    flder = os.path.join(root, 'figures', 'animation')
    _d = os.path.join(flder, 'tif', 'et')
    # out_jp = os.path.join(flder, 'monthly_et_png')
    out_jp = os.path.join(flder, 'monthly_et_png')
    _gif = os.path.join(flder, 'et_monthly.gif')
    csv_ = os.path.join(flder, 'et_monthly.csv')

    naip = os.path.join(flder, 'NAIP_Bozeman.tif')
    # build_et_gif(_d, out_jp, _gif, background=naip, overwrite=True, freq='monthly', out_series=csv_)

    # _d = os.path.join(flder, 'tif', 'irr_navajo')
    # out_jp = os.path.join(flder, 'cummulative_irr_navajo')
    # _gif = os.path.join(flder, 'irr_cummulative_navajo.gif')

    # naip = os.path.join(flder, 'NAIP_Navajo.tif')
    # build_irr_gif(_d, out_jp, _gif, background=naip, overwrite=True)

    line_gif = os.path.join(flder, 'et_time_series.gif')
    et_time_series(csv_, line_gif)
# ========================= EOF ====================================================================
