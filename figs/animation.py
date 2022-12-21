import os
import tempfile

from pandas import Series, to_datetime, DatetimeIndex, date_range, isna
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation

from gage_data import hydrograph


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
                font = ImageFont.truetype('Ubuntu-R.ttf', 60, encoding='unic')
                draw.text((data.shape[2] * 0.1, data.shape[1] * 0.85), u'{}'.format(y), font=font)
                img.thumbnail((int(img.height * 5 / 7), int(img.width * 5 / 7)))
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


def write_cummulative(_dir, jpeg):
    l = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.tif')]
    l = sorted(l, key=lambda n: int(os.path.basename(n).split('.')[0].split('_')[2]), reverse=False)
    max_cumulative = 36.
    first = True
    years = [y for y in range(1986, 2022)]

    for f, yr in zip(l, years):
        with rasterio.open(f, 'r') as src:
            print(os.path.basename(f))
            if first:
                meta = src.meta
                meta['dtype'] = rasterio.uint8
                data = src.read()
                data[data > 0] = 1
                data = data.astype(np.uint8)
                first = False
            else:
                tif = src.read()
                tif[tif > 0] = 1
                tif = tif.astype(np.uint8)
                data = np.append(data, tif, axis=0)

            data[np.isnan(data)] = 0.0
            data[data < 0] = 0.0

            arr = data.sum(axis=0)

    arr = arr.reshape((1, arr.shape[0], arr.shape[1]))
    out_final_file = os.path.join(os.path.dirname(jpeg), 'navajo_final.tif')
    meta['dtype'] = rasterio.dtypes.int16
    with rasterio.open(out_final_file, 'w', **meta) as dst:
        dst.write(arr)


def build_irr_gif(_dir, jpeg, gif, theme='cumulative', background=None, overwrite=False, paste_cmap=None):
    l = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.tif')]
    l = sorted(l, key=lambda n: int(os.path.basename(n).split('.')[0].split('_')[2]), reverse=False)
    jp_l = []
    max_cumulative = 36.
    first = True
    years = [y for y in range(1986, 2022)]

    if paste_cmap:
        paste_img = Image.open(paste_cmap)
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
                max_cumulative = np.ones_like(data) * max_cumulative

                if background:
                    with rasterio.open(background, 'r') as bck:
                        back = bck.read()
                        back = np.moveaxis(back, 0, -1)

                first = False
            else:
                tif = src.read()
                tif[tif > 0] = 1
                tif = tif.astype(np.uint8)
                if theme == 'cumulative':
                    data = np.append(data, tif, axis=0)

            data[np.isnan(data)] = 0.0
            data[data < 0] = 0.0

            zeros = np.array(data[-1, :, :] == 0)
            if theme == 'cumulative':
                rl_sum = data.sum(axis=0)
                arr = rl_sum / max_cumulative

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

            if paste_cmap:
                ratio = 1 / 4.
                h = ratio * img.size[0]
                w = paste_img.size[1] * (h / paste_img.size[0])
                paste_img.thumbnail((h, w))
                img.paste(paste_img, (int(data.shape[2] * 0.05), int(data.shape[1] * 0.75)))

            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype('Ubuntu-R.ttf', size=70, encoding='unic')
            draw.text((data.shape[2] * 0.135, data.shape[1] * 0.77), u'{}'.format(yr),
                      font=font, fill=(0, 0, 0, 255))

            img.save(out_j)
            jp_l.append(out_j)
            print('{} to png'.format(os.path.basename(out_j)))

    out_final_file = os.path.join(os.path.dirname(jpeg), 'navajo_final.tif')
    with rasterio.open(out_final_file, 'w', **meta) as dst:
        dst.write(arr)

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
    durations = [200 for _ in range(len(frames))] + [3000]
    im1.save(gif, save_all=True, append_images=frames, loop=5, duration=durations)


def et_time_series(csv, mp4, daily_q=None):
    df = hydrograph(csv)
    df['cwb'] = df['gm_ppt'] - df['gm_etr']
    df = df.loc['1991-01-01': '2020-12-31', ['cwb', 'et', 'q']]

    idx = date_range(df.index[0], df.index[-1], freq='D')
    df = df.reindex(idx)
    df = df.interpolate('linear', limit=32, axis=0)
    df.loc[isna(df['et']), ['et']] = 0.0
    if daily_q:
        q = hydrograph(daily_q).loc['1991-01-31': '2020-12-31']
        q.drop(columns=['Date'], inplace=True)

    # log of cfs to cubic meter per month
    df['q'] = np.log10(q['q'])

    y_lims = df['q'].min(), df['q'].max()

    idx = [to_datetime(i) for i in df.index]
    cols = list(df.columns)
    data = df.T.values
    data[data == 0.] = np.nan

    data = data[-1:, 5000:]

    fig, ax = plt.subplots(nrows=1, ncols=1, frameon=False, figsize=(10, 3))

    lines = []
    X = [_ for _ in range(data.shape[-1])]
    for i in range(len(data)):
        line, = ax.plot(X, data[i], color=None)
        lines.append(line)

    xfmt = mdates.DateFormatter('%b - %Y')

    def update(ii, *args):
        for i in range(len(data)):
            ax.set_ylim(y_lims[0], y_lims[1])
            ax.xaxis.set_major_formatter(xfmt)
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

            if ii < 365:
                x = [_ for _ in idx[:365]]
                lines[i].set_xdata(x)
                yl = list(data[i, :ii + 1]) + [None for _ in range(365)]
                y = yl[:365]
                lines[i].set_ydata(y)
                ax.set_xlim(left=x[0], right=x[-1])
            else:
                x = idx[ii - 365: ii + 1]
                lines[i].set_xdata(x)
                y = list(data[i, ii - 365:ii + 1])
                lines[i].set_ydata(y)
                ax.set_xlim(left=x[0], right=x[-1])

            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            plt.subplots_adjust(left=0.1, top=0.9, right=0.9, bottom=0.4, hspace=0.5, wspace=0.5)
            plt.ylabel('log (Q) [cfs]')
            plt.xlabel('Date')

        return lines

    fig.tight_layout()
    ani = FuncAnimation(
        fig, update, interval=10, blit=True, frames=data.shape[1])
    ani.save(mp4, dpi=300, writer='ffmpeg')
    # plt.gcf().autofmt_xdate()
    # plt.show()


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    root = '/media/research/IrrigationGIS/gages/gridmet_analysis'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages/gridmet_analysis'

    flder = os.path.join(root, 'figures', 'animation')
    _d = os.path.join(flder, 'tif', 'irr_navajo')
    out_jp = os.path.join(flder, 'cumulative_irr_navajo')
    _gif = os.path.join(flder, 'irr_cumulative_navajo.gif')

    naip = os.path.join(flder, 'NAIP_Navajo.tif')
    paste_cmap_ = 'fig_misc/jet_r_ramp.png'
    # write_cummulative(_d, out_jp)

    csv_ = os.path.join('/media/research/IrrigationGIS/gages/merged_q_ee/'
                        'monthly_ssebop_tc_gm_q_Comp_21DEC2021/06052500.csv')
    line_gif = os.path.join(flder, 'et_time_series.mp4')
    dq = '/media/research/IrrigationGIS/gages/hydrographs/daily_q/06052500.csv'
    # et_time_series(csv_, line_gif, daily_q=dq)
# ========================= EOF ====================================================================
