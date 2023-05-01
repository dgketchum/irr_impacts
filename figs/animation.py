import os
import shutil
from datetime import datetime, timedelta

from pandas import Series, to_datetime, DatetimeIndex, date_range, isna
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
import requests

import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation

from gage_data import hydrograph
from utils.gridmet_data import GridMet


def build_et_gif(_dir, jpeg, gif, background=None, overwrite=False, freq='annual_accum', paste_cmap=None):
    l = [os.path.join(_dir, x) for x in os.listdir(_dir) if x.endswith('.tif')]
    if 'Bozeman' in l[0]:
        l = sorted(l, key=lambda n: (int(os.path.basename(n)[8:12]),
                                     int(os.path.basename(n).split('.')[0][13:])), reverse=False)
        years = [int(os.path.basename(n)[8:12]) for n in l]
        months = [int(os.path.basename(n).split('.')[0][13:]) for n in l]
        max_annual = 1206.
    else:
        l = sorted(l, key=lambda n: (int(os.path.basename(n).split('_')[2]),
                                     int(os.path.basename(n).split('_')[3].split('.')[0])), reverse=False)
        years = [int(os.path.basename(n).split('_')[2]) for n in l]
        months = [int(os.path.basename(n).split('_')[3].split('.')[0]) for n in l]
        max_annual = 1.4

    jp_l = []
    durations = []

    if paste_cmap:
        paste_img = Image.open(paste_cmap)

    first = True
    with rasterio.open(background, 'r') as bck:
        back = bck.read()
        back = np.moveaxis(back, 0, -1)

    for f, y, m in zip(l, years, months):

        if m == 10 and freq == 'annual_accum':
            durations.append(200)
        else:
            durations.append(80)

        out_j = os.path.join(jpeg, os.path.basename(f).replace('.tif', '.png'))

        if os.path.exists(out_j) and not overwrite:
            jp_l.append(out_j)
            continue

        with rasterio.open(f, 'r') as src:

            if (m == 4 and freq == 'annual_accum') or first:
                data = src.read()
                max_annual_depth = np.ones_like(data) * max_annual
                first = False
            else:
                tif = src.read()
                data = np.append(data, tif, axis=0)

            data[np.isnan(data)] = 0.0
            data[data < 0] = 0.0

            rl_sum = data.sum(axis=0)
            arr = rl_sum / max_annual_depth

            cm = plt.get_cmap('viridis_r')
            colored_image = cm(arr)[0, :, :, :]
            rgba = (colored_image * 255).astype(np.uint8)

            zeros = arr == 0
            zeros = np.repeat(zeros[0, :, :, np.newaxis], 3, axis=2)
            rgba[:, :, :3] = np.where(zeros, back[:, :, :3], rgba[:, :, :3])
            img = Image.fromarray(rgba)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype('Ubuntu-R.ttf', 60, encoding='unic')

            img.thumbnail((int(img.height * 5 / 7), int(img.width * 5 / 7)))

            if paste_cmap:
                paste_ = paste_img.copy()
                paste_.thumbnail((int(paste_.height * 0.2), int(paste_.width * 0.2)))
                img.paste(paste_, (int(img.height * 0.05), int(img.width * 0.7)))

            if freq == 'annual_accum':
                draw.text((data.shape[2] * 0.1, data.shape[1] * 0.85), u'{}'.format(y), font=font)
            else:
                draw.text((data.shape[2] * 0.12, data.shape[1] * 0.8), u'{} / {}'.format(str(m).zfill(2), y),
                          font=font, fill='black')

            img.save(out_j)
            jp_l.append(out_j)
            print(os.path.basename(out_j), '{:.1f}'.format(rl_sum.max()))

    print('{} seconds'.format(sum(durations) / 1000.))

    def gen_frame(path):
        im = Image.open(path)
        return im

    first, frames = True, []
    for f in jp_l[-7:]:
        if first:
            im1 = gen_frame(f)
            first = False
        else:
            frames.append(gen_frame(f))

    im1.save(gif, save_all=True, append_images=frames, duration=durations)


def write_cummulative_irr(_dir, jpeg):
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


def animated_time_series(mp4, daily_data=None, param='q'):
    start, end = '1991-01-31', '2020-12-31'
    # Gallatin at Logan, MT
    lon, lat = -111.1489, 45.6724

    if param == 'q':
        df = hydrograph(daily_data).loc[start: end]
        df['q'] = np.log10(df['q'])
        y_lims = df['q'].min(), df['q'].max()
        y_label = 'log (Q) [cfs]'
        color = 'blue'
    else:
        grd = GridMet(variable=param, start=start, end=end,
                      lat=lat, lon=lon)
        df = grd.get_point_timeseries()
        df.dropna(how='any', axis=0, inplace=True)
        y_lims = df[param].min(), df[param].max()

        if param == 'etr':
            color = 'red'
            y_label = 'Reference ET mm day$^-1$'
        else:
            color = 'green'
            y_label = 'Precipitation mm day$^-1$'

    idx = [to_datetime(i) for i in df.index]
    data = df.T.values

    if not param == 'pr':
        data[data == 0.] = np.nan

    fig, ax = plt.subplots(nrows=1, ncols=1, frameon=False, figsize=(10, 3))

    lines = []
    X = [_ for _ in range(data.shape[-1])]
    for i in range(len(data)):
        line, = ax.plot(X, data[i], color=None)
        lines.append(line)

    xfmt = mdates.DateFormatter('%b - %Y')

    def update(ii, *args):
        ax.set_ylim(y_lims[0], y_lims[1])
        ax.xaxis.set_major_formatter(xfmt)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

        if ii < 365:
            x = [_ for _ in idx[:365]]
            lines[i].set_xdata(x)
            yl = list(data[0, :ii + 1]) + [None for _ in range(365)]
            y = yl[:365]

        else:
            x = idx[ii - 365: ii + 1]
            lines[i].set_xdata(x)
            y = list(data[0, ii - 365:ii + 1])

        lines[i].set_ydata(y)
        lines[i].set_color(color)
        ax.set_xlim(left=x[0], right=x[-1])

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.subplots_adjust(left=0.1, top=0.9, right=0.9, bottom=0.4, hspace=0.5, wspace=0.5)
        plt.ylabel(y_label)
        plt.xlabel('Date')
        return lines

    fig.tight_layout()
    ani = FuncAnimation(
        fig, update, interval=10, blit=True, frames=data.shape[1])
    ani.save(mp4, dpi=300, writer='ffmpeg')
    # plt.gcf().autofmt_xdate()
    # plt.show()


def usdm_png(png_dir):
    dt_str = '20230425'
    dt = datetime.strptime(dt_str, '%Y%m%d')
    url = 'https://droughtmonitor.unl.edu/data/png/{}/{}_west_text.png'
    for i in range(365 * 10):
        resp = requests.get(url.format(dt_str, dt_str), stream=True)
        out_png = os.path.join(png_dir, '{}.png'.format(dt_str))
        with open(out_png, 'wb') as f:
            resp.raw.decode_content = True
            shutil.copyfileobj(resp.raw, f)
        dt = (dt - timedelta(days=7))
        dt_str = dt.strftime('%Y%m%d')
        if dt < datetime(2016, 1, 1):
            break
        print(dt_str)


def usdm_animation(png_dir, modified, gif, overwrite_png=False):
    im1 = None
    l = sorted([os.path.join(png_dir, x) for x in os.listdir(png_dir)])
    dt_strs = [os.path.basename(f).split('.')[0] for f in l]
    png_l = []
    for f, dstr in zip(l, dt_strs):
        mod_png = os.path.join(modified, '{}.png'.format(dstr))
        if os.path.exists(mod_png) and not overwrite_png:
            continue
        img = Image.open(f)
        dt = datetime.strptime(dstr, '%Y%m%d')
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('Ubuntu-R.ttf', size=40, encoding='unic')
        draw.text((img.width * 0.7, img.height * 0.2), u'{}'.format(dt.strftime('%m / %Y')),
                  font=font, fill=(0, 0, 0, 255))
        draw.rectangle(((img.width * 0.6, img.height * 0.0),
                        (img.width * 0.95, img.height * 0.2)), fill='white')
        draw.rectangle(((img.width * 0.7, img.height * 0.75),
                        (img.width * 0.95, img.height * 0.85)), fill='white')
        img.save(mod_png)
        png_l.append(mod_png)
        print('{} to png'.format(os.path.basename(mod_png)))

    if not png_l:
        png_l = [os.path.join(modified, '{}.png'.format(dstr)) for dstr in dt_strs]

    def gen_frame(path):
        im = Image.open(path)
        return im

    first, frames = True, []
    for f in png_l:
        if first:
            im1 = gen_frame(f)
            first = False
        else:
            frames.append(gen_frame(f))
    durations = [50 for _ in range(len(frames))] + [3000]
    print('writing {}'.format(gif))
    im1.save(gif, save_all=True, append_images=frames, loop=5, duration=durations)


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    root = '/media/research/IrrigationGIS/impacts'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/impacts'
    flder = os.path.join(root, 'figures', 'animation')
    tif = os.path.join(flder, 'tif', 'et_navajo')
    png = os.path.join(flder, 'cumulative_et_navajo')
    naip = os.path.join(flder, 'NAIP_Navajo.tif')
    et_ = os.path.join(flder, 'et_navajo.gif')
    cmap_ = '/home/dgketchum/PycharmProjects/irr_impacts/figs/fig_misc/viridis_ramp.png'
    build_et_gif(tif, png, et_, background=naip, overwrite=False, freq='annual_accum', paste_cmap=cmap_)

    tif = os.path.join(flder, 'tif', 'irr_navajo')
    png = os.path.join(flder, 'cumulative_irr_navajo')
    gif = os.path.join(flder, 'irr_cumulative_navajo_def.gif')
    naip = os.path.join(flder, 'NAIP_Navajo.tif')
    cmap_ = '/home/dgketchum/PycharmProjects/irr_impacts/figs/fig_misc/jet_r_ramp.png'
    # build_irr_gif(tif, png, gif, background=naip, overwrite=True, paste_cmap=cmap_)

# ========================= EOF ====================================================================
