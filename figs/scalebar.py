import numpy as np
import cartopy.crs as ccrs


def scale_bar(ax, projection, bars=4, length=None, location=(0.45, 0.22), linewidth=10,
              col='black'):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """

    x0, x1, y0, y1 = ax.get_extent(projection)
    sbx = x0 + (x1 - x0) * location[0]
    txt_x = x0 + (x1 - x0) * (location[0] - 0.001)
    sby = y0 + (y1 - y0) * location[1]
    txt_y = y0 + (y1 - y0) * (location[1] + 0.005)
    unit_y = ((abs(y0) + abs(y1)) * (location[1] - 0.025)) + y0

    if not length:
        length = (x1 - x0) / 5000  # in km
        ndim = int(np.floor(np.log10(length)))  # number of digits in number
        length = round(length, -ndim)  # round to 1sf

        # Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']:
                return int(x)
            else:
                return scale_number(x - 10 ** ndim)

        length = scale_number(length)

    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx, sbx + length * 1000 / bars]
    # Plot the scalebar chunks
    barcol = 'black'
    for i in range(0, bars):
        # plot the chunk
        ax.plot(bar_xs, [sby, sby], transform=projection, color=barcol, linewidth=linewidth)
        # alternate the colour
        if barcol == 'white':
            barcol = 'black'
        else:
            barcol = 'white'
        # Generate the x coordinate for the number
        bar_xt = txt_x + i * length * 1000 / bars
        # Plot the scalebar label for that chunk
        ax.text(bar_xt, txt_y, str(round(i * length / bars)), transform=projection,
                horizontalalignment='center', verticalalignment='bottom',
                color=col, fontsize=15)
        # work out the position of the next chunk of the bar
        bar_xs[0] = bar_xs[1]
        bar_xs[1] = bar_xs[1] + length * 1000 / bars
    # Generate the x coordinate for the last number
    bar_xt = txt_x + length * 1000
    # Plot the last scalebar label
    ax.text(bar_xt, txt_y, str(round(length)), transform=projection,
            horizontalalignment='center', verticalalignment='bottom',
            color=col, fontsize=15)
    # Plot the unit label below the bar
    bar_xt = sbx + length * 1000 / 2
    ax.text(bar_xt, unit_y, 'km', transform=projection, horizontalalignment='center',
            verticalalignment='bottom', color=col, fontsize=15)

    x_north = sbx + length * 1000 / 2
    x_north_end = sbx - 10000 + length * 1000 / 2 + 30000
    y_north = y0 + (y1 - y0) * (location[1] + 0.025)
    y_north_end = y0 + (y1 - y0) * (location[1] + 0.1)
    ax.annotate('N',
                xy=(x_north, y_north_end),
                xytext=(x_north, y_north - 15000),
                ha='center',
                va='center',
                fontsize=18)
    ax.annotate('',
                xy=(x_north_end, y_north_end),
                xytext=(x_north + 5000, y_north + 15000),
                arrowprops=dict(facecolor='black', width=5, headwidth=15, lw=0.5),
                ha='center',
                va='center',
                fontsize=18)
