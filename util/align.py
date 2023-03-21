from PIL import Image
import math


USE_PIXEL_CENTER = False


def align_img(img, quad, w, h):
    return img.transform((w, h), Image.QUAD, quad, Image.BILINEAR)


def align_coords(coords, quad, transform_size):
    nw = align_point(coords[0], coords[1], quad, transform_size)
    sw = align_point(coords[2], coords[3], quad, transform_size)
    se = align_point(coords[4], coords[5], quad, transform_size)
    ne = align_point(coords[6], coords[7], quad, transform_size)
    return nw + sw + se + ne


def align_point(x_prime, y_prime, quad, transform_size):
    w = transform_size
    h = transform_size

    # name of each coorinate in quad
    # (e.g. x coord of southwest corner = swx)
    nwx, nwy, swx, swy, sex, sey, nex, ney = quad

    # deltas between the coords
    # (e.g. change in x coord between the two north corners = dnx)
    dnx = nex - nwx
    dwx = swx - nwx
    dsx = sex - swx
    dny = ney - nwy
    dwy = swy - nwy
    dsy = sey - swy

    # define some shorthands that will be useful
    a0 = nwx
    a1 = dnx / w
    a2 = dwx / h
    a3 = (dsx - dnx) / (w * h)
    b0 = nwy
    b1 = dny / w
    b2 = dwy / h
    b3 = (dsy - dny) / (w * h)

    '''
    the problem can now be defined as:

    solve the following system of equations for x and y

    x_prime = a0 + a1 * x + a2 * y + a3 * x * y
    y_prime = b0 + b1 * x + b2 * y + b3 * x * y
    '''

    # additional shorthands for the next step
    p = b2 * a3 - b3 * a2
    q = (b0 * a3 - b3 * a0) + (b2 * a1 - b1 * a2) + (b3 * x_prime - a3 * y_prime)
    r = (b0 * a1 - b1 * a0) + (b1 * x_prime - a1 * y_prime)

    '''
    solving for y without any x terms yields this quadradic equation:

    p * y^2 + q * y + r = 0

    if p == 0, the solution is:

    y = -r / q

    otherwise, the solution is:

    y = (-q +/- sqrt(q^2 - 4 * p * r)) / (2 * p)

    after inspection, it seems +/- should always be treated as +
    (likely because of the constraints imposed via the ordering of the corners)
    '''
    if p == 0.:
        y = -r / q
    else:
        sqrt_term = math.sqrt(q ** 2 - 4 * p * r)
        y = (-q + sqrt_term) / (2 * p)

    # using this solution for y, solving for x yields:
    x = (x_prime - a0 - a2 * y) / (a1 + a3 * y)

    if USE_PIXEL_CENTER:
        x -= 0.5
        y -= 0.5

    return x, y


def unalign_coords(coords, quad, transform_size):
    nw = unalign_point(coords[0], coords[1], quad, transform_size)
    sw = unalign_point(coords[2], coords[3], quad, transform_size)
    se = unalign_point(coords[4], coords[5], quad, transform_size)
    ne = unalign_point(coords[6], coords[7], quad, transform_size)
    return nw + sw + se + ne


def unalign_point(x, y, quad, transform_size):
    nwx, nwy, swx, swy, sex, sey, nex, ney = quad
    w = transform_size
    h = transform_size

    if USE_PIXEL_CENTER:
        x += 0.5
        y += 0.5

    x_prime = nwx + x * (nex - nwx) / w + y * (swx - nwx) / h + \
              x * y * (sex - swx - nex + nwx) / (w * h)

    y_prime = nwy + x * (ney - nwy) / w + y * (swy - nwy) / h + \
              x * y * (sey - swy - ney + nwy) / (w * h)

    return x_prime, y_prime
