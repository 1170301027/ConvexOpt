#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tkinter

import numpy as np
from PIL import Image, ImageTk
from scipy.linalg import svd

MAX_ITERS = 1000
TOL = 1.0e-7

weight = 0
height = 0
def bitmap_to_mat(bitmap_seq):
    matrix = []
    for bitmap_file in bitmap_seq:
        im = Image.open(bitmap_file).convert('L')
        print(len(im.getdata()))
        global weight,height
        weight, height = im.size
        im.save("bmp_out.jpg")
        pixels = list(im.getdata())
        matrix.append(pixels)

    return np.array(matrix)


bmp_seq = map(
    lambda s: os.path.join("bmp_dir", s),
    os.listdir("bmp_dir")
)
res = bitmap_to_mat(bmp_seq)
np.save("bmp_out", res)


def show_img(matrices, w, h):
    mats = [np.load(x) for x in matrices]
    tk_win = tkinter.Toplevel()
    tk_win.title('RPCA')
    canvas = tkinter.Canvas(tk_win, width=4 * w, height=2 * h)
    canvas.pack()
    tk_ims = [None for _ in mats]
    for i, row in enumerate(mats[0]):
        ims = [Image.new('L', (w, h)) for _ in mats]
        for j, im in enumerate(ims):
            emm = list(map(float, list(mats[j][i])))
            im.putdata(emm)
            tk_ims[j] = ImageTk.PhotoImage(im)
            canvas.create_image((j * w) + 200, h, image=tk_ims[j])
            canvas.update()
            print("done")
    canvas.mainloop()

def converged(Z, d_norm):
    err = np.linalg.norm(Z, 'fro') / d_norm
    print('ERR', err)
    return err < TOL

#D = A + E

#minA,E rank(A) + λ||E||0
#minA,E ||A||∗ + λ||E||1

#L(A,Z,Y)=||A||∗+λ||Z||1+1/2μ||D−A−E||2F+<Y,D−A−E>
def svd_(X, k=-1):
    U, S, V = svd(X, full_matrices=False)
    if k < 0:
        return U, S, V
    else:
        return U[:, :k], S[:k], V[:k, :]


def rpca(X):
    m, n = X.shape
    lamda = 1. / np.sqrt(m)
    Y = X
    u, s, v = svd_(Y, k=1)
    norm_two = s[0]
    norm_inf = np.linalg.norm(Y[:], np.inf) / lamda
    dual_norm = max(norm_two, norm_inf)
    Y = Y / dual_norm
    A_hat = np.zeros((m, n))
    E_hat = np.zeros((m, n))
    mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    d_norm = np.linalg.norm(X, 'fro')
    num_iters = 0
    total_svd = 0
    sv = 1
    while True:
        num_iters += 1
        temp_T = X - A_hat + (1 / mu) * Y
        E_hat = np.maximum(temp_T - lamda / mu, 0)
        E_hat = E_hat + np.minimum(temp_T + lamda / mu, 0)
        u, s, v = svd_(X - E_hat + (1 / mu) * Y, sv)
        diagS = np.diag(s)
        svp = len(np.where(s > 1 / mu))
        if svp < sv:
            sv = min(svp + 1, n)
        else:
            sv = min(svp + round(0.05 * n), n)
        A_hat = np.dot(
            np.dot(
                u[:, 0:svp],
                np.diag(s[0:svp] - 1 / mu)
            ),
            v[0:svp, :]
        )
        total_svd = total_svd + 1
        Z = X - A_hat - E_hat
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)
        if converged(Z, d_norm) or num_iters >= MAX_ITERS:
            return A_hat, E_hat


data = np.load("bmp_out.npy")
A_hat, E_hat = rpca(data)
np.save("out_low_rank", A_hat)
np.save("out_sparse", E_hat)
show_img(["bmp_out.npy", "out_low_rank.npy", "out_sparse.npy"],weight,height)