# -*- coding: utf-8 -*-
# Created on Tue Jul  8 09:53:10 2014

import numpy as np
import pyfftw
cimport numpy as cnp
import time
import cmath

#def getac(int ntaus, corrmethod, cnp.ndarray[cnp.uint16_t, ndim=3] imgs):
##    t = time.time()
##    print time.time() - t
#    print 'calculating correlations',
#    if corrmethod == 'fixed':
#        corr = getacfixed(ntaus, imgs)
#    elif corrmethod == 'brute force':
#        corr = getacvariable(ntaus, imgs)
#        print '\ncalculating normalizations',
#        normalizeacvariable(ntaus, imgs, corr)
#    corrmean = np.mean(corr, axis=0)
#    sd = np.std(corr, axis=0)
#    print 'done'
#    return corrmean, sd


def getacnorm(int ntaus, corrmethod, cnp.ndarray[cnp.uint16_t, ndim=3] imgs):
    '''Definition g = G - 1'''
    cdef cnp.ndarray[cnp.double_t, ndim=2] corr
    cdef cnp.ndarray[cnp.double_t, ndim=2] norm
#    t = time.time()
    corr = getacunnorm(ntaus, corrmethod, imgs)
#    print 'time (', corrmethod, '):', time.time() - t
    print 'calculating normalizations',
    norm = getnormvariable(ntaus, imgs)
    corr = normalize(corr, norm)
    print 'done'
    return corr


def getacunnorm(int ntaus, corrmethod, cnp.ndarray[cnp.uint16_t, ndim=3] imgs):
    print 'calculating correlations',
    cdef cnp.ndarray[cnp.double_t, ndim=2] corr
    print corrmethod,
    if corrmethod == 'brute force':
        corr = getacunnormvariable(ntaus, imgs)
    if corrmethod == 'fft':
        corr = getacunnormfft(ntaus, imgs)
    print 'done'
    return corr


def getnorm(int ntaus, corrmethod, cnp.ndarray[cnp.uint16_t, ndim=3] imgs):
    cdef cnp.ndarray[cnp.double_t, ndim=2] norms
    if corrmethod == 'brute force' or corrmethod == 'fft':
        print 'calculating norms',
        norms = getnormvariable(ntaus, imgs)
    print 'done'
    return norms


cdef cnp.ndarray[cnp.double_t, ndim=2] getacunnormfft(int ntaus,
cnp.ndarray[cnp.uint16_t, ndim=3] imgs):
    '''Count line-scan autocorrelation with Fast Fourier Transform. For smaller
    tau values, more pixel pairs are used for calculations. This method does
    not normalize at all. Even rows are not averaged.'''
    cdef cnp.ndarray[cnp.double_t, ndim=2] corr = np.empty((len(imgs), ntaus), dtype=np.double)
    cdef cnp.ndarray[cnp.complex128_t, ndim=2] fft
    cdef cnp.ndarray[cnp.double_t, ndim=2] ifft
    cdef int nimgs = len(imgs)
    cdef int nrows = len(imgs[0])
    cdef int npx = len(imgs[0,0])
    cdef int i
    cdef int j
    cdef cnp.ndarray[cnp.uint16_t, ndim=3] imgs2 = np.pad(imgs,
            ((0,0),(0,0),(0,npx*2)), mode='constant')
    fft_object = pyfftw.builders.rfft2(imgs2[0], axes=(1,))
    ifft_object = pyfftw.builders.irfft2(fft_object(), axes=(1,))
    cdef float part = max(1,(len(imgs)-1)/10.)
    for i in range(nimgs):
        fft_object.get_input_array()[:] = imgs2[i] # element-wise assignment in np.arrays
        fft = fft_object()
        ifft_object.get_input_array()[:] = fft.real**2 + fft.imag**2 # np.conjugate(fft) * fft
        ifft = ifft_object()
        corr[i] = np.sum(ifft, axis=0)[:ntaus] # rows should not be averaged in this method
        if i%part < 1:
            print int(i/part),

#    fft_object = pyfftw.builders.rfft2(imgs2[0], axes=(1,))
#    cdef float part = max(1,(len(imgs)-1)/10.)
#    for i in range(nimgs):
#        fft_object.get_input_array()[:] = imgs2[i]**2 # element-wise assignment in np.arrays
#        fft = fft_object()
#        corr[i] = np.sum(fft, axis=0)[:ntaus] # rows should not be averaged in this method
#        if i%part < 1:
#            print int(i/part),
    return corr


cdef cnp.ndarray[cnp.double_t, ndim=2] getacunnormvariable(int ntaus,
cnp.ndarray[cnp.uint16_t, ndim=3] imgs):
    '''Count line-scan autocorrelation. For smaller tau values, more pixel pairs
    are used for calculations. This method does not normalize at all. Even rows
    are not averaged.'''
    cdef cnp.ndarray[cnp.double_t, ndim=2] corr = np.empty((len(imgs), ntaus), dtype=np.double)
    cdef cnp.ndarray[cnp.double_t, ndim=1] corr_tau = np.zeros(ntaus, dtype=np.double)
    cdef cnp.uint32_t corr_ab = 0
    cdef cnp.ndarray[cnp.uint16_t, ndim=2] img
    cdef int nimgs = len(imgs)
    cdef int nrows = len(imgs[0])
    cdef int npx = len(imgs[0,0])
    cdef int i
    cdef int j
    cdef int px
    cdef int tau
    cdef float part = max(1,(len(imgs)-1)/10.)
    for i in range(nimgs):
        img = imgs[i]
        for tau in range(ntaus):
            for j in range(nrows):
                for px in range(npx-tau):
                    corr_ab += img[j,px]*img[j,px+tau]
            corr_tau[tau] = corr_ab
            corr_ab = 0
        corr[i] = corr_tau
        corr_tau.fill(0)
        if i%part < 1:
            print int(i/part),
    # fixed
#    for i in range(nimgs):
#        img = imgs[i]
#        for tau in range(ntaus):
#            for j in range(nrows):
#                for px in range(npx-ntaus+1):
#                    corr_ab += img[j,px]*img[j,px+tau]
#            corr_tau[tau] = corr_ab
#            corr_ab = 0
#        corr[i] = corr_tau
#        corr_tau.fill(0)
#        if i%part < 1:
#            print int(i/part),
    return corr


cdef cnp.ndarray[cnp.double_t, ndim=2] getnormvariable(int ntaus,
cnp.ndarray[cnp.uint16_t, ndim=3] imgs):
    '''Count normalization for line-scan autocorrelation.'''
    cdef cnp.ndarray[cnp.double_t, ndim=2] norm = np.zeros((len(imgs), ntaus), dtype=np.double)
    cdef cnp.ndarray[cnp.uint16_t, ndim=2] img
    cdef int nimgs = len(imgs)
    cdef int nrows = len(imgs[0])
    cdef int npx = len(imgs[0,0])
    cdef int i
    cdef int j
    cdef int px
    cdef int tau
    cdef double npoints
    cdef cnp.uint32_t norm1 = 0
    cdef cnp.uint32_t norm2 = 0
    cdef float part = max(1,(len(imgs)-1)/10.)
    for i in range(nimgs):
        img = imgs[i]
        # first calculate initial sums (tau == ntaus)
        for j in range(nrows):
            for px in range(npx-ntaus):
                norm1 += img[j,px]
                norm2 += img[j,px+ntaus]
        # for each tau value add the corresponding column to the norms
        for tau in range(ntaus-1, -1, -1):
            for j in range(nrows):
                norm1 += img[j,npx-tau-1]
                norm2 += img[j,tau]
            npoints = nrows*(npx-tau)
            norm[i,tau] = (1.*norm1*norm2)/npoints
        norm1 = 0
        norm2 = 0
        if i%part < 1:
            print int(i/part),
    # average whole image
#    for i in range(nimgs):
#        img = imgs[i]
#        avg = np.mean(img)
#        for tau in range(ntaus-1, -1, -1):
#            npoints = nrows*(npx-tau)
#            norm[i,tau] = npoints*avg**2
#        norm1 = 0
#        norm2 = 0
#        if i%part < 1:
#            print int(i/part),
    # fixed number of points
#    for i in range(nimgs):
#        img = imgs[i]
#        avg = np.mean(img)
#        for tau in range(ntaus-1, -1, -1):
#            npoints = nrows*(npx-ntaus+1)
#            norm[i,tau] = npoints*avg**2
#        norm1 = 0
#        norm2 = 0
#        if i%part < 1:
#            print int(i/part),
    return norm


cdef cnp.ndarray[cnp.double_t, ndim=2] normalize(
cnp.ndarray[cnp.double_t, ndim=2] corr, cnp.ndarray[cnp.double_t, ndim=2] norm):
    '''Normalize so that the result is in accordance with the definition g = G - 1'''
    cdef int i
    cdef int j
    for i in range(len(corr)):
        for j in range(len(corr[i])):
            corr[i,j] = corr[i,j]/norm[i,j] - 1
    return corr
            


cdef cnp.ndarray[cnp.double_t, ndim=2] getacnormvariable(int ntaus,
cnp.ndarray[cnp.uint16_t, ndim=3] imgs, cnp.ndarray[cnp.double_t, ndim=2] corr):
    cdef cnp.ndarray[cnp.double_t, ndim=2] corrn = np.zeros((len(imgs), ntaus), dtype=np.double)
    cdef cnp.ndarray[cnp.uint16_t, ndim=2] img
    cdef int nimgs = len(imgs)
    cdef int nrows = len(imgs[0])
    cdef int npx = len(imgs[0,0])
    cdef int i
    cdef int j
    cdef int px
    cdef int tau
    cdef double npoints
    cdef cnp.uint32_t norm1 = 0
    cdef cnp.uint32_t norm2 = 0
    cdef float part = max(1,(len(imgs)-1)/10.)
    for i in range(nimgs):
        img = imgs[i]
        # first calculate initial sums (tau == ntaus)
        for j in range(nrows):
            for px in range(npx-ntaus):
                norm1 += img[j,px]
                norm2 += img[j,px+ntaus]
        # for each tau value add the corresponding column to the norms
        for tau in range(ntaus-1, -1, -1):
            for j in range(nrows):
                norm1 += img[j,npx-tau-1]
                norm2 += img[j,tau]
            npoints = nrows*(npx-tau)
            corrn[i,tau] = corr[i,tau]*npoints/(1.*norm1*norm2) - 1
        norm1 = 0
        norm2 = 0
        if i%part < 1:
            print int(i/part),
    return corrn


cdef cnp.ndarray[cnp.double_t, ndim=2] getacfixed(int ntaus, cnp.ndarray[cnp.uint16_t, ndim=3] imgs):
    '''Count line-scan autocorrelation. All tau values are calculated from the same
    amount of pixel pairs (tau does not affect). No difference to correlation2
    if data is very homogenous (this is faster, though). Features on the
    right edge of the image may be neglected at short tau values'''
    cdef cnp.ndarray[cnp.double_t, ndim=2] corr = np.zeros((len(imgs), ntaus), dtype=np.double)
    cdef cnp.ndarray[cnp.uint16_t, ndim=2] img
    cdef int nimgs = len(imgs)
    cdef int nrows = len(imgs[0])
    cdef int maxpx = len(imgs[0,0]) - ntaus
    cdef int i
    cdef int j
    cdef int px
    cdef int tau
    cdef cnp.uint16_t b
    cdef double npixels = nrows*(maxpx+1)
    cdef cnp.uint32_t corr_ab = 0
    cdef cnp.uint32_t norm1 = 0
    cdef cnp.uint32_t norm2 = 0
    cdef float part = max(1,(len(imgs)-1)/10.)
    for i in range(nimgs):
        img = imgs[i]
        for j in range(nrows):
            for px in range(maxpx+1):
                norm1 += img[j,px]
        for tau in range(ntaus):
            for j in range(nrows):
                for px in range(maxpx+1):
                    b = img[j,px+tau]
                    corr_ab += img[j,px]*b
                    norm2 += b
            corr[i,tau] = corr_ab*npixels/(1.*norm1*norm2) - 1
            corr_ab = 0
            norm2 = 0
        norm1 = 0
        if i%part < 1:
            print int(i/part),
    return corr


cdef cnp.ndarray[cnp.double_t, ndim=2] getacunnormfftct(int ntaus,
cnp.ndarray[cnp.uint16_t, ndim=3] imgs):
    '''WORK IN PROGRESS! Count line-scan autocorrelation with Fast Fourier Transform. For smaller
    tau values, more pixel pairs are used for calculations. This method does
    not normalize at all. Even rows are not averaged.'''
    cdef cnp.ndarray[cnp.double_t, ndim=2] corr = np.zeros((len(imgs), ntaus), dtype=np.double)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] fft = np.zeros(len(imgs[0,0]), dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] ifft
    cdef int nimgs = len(imgs)
    cdef int nrows = len(imgs[0])
    cdef int npx = len(imgs[0,0])
    cdef int i
    cdef int j
    
    cdef cnp.ndarray[cnp.complex128_t, ndim=3] cimgs = imgs.astype(np.complex128)
    cdef int n = npx
    cdef int levels = int(np.log2(n))
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] expsfw = np.empty(n//2, dtype=np.complex128)
    cdef cnp.ndarray[cnp.complex128_t, ndim=1] expsbw = np.empty(n//2, dtype=np.complex128)
    for i in range(n//2):
        expsfw[i] = cmath.exp(-2j * cmath.pi * i / n)
    for i in range(n//2):
        expsbw[i] = cmath.exp(2j * cmath.pi * i / n)
    cdef cnp.ndarray[cnp.long_t, ndim=1] reversedinds = np.empty(n, dtype=np.long)
    cdef int y
    for i in range(n):
        y = 0
        for j in range(levels):
            y = (y << 1) | (i & 1)
            i >>= 1
        reversedinds[i] = y
    
    for i in range(nimgs):
        for j in range(nrows):
            fft = fftct(n, cimgs[i,j], fft, expsfw, reversedinds)
            ifft = fftct(n, fft, fft, expsbw, reversedinds).real[:ntaus]
            corr[i] += ifft
    return corr
    
cdef cnp.ndarray[cnp.complex128_t, ndim=1] fftct(int n,
cnp.ndarray[cnp.complex128_t, ndim=1] vin,
cnp.ndarray[cnp.complex128_t, ndim=1] vout,
cnp.ndarray[cnp.complex128_t, ndim=1] exps,
cnp.ndarray[cnp.long_t, ndim=1] reversedinds):
    # Initialization
    cdef int i
    for i in range(n):
        print i, n, reversedinds[i], len(vin), len(vout)
        vout[i] = vin[reversedinds[i]]  # Copy with bit-reversed permutation
    
    # Radix-2 decimation-in-time FFT
    cdef int size = 2
    cdef int halfsize
    cdef int tablestep
    cdef int k
    while size <= n:
        halfsize = size // 2
        tablestep = n // size
        for i in range(0, n, size):
            k = 0
            for j in range(i, i + halfsize):
                temp = vout[j + halfsize] * exps[k]
                vout[j + halfsize] = vout[j] - temp
                vout[j] += temp
                k += tablestep
        size *= 2
    return vout











