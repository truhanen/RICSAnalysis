# -*- coding: utf-8 -*-
# Created on Tue Oct  1 13:21:36 2013

'''
Cython algorithms for signal correlation analysis.
'''

import numpy as np
cimport numpy as cnp
import time
import sys
from cpython cimport bool
from libc.stdlib cimport malloc, realloc, free

# dtypeByte = np.uint8
ctypedef cnp.uint8_t uint8_t  # 255
# dtypeRaw = np.uint16
ctypedef cnp.uint16_t uint16_t  # 65 535
# dtypeTime = np.uint32
ctypedef cnp.uint32_t uint32_t  # 4 294 967 295
# dtypeCorr = np.float
ctypedef cnp.float_t float_t  # 32 bits

cdef int lenWord = 4  # number of clock cycles in a word

cdef cnp.ndarray[float_t, ndim = 1] getNormalizationsC(int lenCorr, int totCount, cnp.ndarray[uint32_t, ndim=1] countDwellInds, int dwellCount):
    """see symmetric acf normalization (Saffarian & Elson)"""
    cdef cnp.ndarray[float_t, ndim = 1] normalizations = np.zeros(lenCorr, dtype=np.float)
#    cdef float_t* normalizations2 = <float_t*> malloc(lenCorr*sizeof(float_t))
    cdef int totCounti = 0
    cdef int totCountj = 0
    cdef float_t resSum = 0
    cdef int i
    cdef int v
    # Types of normalization:
    # 1) Symmetric (Saffarian & Elson)
    # 2) Simple intensity average (sum of residuals between 1 and 2 is of
    # order 1e-05 (dwell time 4))

    # 1)
    for v in range(lenCorr, 0, -1):  # v decreases from maxTau to 1
        # count the sums of the normalization
        # add to totCounti if there are uncounted counts between 1 and N-v
        while totCounti < totCount and countDwellInds[totCounti] <= dwellCount - v:
            totCounti += 1
        # add to totCountj if there are uncounted counts between v and N
        while totCountj < totCount - 1 and countDwellInds[(totCount - 1) - totCountj] >= v:
            totCountj += 1
        # calculate normalization for the given tau value (v)
        normalizations[v - 1] = (
            1. * (dwellCount - v)) ** -2 * totCounti * totCountj
    # 2)
#    for i in range(lenCorr):
#        normalizations2[i] = (1.*totCount/dwellCount)**2
#    for i in range(lenCorr):
#        resSum += abs(normalizations2[i] - normalizations[i])
# print '{:.3e} sum of residuals between normalizations 1 &
# 2'.format(resSum)

    return normalizations

cdef cnp.ndarray[uint32_t, ndim = 1] getCorrelationsC(int lenCorr, int totCount, cnp.ndarray[uint32_t, ndim=1] countDwellInds):
    """Return the correlation spectrum as a function of tau, summing over t."""
    cdef cnp.ndarray[uint32_t, ndim = 1] correlations = np.zeros(lenCorr, dtype=np.uint32)
    cdef int i
    cdef int j
    cdef int tau
    cdef float part = (totCount - 1) / 10.
    # for each countTime...
    for i in range(totCount - 1):
        j = i + 1
        tau = countDwellInds[j] - countDwellInds[i]
        # ...starting from the index corresponding tau value 1...
        while tau < 1 and j < totCount - 1:
            j += 1
            tau = countDwellInds[j] - countDwellInds[i]
        # ...iterate through the following countDwellInds and add correlation to the
        # corresponding tau values
        while tau <= lenCorr and j < totCount - 1:
            correlations[tau - 1] += 1  # tau = 1 at index 0
            j += 1
            tau = countDwellInds[j] - countDwellInds[i]
        if i % part < 1:
            print int(i / part),
    print
    return correlations


def getCorrelations(int lag, cnp.ndarray[uint32_t, ndim=1] t, cnp.ndarray[long, ndim=1] F):
    """Return the correlation spectrum as a function of t (not tau), with fixed lag value.
    Definition g(t)."""
    cdef int tot = sum(F)  # total amount of counts
    cdef int N = t[-1] + 1  # total amount of dwell indices
    cdef float FAvg = 1. * tot / N  # average counts per dwell
    cdef float FMS = FAvg * FAvg  # mean square counts
    cdef cnp.ndarray[float_t, ndim = 1] cor = np.zeros(N, dtype=np.float)  # correlations
    cdef int i
    cdef int j = 1
    cdef int k = 0
    cdef int tau
#    cdef float part = (tot-1)/10.
    # for each countTime...
    for i in range(len(t)):
        tau = t[j] - t[i]
        # ...go to the index that is closest to the desired lag value
        while tau < lag and j < len(t) - 1:
            j += 1
            tau = t[j] - t[i]
        # ...and add correlation
        if tau == lag:
            cor[t[i]] = (F[i] - FAvg) * (F[j] - FAvg)
        else:
            cor[t[i]] = (F[i] - FAvg) * (-FAvg)
        # correlate also backwards with time points that have no counts
        if t[i] >= lag:
            tau = t[i] - t[k]
            while tau > lag and k < i:
                k += 1
                tau = t[i] - t[k]
            if tau != lag:
                cor[t[i] - lag] = (-FAvg) * (F[i] - FAvg)
#        if i%part < 1:
#            print int(i/part),
    for i in range(len(cor)):
        if cor[i] == 0:
            cor[i] = FMS
#    print '\ncorrelations done\n'
    return cor


def getAcf(int lenCorr, cnp.ndarray[uint32_t, ndim=1] countDwellInds, int dwellCount):
    cdef int totCount = len(countDwellInds)  # total amount of counts
    cdef cnp.ndarray[uint32_t, ndim = 1] correlations  # values are usually lower than 1000
    cdef cnp.ndarray[float_t, ndim = 1] normalizations
    cdef cnp.ndarray[float_t, ndim = 1] acf = np.zeros(lenCorr, dtype=np.float)
    cdef cnp.ndarray[float_t, ndim = 1] acfSd = np.zeros(lenCorr, dtype=np.float)

    print 'calculating acf'
    # get correlations for each value of tau
    correlations = getCorrelationsC(lenCorr, totCount, countDwellInds)
    print 'correlations done'

    normalizations = getNormalizationsC(
        lenCorr, totCount, countDwellInds, dwellCount)
    print 'normalizations done'

    cdef double a = 1. * dwellCount / totCount
    # average, normalize and convert to numpy array for return
    for i in range(lenCorr):
        acf[i] = 1. / (dwellCount - (i + 1)) * \
            correlations[i] / normalizations[i]
        # shot noise
        acfSd[i] = a / ((dwellCount - (i + 1)) ** 0.5)
        # shot + particle noise
#        acfSd[i] = (1.+q)/((dwellCount - (i + 1))**0.5*totCount/dwellCount)
    print 'acf done\n'

    return acf, acfSd


cdef int * getPhotonCounts(uint8_t highByte):
    """Return an array of integers that tell the count number in each bt in the
    raw high byte (first 8 bits in a raw word)
    """
    # Returns counts for each clock cycle in the word. make sure high != 0
    cdef int * counts = <int * > malloc(lenWord * sizeof(int))
    cdef int channel1
    cdef int channel2
    cdef uint8_t match
    cdef int i
    for i in range(lenWord):
        match = 2 ** (2 * i)
        channel1 = highByte & match != 0
        match = 2 ** (2 * i + 1)
        channel2 = highByte & match != 0
#        counts[i] = channel1 + channel2
        counts[i] = channel2
    return counts

cdef int getPhotonCountsTotal(uint16_t rawWord):
    """Return the number of counts in the raw word (16 bits)."""
    cdef cnp.uint8_t highByte = np.uint8(rawWord >> 8)  # the first 8 bits in a raw word tell the counts
    cdef int counts = 0
    if highByte == 0:
        return counts
    for i in range(lenWord):
        counts += int(highByte & np.uint8(2 ** (2 * i)) != 0) + \
            int(highByte & np.uint8(2 ** (2 * i + 1)) != 0)
    return counts

cdef int getCountSumC(int lenRaw, uint16_t * raw):
    """Returns the number of counts in the raw data."""
    cdef int counts = 0
    for i in range(lenRaw):
        counts += getPhotonCountsTotal(raw[i])
        if i % 100000 == 0:
            print 'counting %.2f' % (1. * i / lenRaw)
    return counts

cdef uint16_t * getCRaw(cnp.ndarray[uint16_t, ndim=1] rawNp):
    """Convert the given raw data from numpy array to c array."""
    cdef uint16_t * raw = <uint16_t * > malloc(len(rawNp) * sizeof(uint16_t))
    # convert rawNp to a c-type array
    for i in range(len(rawNp)):
        raw[i] = rawNp[i]
    return raw


def getCountDwellInds(cnp.ndarray[uint32_t, ndim=1] countTimes, int dwell):
    """Divide the countTimes into dwells starting from minTau. Return: array of
    dwell indices starting from 0."""
#    countDwellInds = []
    cdef cnp.ndarray[uint32_t, ndim = 1] countDwellInds = np.zeros(len(countTimes), dtype=np.uint32)
    cdef int firstDwellInd = int(countTimes[0] / dwell + 1)
    cdef int i
    for i in range(len(countTimes)):
        countDwellInds[i] = int(countTimes[i] / dwell + 1) - firstDwellInd
    return countDwellInds


def getCountDwellInds2(cnp.ndarray[uint32_t, ndim=1] t, int dwell):
    """Divide the countTimes into dwells starting from minTau. Return: array of
    dwell indices starting from 0, array of counts for each index."""
    cdef cnp.ndarray[uint32_t, ndim = 1] dwellInds = np.zeros(len(t), dtype=np.uint32)
    cdef int dwellInd0 = int(t[0] / dwell + 1)
    cdef int i
    for i in range(len(t)):
        dwellInds[i] = int(t[i] / dwell + 1) - dwellInd0
    dwellInds, ind = np.unique(dwellInds, return_inverse=True)
    F = np.bincount(ind)  # F has now the amounts of unique dwellInds
    return dwellInds, F


def getCountTimeIndsFromSeconds(dataStart, dataEnd, btLenUs, cnp.ndarray[uint32_t, ndim=1] countTimesNp):
    """Convert dataStart and dataEnd from seconds to countTime indices."""
    cdef int dataStartBt = dataStart * 1e6 / btLenUs  # time conversion from seconds to clock cycles
    cdef int dataEndBt = dataEnd * 1e6 / btLenUs
    cdef int i = 0
    while i < len(countTimesNp) and countTimesNp[i] < dataStartBt:
        i += 1
    dataStartInd = i
    if dataEnd == -1:
        return dataStartInd, len(countTimesNp) - 1
    while i < len(countTimesNp) and countTimesNp[i] < dataEndBt:
        i += 1
    dataEndInd = i
    return dataStartInd, dataEndInd


def getTotalCount(cnp.ndarray[uint16_t, ndim=1] rawNp):
    """Return the number of counts in the raw data."""
    cdef uint16_t * raw = getCRaw(rawNp)
    cdef int countSum = getCountSumC(len(rawNp), raw)
    return countSum

    # convert to numpy array for return
    countTimesNp = np.zeros(totalCount, dtype=np.uint32)
    for i in range(totalCount):
        countTimesNp[i] = countTimes[i]
    free(countTimes)
    free(raw)

    return countTimesNp, t


def getCountTimes(cnp.ndarray[uint16_t, ndim=1] raw, bool verbose):
    """Convert the raw data file from CC2 16-bit format to an array of photon
    arrival times (count times). Return countTimes (in clock cycles) and the total amount of
    clock cycles in the data.
    """
    cdef int maxCounts = 1000
    cdef cnp.ndarray[uint32_t, ndim = 1] countTimes = np.zeros(maxCounts, dtype=np.uint32)
    cdef uint32_t t = 0
    cdef int countTimeInd = 0
    cdef int * counts
    cdef int totalCount = 0
    cdef uint8_t high
    cdef uint8_t low

    # iterate through all words
    cdef float part = len(raw) / 10.
    cdef int i
    cdef int j
    for i in range(len(raw)):
        high = raw[i] >> 8  # the first 8 bits in a raw word tell the counts
        low = raw[i]  # the last 8 bits in a raw word tell the elapsed cycles
        t += low
        if high != 0:
            counts = getPhotonCounts(high)
            for j in range(lenWord):
                # mark the times at which a count was recorded (counts[j] is either
                # 1 or 2, consecutive countTimes may be the same twice in a row
                if counts[j] != 0:
                    totalCount += 1
                    countTimes[countTimeInd] = t + j
                    countTimeInd += 1
                    if counts[j] == 2:
                        print 'count was 2'
                        totalCount += 1
                        countTimes[countTimeInd] = t + j
                        countTimeInd += 1
            # resize countTimes if needed
            if countTimeInd > maxCounts - 9:
                maxCounts = maxCounts * 2
                countTimes = np.resize(countTimes, maxCounts)
            free(counts)
        t += lenWord - 1
        if verbose and i % part < 1:
            print int(i / part),
    if verbose:
        print '\n'

    countTimes = np.resize(countTimes, totalCount)

    return countTimes, t
