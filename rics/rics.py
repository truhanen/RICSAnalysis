# -*- coding: utf-8 -*-
# Created on Mon Jul  7 16:36:56 2014

import pyximport; pyximport.install()
import ricsx
import os, sys
lib_path = os.path.abspath('/home/tutaruha/documents/work/Opiskelu/Kurssit/FYSZ490 Pro gradu/data/koodi/rics')
sys.path.append(lib_path)
from PIL import Image
import numpy as np
import scipy.odr as odr
import inspect
import cPickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import read_nd2
import scipy.interpolate as inter

g = 1./2./np.sqrt(2) # gamma for Gaussian laser profile

class Measurement:
    def __init__(self, path, ntaus=None, maximg=None, simpath=None,
                 fitmethod='odr', fitdata='cython', fitstart=3, fiterr=True,
                 guess=[1.35,3,87.,.25,-.03], fixed=[1,0,1,0,1], dwell=9.5,
                 pxsize=.02, crop=(0,0,0,0), corrmethod='fft', acf=None,
                 detectormeas=None):
        '''Class for easy rics measurement data (confocal 2D image) processing
        and analysis. Parameters: path: image (.tif or .nd2) file path,
        ntaus: length of the calculated autocorrelation
        estimator (if None, check the getntaus()-function), maximg: maximum
        frames to load from the file, simpath: path of .txt data file exported
        from simfcs.exe, fitmethod: fit method, fitdata: 'cython' or 'simfcs',
        fitstart: index of the first autocorrelation value to be fitted,
        fiterr: use acf standard deviations in the fitting process
        guess: initial fit parameter values, fixed: fixed fit parameters (0 means fixed),
        pxsize and pxtime: pixel size and dwell time used int the measurement,
        crop: margins to be cropped off from the loaded image frames
        (top, bottom, left, right), corrmethod: when calculating correlations,
        use fixed or variable amount of pixel pairs for each tau value, or fft-
        based calculation ('fixed', 'brute force', or 'fft'); fixed is faster but
        less accurate at short tau values; fft corresponds to fixed and it is
        über fast (but not yet implemented), acf: fit model (function object),
        detectormeas: a Measurement instance containging only background
        signal (should have rics.detectoracf as the fit model); if not None,
        the unnormalized detectoracf fit will be substracted from the
        unnormalized ac of this instance during calculation.'''
        self.path = path
        self.ntaus = ntaus
        self.maximg = maximg
        self.simpath = simpath
        self.fitmethod = fitmethod
        self.fitdata = fitdata
        self.fitstart = fitstart
        self.fiterr = fiterr
        self.guess = guess
        self.fixed = fixed
        self.dwell = dwell
        self.pxsize = pxsize
        self.crop = crop
        self.corrmethod = corrmethod
        self.detectormeas = detectormeas
        self._imgs = None
        self._acnorm = None
        self._acsd = None
        self._acysim = None
        self._fitparams = None
        self._fiterrors = None
        self._fitcov = None
        self._fitteddata = None
        if acf == None:
            self.defacf()
        else:
            self.acf = acf
        

    # for pickling
    def __getstate__(self):
        d = self.__dict__.copy()
        del d['acf']
        del d['_imgs'] # images are saved to .imgs file in getimages()
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._imgs = None
        self.defacf()

    def save(self, path=None):
        if path == None:
            path = self.getsavepath()
        f = open(path, 'wb')
        cPickle.dump(self, f, 2)
        f.close()
        print 'Measurement saved to', path
    
    def getsavepath(self):
        return self.path + '.meas'
    
    @classmethod
    def load(cls, path):
        meas = cPickle.load(open(path, 'rb'))
        print 'Measurement loaded from', path
        return meas
    
    def defacf(self):
        def acf(p, x):
            '''Theoretical line scan ACF for one-component 3D diffusion with
            g(0) (<n>)                    p[0]
            Structure parameter           p[1]
            Diffusion coefficient         p[2]
            Beam waist                    p[3]
            Background signal             p[4]
            g = g(0) / (1+4*D*x*tp/w**2) / sqrt(1+4*D*x*tp/w**2/k**2)
                     * exp( -(r*x/w)**2 / (1+4*D*x*tp/w**2) )
                     + bg
            '''
            return p[0]/(1.+4.*1.e-6*p[2]*x*self.dwell/p[3]**2)\
                       /(1.+4.*1.e-6*p[2]*x*self.dwell/p[3]**2/p[1]**2)**.5\
                       *np.exp(-(self.pxsize*x/p[3])**2/(1.+4.*1.e-6*p[2]*x*self.dwell/p[3]**2))\
                       +p[4]
            # without background
#            return p[0]/(1.+4.*1.e-6*p[2]*x*self.dwell/p[3]**2)\
#                       /(1.+4.*1.e-6*p[2]*x*self.dwell/p[3]**2/p[1]**2)**.5\
#                       *np.exp(-(self.pxsize*x/p[3])**2/(1.+4.*1.e-6*p[2]*x*self.dwell/p[3]**2))
        self.acf = acf
    
    def getac(self):
        '''Return the average normalized ac of the cropped image sequence,
        (tau, acf).'''
        return range(self.getntaus()), np.mean(self.getacnorm(), axis=0)
    
    def getacnorm(self):
        '''Return normalized ac values of each frame.'''
        if self._acnorm == None:
            if self.detectormeas != None: # desperate try of detector correlation correction
                _, detectoracunnorm = self.detectormeas.getfitunnorm(x=np.arange(self.getntaus()))
                acunnorm = [acunnormi - detectoracunnorm for acunnormi in self.getacunnorm()]
                norm = self.getnorm()
                self._acnorm = [normed - 1 for normed in acunnorm/norm]
            else:
                self._acnorm = ricsx.getacnorm(self.getntaus(), self.corrmethod,
                                               self.getimagescropped())
        return self._acnorm
    
    def getacunnorm(self):
        '''Return unnormalized acfs of each frame.'''
        return ricsx.getacunnorm(self.getntaus(), self.corrmethod,
                                 self.getimagescropped())
    
    def getnorm(self):
        '''Return correlation normalization of each frame.'''
        return ricsx.getnorm(self.getntaus(), self.corrmethod,
                             self.getimagescropped())
    
    def getacsd(self):
        '''Return acf standard deviations of each tau value, calculated from
        the acfs of different frames.'''
        if self._acsd == None:
            self._acsd = np.std(self.getacnorm(), axis=0)
        return self._acsd
    
    def getacsim(self):
        if self._acysim == None:
            self._acysim = loadacsim(self.simpath)
        return (range(len(self._acysim)), self._acysim)
    
    def getntaus(self):
        if self.ntaus == None:
            self.ntaus = len(self.getimagescropped()[0][0])/2
        return self.ntaus
    
    def getimages(self):
        '''Return images as numpy array.'''
        if self._imgs == None:
            imgspath = self.path + '.imgs'
            if os.path.isfile(imgspath):
                self._imgs = load(imgspath)
                print len(self._imgs), 'frames loaded from', imgspath
            elif self.path.endswith('.tif'):
                self._imgs = loadtif(self.path, self.maximg)
            elif self.path.endswith('.nd2'):
                self._imgs = loadnd2(self.path, self.maximg)
            if not os.path.isfile(imgspath):
                save(imgspath, self._imgs)
                print 'images saved to', imgspath
        return self._imgs[:self.maximg]
    
    def getimagescropped(self):
        imgs = self.getimages()
        cropped = []
        for img in imgs:
            cropped.append(img[self.crop[0]: len(img)-self.crop[1],
                               self.crop[2]: len(img[0])-self.crop[3]])
        return np.array(cropped)
    
    def getspline(self):
        '''Return a spline fit as a one-argument function object.'''
        acx, acy, acsd = self.getfitdata()
        return inter.UnivariateSpline(acx, acy, w=acsd)
    
    def getfitdata(self):
        '''Return data to be fitted to the model.'''
        acsd = None
        if self.fitdata == 'cython':
            acx, acy = self.getac()
            acsd = self.getacsd()
        elif self.fitdata == 'simfcs':
            acx, acy = self.getacsim()
        if acsd == None:
            return acx[self.fitstart:], acy[self.fitstart:], None
        return acx[self.fitstart:], acy[self.fitstart:], acsd[self.fitstart:]
    
    def getfit(self, x=None):
        '''Return values of the fitted model, (x,y). Parameters: x: list of x
        values at which to calculate the acf.'''
        params,_ = self.getfitparams()
        if x == None:
            acx,_,_ = self.getfitdata()
            x = np.arange(acx[0], acx[-1], 1)
        y = self.acf(params, x)
        return x, y
    
    def getfitres(self):
        '''Return fit residuals, (x,y).'''
        params, errors = self.getfitparams()
        acx, acy, _ = self.getfitdata()
        resy = np.array([acy[xi] - self.acf(params, acx[xi]) for xi in range(len(acx))])
        return acx, resy
    
    def getfitunnorm(self, x=None):
        '''Fit to normalized data and multiply by norm to get unnormalized fit.'''
        assert len(x) <= self.getntaus(), 'x (length ' + str(len(x)) \
                                          + ') is longer than ntaus (' \
                                          + str(self.getntaus()) + ')'
        _, fity = self.getfit(x)
        return x, fity*np.mean(self.getnorm(), axis=0)[:len(x)]

    def getfitparams(self):
        '''Calculate and fit to the autocorrelation estimator. Parameters:
         Parameters: data: data to fit ('cython' or 'simfcs'). Returns: fitparams:
        fitted parameter values, fiterrors: standard errors of the fitted parameters'''
        if self._fitparams == None or self._fiterrors == None or self._fitteddata != self.fitdata:
            acx, acy, acsd = self.getfitdata()
            # assume equal sds (weights)
            if self.fiterr == False or acsd == None:
                acsd = len(acy)*[1.]
            fitxsd = len(acx)*[1e-99]
            data = odr.RealData(x=acx, y=acy, sx=fitxsd, sy=acsd)
            model = odr.Model(self.acf)
            fit = odr.ODR(data, model, self.guess, ifixb=self.fixed, maxit=1000, job=10)    
            output = fit.run()
            self._fitparams = output.beta     # 'beta' is an array of the parameter estimates
            self._fitcov = output.cov_beta   # parameter covariance matrix
            self._fiterrors = output.sd_beta # parameter standard uncertainties
            if False:
                print "Model:"
                print "  " + inspect.getsource(self.acf)
            if self.guess != None:
                print "Estimated parameters and uncertainties"
                for i in range(len(self._fitparams)) :
                    print ("   p[{0}] = {1:10.5g} +/- {2:10.5g}"+
                           "          (Starting guess: {3:10.5g})").\
                            format(i,self._fitparams[i],self._fiterrors[i],self.guess[i])
            if False:
                print "\nCorrelation Matrix :",
                for i,row in enumerate(self._fitcov):
                    print
                    for j in range(len(self._fitparams)) :
                        print "{0:< 8.3g}".format(self._fitcov[i,j]/np.sqrt(self._fitcov[i,i]*self._fitcov[j,j])),
            print
        self._fitteddata = self.fitdata
        return self._fitparams, self._fiterrors
    
    def plot(self, cors=True, simfcs=True, cython=True, sd=True, fit=True, spline=False,
             imgmean=True, figsize=None, detector=False):
        fig = plt.figure(figsize=figsize)
        title = self.path[self.path.rfind('/')+1:]
        fig.suptitle(title)
        if cors:
            if imgmean:
                plt.subplot2grid((5,10), (0,0), rowspan=5, colspan=5)
            # load simfcs export
            if simfcs and self.simpath != None:
                acxsim, acysim = self.getacsim()
                plt.plot(acxsim, acysim, 's', label='simfcs')
#                plt.plot(acxsim, 1.33*acysim+.0008, 's', label='simfcs (fft)')
            if cython:
                acx, acy = self.getac()
                plt.plot(acx[1:], acy[1:], 's', label=self.corrmethod)
                if sd:
                    acsd = self.getacsd()
                    plt.plot(acx[1:], acsd[1:], label='acf std')
            # fit
            if fit:
                fitx, fity = self.getfit()
                plt.plot(fitx, fity, label='odr fit')
                fitresx, fitresy = self.getfitres()
                plt.plot(fitresx, fitresy, label='odr fit res')
            if spline:
                spline = self.getspline()
                x,_,_ = self.getfitdata()
                plt.plot(x, spline(x), label='spline fit')
            if detector and self.detectormeas is not None:
                x, y = self.detectormeas.getac()
                fitx, fity = self.detectormeas.getfit()
                plt.plot(x[1:], y[1:], 's', label='detector ac')
                plt.plot(fitx, fity, label='detector fit')
            plt.grid()
#            plt.title(title)
            plt.xlabel(r'\xi')
            plt.ylabel(r'G(\xi)')
            plt.legend()
        # plot image average etc.
        if imgmean:
            imgs = self.getimages()
            imgmean = np.mean(imgs, axis=0)
            ymean = np.mean(imgmean, axis=0)
            xmean = np.mean(imgmean, axis=1)
            if cors:
                plt.subplot2grid((5,10), (0,5), rowspan=4, colspan=4)
            else:
                plt.subplot2grid((5,5), (0,0), rowspan=4, colspan=4)
#                plt.title(title)
            plt.imshow(imgmean, interpolation='none', aspect='auto')
            x = self.crop[2]-.5
            y = self.crop[0]-.5
            w = len(imgmean[0])-self.crop[2]-self.crop[3]
            h = len(imgmean)-self.crop[0]-self.crop[1]
            croparea = Rectangle((x, y), w, h, ec='red', fc='none')
            plt.gca().add_patch(croparea)
            if cors:
                plt.subplot2grid((5,10), (4,5), rowspan=1, colspan=4)
            else:
                plt.subplot2grid((5,5), (4,0), rowspan=1, colspan=4)
            plt.plot(ymean)
            plt.xlim(0, len(ymean))
            if cors:
                plt.subplot2grid((5,10), (0,9), rowspan=4, colspan=1)
            else:
                plt.subplot2grid((5,5), (0,4), rowspan=4, colspan=1)
            plt.plot(xmean, range(len(xmean),0,-1))
            plt.ylim(0, len(xmean))

def detectoracf(p, x):
    return p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 + p[4]*np.sin(p[5]*x+p[6])*x**p[7]

def detectoracf2(p, x):
    return p[0] + p[1]*x + p[2]*x**2

def save(path, data):
    f = open(path, 'wb')
    cPickle.dump(data, f, 2)
    f.close()

def load(path):
    return cPickle.load(open(path, 'rb'))

def loadacsim(path):
    '''Load simfcs ac file (remember to remove "Data" string from the first line).'''
    print path
    acsim = np.loadtxt(path, unpack=True)
    # take only first column, if there are many
    if len(acsim.shape) > 1:
        acsim = acsim[0]
    return acsim

def loadtif(path, maximg):
    img = Image.open(path)
    imgs = []
    print 'loading frames',
    for i in xrange(1000000):
        if i == maximg:
            break
        try:
            img.seek(i)
            imgs.append(np.array(img.copy()))
            if i%10 == 0:
                print i,
        except EOFError:
            break
    print
    return np.array(imgs, dtype=np.uint16)

def loadnd2(path, maximg):
    f = open(path, 'r')
    nd2 = read_nd2.Nd2File(f)
    imgs = []
    print 'loading frames',
    for i in xrange(1000000):
        if i == maximg:
            break
        try:
            res = nd2.get_image(i)
            if i%10 == 0:
                print i,
        except:
            break
        img = np.array(res[1], dtype=np.uint16)
        # from 1D to 2D matrix; assume aspect ratio 1:1
        size = np.sqrt(len(img))
        img = np.resize(img, (size, size))
        imgs.append(img)
    print
    f.close()
    return np.array(imgs)

















