# -*- coding: utf-8 -*-
# Created on Thu Mar 13 12:13:56 2014

import pyximport; pyximport.install()
import CC2UtilsX
import TheorAcfFuncs as funcs
import numpy as np
import scipy.odr as odr
import os
import inspect
import cPickle

def save(path, data):
    f = open(path, 'wb')
    cPickle.dump(data, f, 2)
    f.close()

def load(path):
    return cPickle.load(open(path, 'rb'))

btinus = .05

class OwnResults:
    def __init__(self, path, cc2results=None, counttimes=None, mintau=.05, maxtau=1.e3,
                 dwell=4, datastart=0, dataend=-1):
        '''Parameters: path: directory containing raw and .txt files (data and table)
        datastart and dataend: in seconds'''
        self.path = path
        self.cc2results = cc2results
        self.counttimes = counttimes
        self.mintau = mintau
        self.maxtau = maxtau
        self.dwell = dwell
        self.datastart = datastart
        self.dataend = dataend
        self.raws = None
        self.rawpaths = None
        self.fits = None
        self.acfs = None
        if cc2results == None:
            self.cc2results = CC2Results(self.path)
    
    def setstartend(self, datastart, dataend):
        self.datastart = datastart
        self.dataend = dataend
    
    def getresultcount(self):
        return len(self.getrawpaths())

    def getraw(self, i):
        '''Return raw data as binary numpy array.'''
        if self.raws == None:
            self.raws = [None]*len(self.getrawpaths())
        if self.raws[i] == None:
            with open(self.getrawpaths()[i], "rb") as f:
                self.raws[i] = np.fromfile(f, '<u2')[30:] # omit ConfoCor 2 text
        return self.raws[i]
    
    def getrawpaths(self):
        '''Return raw file pathnames.'''
        if self.rawpaths == None:
            self.rawpaths = []
            for filename in sorted(os.listdir(self.path)):
                if filename.startswith('FcsRaw.') and not filename.endswith(('.npy', '.pickle')):
                    self.rawpaths.append(self.path + filename)
        return self.rawpaths
    
    def getparams(self, i, loadacf=False, saveacf=False, acfsuffix=None,
                  func=funcs.triplet, fixed={}, fitstart=0, loadfit=False,
                  savefit=False, fitsuffix='fit.pickle', verbose=True):
        ''' Return fitted parameters.
        Parameters: fitstart: in microseconds'''
        if self.fits == None:
            self.fits = [None]*self.getresultcount()
        if self.fits[i] != None and self.fits[i].acf.datastart == self.datastart \
           and self.fits[i].acf.dataend == self.dataend:
            return self.fits[i].getparams(loadfit=loadfit, savefit=savefit, verbose=verbose)
        if self.acfs == None or self.acfs[i] == None or self.acfs[i].datastart != self.datastart \
           or self.acfs[i].dataend != self.dataend:
            self.getacf(i, loadacf=loadacf, saveacf=saveacf, acfsuffix=acfsuffix,
                        verbose=verbose)
        guess = self.cc2results.getparams()[i]
        self.fits[i] = Fit(self.getrawpaths()[i]+fitsuffix, self.acfs[i], func, guess, fixed, fitstart)
        return self.fits[i].getparams(loadfit=loadfit, savefit=savefit, verbose=verbose)
    
    def getacf(self, i, loadacf=False, saveacf=False, acfsuffix=None,
               verbose=True):
        """Load or calculate and save acf. Definition G = g + 1.
    
        Parameters: minTau & maxTau in us, dwell in clock cycles (0.05 us), dataStart & dataEnd in seconds.
        Filenameend: will be used as the saved or loaded filename suffix; if None,
        'acf.npy' will be used."""
        if self.acfs == None:
            self.acfs = [None]*self.getresultcount()
        if self.acfs[i] != None and self.acfs[i].datastart == self.datastart \
           and self.acfs[i].dataend == self.dataend:
            return self.acfs[i]
        if acfsuffix == None:
            acfpath = self.getrawpaths()[i] + 'acf.pickle'
        else:
            acfpath = self.getrawpaths()[i] + acfsuffix
        if loadacf and os.path.isfile(acfpath): # load old acf
            acf = load(acfpath)
            same = np.allclose(self.mintau, acf.mintau) and np.allclose(self.maxtau, acf.maxtau) \
                    and np.allclose(self.datastart, acf.datastart) and np.allclose(self.dataend, acf.dataend) \
                    and self.dwell == acf.dwell
            if not same and verbose:
                print 'acf with the desired parameters was not found\n'
            if verbose:
                print 'acf loaded with parameters\n' + acf.tostring()
            acf.fromfile = True
        else: # compute new acf
            counttimes = self.getcounttimes(i, verbose)
            cycleCount = counttimes[-1] - counttimes[0]
            # downsample data to dwells
            countDwellInds = CC2UtilsX.getCountDwellInds(counttimes, self.dwell)
            dwellCount = cycleCount/self.dwell
            # convert minTau and maxTau from microseconds to dwell indices
            minTauDwell = max(int(self.mintau/(btinus*self.dwell)), 1)
            maxTauDwell = max(int(self.maxtau/(btinus*self.dwell)), 1)
            # compute acf
            ac, ac_sd = CC2UtilsX.getAcf(maxTauDwell, countDwellInds, dwellCount)[minTauDwell-1:]
            lags = list(btinus*self.dwell*(minTauDwell + x) for x in range(len(ac)))
            acf = Acf(lags, ac, True, ac_sd=ac_sd)
            acf.mintau = self.mintau; acf.maxtau = self.maxtau; acf.datastart = self.datastart
            acf.dataend = self.dataend; acf.dwell = self.dwell; acf.fromfile = False
            # save and return the computed acf etc.
            if saveacf:
                save(acfpath, acf)
                if verbose:
                    print 'acf saved with parameters\n' + acf.tostring()
        self.acfs[i] = acf
        return acf
    
    def getcounttimes(self, i, verbose):
        '''Calculate count times from raw data. Return: Array of count times in clock cycles.'''
        if self.counttimes == None:
            self.counttimes = [None]*self.getresultcount()
        if self.counttimes[i] == None:
            self.counttimes[i], cyclecount = CC2UtilsX.getCountTimes(self.getraw(i),
                                                                     verbose)
        datastartind, dataendind = CC2UtilsX.getCountTimeIndsFromSeconds(self.datastart,
                                                                         self.dataend,
                                                                         btinus,
                                                                         self.counttimes[i])
        # clip countTimes from the desired points
        return self.counttimes[i][datastartind:dataendind]
    
    def getcountrate(self, i, verbose=False):
        '''Return: average count rate in kHz.'''
        counttimes = self.getcounttimes(i, verbose)
        # dataset length in seconds
        length = (counttimes[-1] - counttimes[0]) * btinus / 10**6
        cr = len(counttimes) / length / 10**3
        return cr
        
    def getsd(self, i, loadacf=False, saveacf=False, acfsuffix=None,
              func=funcs.triplet, div=100, verbose=True):
        """Return (t, sd), where t is the list of lag times and sd is the list
        of standard deviations of the autocorrelation function."""
        acf = self.getacf(i, loadacf=loadacf, saveacf=saveacf, acfsuffix=acfsuffix,
                    verbose=verbose)
        fitp = self.getparams(i).getfitvalues(func)
        lags = acf.lags
        ac = acf.ac
        step = len(lags)/div
        t = []
        sd = []
        for i in range(div):
            t.append(lags[i*step])
            sd.append(0)
            # SD = sqrt( mean( (y - mean(y))**2 ) )
            for j in range(i*step, (i+1)*step):
                sd[i] += (ac[j] - 1 - func(fitp, lags[j]))**2
            sd[i] = (1.*sd[i]/step)**.5
        return t, sd
    
    def getsignal(self, i, func=funcs.triplet, loadacf=False, saveacf=False, acfsuffix=None):
        '''Calculate acf signal. Return (signal, dsignal).'''
        params = self.getparams(i, func=func)
        T = params.values['Triplet Fraction']
        dT = params.errors['Triplet Fraction']
        n = params.values['Number Particles']
        dn = params.errors['Number Particles']
        signal = 1./(1-T)/n
        dsignal = np.sqrt((1./(1-T)**2/n*dT)**2 + (1/(1-T)/n**2*dn)**2)
        return signal, dsignal
        
    def getsignaltonoiseratio(self, i, func=funcs.triplet, loadacf=False, saveacf=False, acfsuffix=None):
        '''Calculate signal to noise ratio.'''
        sd = self.getsd(i, func=func, loadacf=loadacf, saveacf=saveacf, acfsuffix=acfsuffix)[1][0]
        signal = self.getsignal(i, func=func, loadacf=loadacf, saveacf=saveacf, acfsuffix=acfsuffix)[0]
        return signal/sd
    
    def getbias(self, i, loadacf=False, saveacf=False, acfsuffix=None,
                func=funcs.triplet, fixed=[], fitstart=0, loadfit=False, savefit=False,
                fitsuffix='fit.pickle', verbose=True):
        counttimes = self.getcounttimes(i, verbose=verbose)
        counttimes, F = CC2UtilsX.getCountDwellInds2(counttimes, self.dwell)
#        params = self.getparams(i, loadacf=loadacf, saveacf=saveacf, acfsuffix=acfsuffix,
#                func=func, fixed=fixed, fitstart=fitstart, loadfit=loadfit, savefit=savefit,
#                fitsuffix=fitsuffix, verbose=verbose)
#        k = params.values['Structure Parameter']
#        tau_D = params.values['Diffusion Time 1']
#        n = params.values['Number Particles']
#        T = params.values['Triplet Fraction']
#        print 'T', T
        # power: 15 %
#        tau_D = 29.3 # short measurements -> bad fits -> set parameters from fits to the whole data
#        k = 5. # short measurements -> bad fits -> set parameters from fits to the whole data
#        n = .756
        # power: 2 %
        tau_D = 24.38 # short measurements -> bad fits -> set parameters from fits to the whole data
        k = 5. # short measurements -> bad fits -> set parameters from fits to the whole data
        n = .527
        dwell_us = self.dwell*btinus
        N = counttimes[-1]
        v = self.maxtau / dwell_us
        F_avg = 1.*np.sum(F) / counttimes[-1]
        m = F_avg / n
        print 'counttimes[-1]', counttimes[-1]
        print 'counttimes[0]', counttimes[0]
        print 'k', k
        print 'tD', tau_D
        print 'n', n
        print 'dwell_us', dwell_us
        print 'N', N
        print 'v', v
        print '<F>', F_avg
        print 'm', m
        bias = self.bias(N, v, k, tau_D, dwell_us, F_avg, m)
        return bias
    
    def bias(self, N, v, k, tau_D, dwell_us, F_avg, m):
        '''See Saffarian 2003'''
        gamma = 2**(-3./2)
        tcr = tau_D/dwell_us
        print 'tcr', tcr
        Nf = N - v
        b = k**-2
        A = ( 1 + Nf * b / tcr)**.5 - 1
        F = ( 1 - b )**.5
        B = np.tanh( F * A / (b + A) )
        C = b * ( 1 + Nf / tcr )
        E = - F * A
        D = 4 * tcr**2 / ( b * F )
        
        return (Nf + gamma * m * D * ( C * B + E ) ) / ( Nf**2 * F_avg )


class CC2Results:
    def __init__(self, path):
        '''Parameters: path: directory containing .fcs and .txt files (data and table)'''
        self.path = path
        self.acfs = None
        self.crs = None
    
    def getacf(self, i):
        '''Return acf instance.'''
        if self.acfs == None:
            self.getacfs()
        return self.acfs[i]
    
    def getacfs(self):
        if self.acfs != None:
            return self.acfs
        self.acfs = []
        xFull_yFull = [] # list of tuples of x and y coordinate lists
        paths = filter(os.path.isfile, [os.path.join(self.path, s) for s in os.listdir(self.path)])
        try:
            for p in paths:
                if p[-4:] == '.fcs':
                    xFull_yFull.append(np.genfromtxt(p, delimiter=', ', skip_header=16,
                                             skip_footer=1, unpack=True))
        except IOError:
            return None
        # the file contains both the acf and the count rate values
        # first x value of count rate is 0.0
        for xFull, yFull in xFull_yFull:
            countRateStart = np.where([x == 0.0 for x in xFull])[0][1]
            lags = xFull[:countRateStart]*1e3 # from milliseconds to microseconds
            ac = yFull[:countRateStart]
            self.acfs.append(Acf(lags, ac, True))
        return self.acfs
    
    def getcr(self, i):
        '''Return acf instance.'''
        if self.crs == None:
            self.getcrs()
        return self.crs[i]
    
    def getcrs(self):
        if self.crs != None:
            return self.crs
        self.crs = []
        xFull_yFull = [] # list of tuples of x and y coordinate lists
        paths = filter(os.path.isfile, [os.path.join(self.path, s) for s in os.listdir(self.path)])
        try:
            for p in paths:
                if p[-4:] == '.fcs':
                    xFull_yFull.append(np.genfromtxt(p, delimiter=', ', skip_header=16,
                                             skip_footer=1, unpack=True))
        except IOError:
            return None
        # the file contains both the acf and the count rate values
        # first x value of count rate is 0.0
        for xFull, yFull in xFull_yFull:
            countRateStart = np.where([x == 0.0 for x in xFull])[0][1]
            time = xFull[countRateStart:]
            cr = yFull[countRateStart:]
            self.crs.append((time, cr))
        return self.crs
    
    def getparams(self):
        '''Returns:
            List of Parameters-instances.
        '''
        tablepath = self.path + '.txt'
        try:
            rawparams = np.genfromtxt(tablepath, skip_header=1, delimiter='\t', dtype=None)
        except IOError:
            print 'did not found .txt-file from', tablepath
            return None
        loadednames = rawparams[0]
        params = []
        for i in range(len(rawparams)-1):
            params.append(Parameters())
        for i, name in enumerate(loadednames):
            if name.rfind('[') > -1:
                name = name[:name.rfind('[')-1] # omit units, e.g. "[ us ]" (greek letters are dull)
            if name in params[0].names:
                for j in range(1, len(rawparams)):
                    try:
                        value = float(rawparams[j][i])
                        if name == 'Triplet Fraction':
                            value *= .01
                    except ValueError:
                        value = rawparams[j][i]
                    params[j-1].values[name] = value
        return params
    
    def getfitparams(self, i, func=funcs.triplet, allparams=None):
        if allparams == None:
            allparams = self.getparams()
        allparams[i].func = func
        return allparams[i].getfitvalues()
        

class Acf:
    '''Autocorrelation function and metadata.'''
    def __init__(self, lags, ac, G, ac_sd=None):
        '''Parameters:
            lags: lag time values (microseconds)
            ac: autocorrelation values
            ac_sd: autocorrelation standard deviations
            G: True if ac values correspond to definition G = g + 1'''
        self.lags = lags
        self.ac = ac
        self.ac_sd = ac_sd
        self.G = G
        self.mintau = None
        self.maxtau = None
        self.datastart = None
        self.dataend = None
        self.dwell = None
        self.fromfile = None
    
    def ustoind(self, us):
        return int(us/(btinus*self.dwell))
    
    def tostring(self):
        s = '{0:>10} min tau (us)\n{1:>10} max tau (us)\n{2:>10} dwell (clock cycles (0.05 us))\n{3:>10} data start (s)\n{4:>10} data stop (s)\n'
        return s.format(self.mintau, self.maxtau, self.dwell, self.datastart, self.dataend)

class Fit:
    def __init__(self, path, acf, func, guess, fixed, fitstart):
        '''Parameters:
        guess: Parameters instance
        fixed: parameter names that should be fixed in the fit
        fitstart: in microseconds
        '''
        self.path = path
        self.acf = acf
        self.func = func
        self.guess = guess
        self.fixed = fixed
        self.fitstart = self.acf.ustoind(fitstart)
        self.output = None
        self.params = None
    
    def getparams(self, loadfit=False, savefit=False, verbose=True):
        '''Return fitted values as a Parameters instance.'''
        if self.params != None:
            return self.params
        if self.output == None:
            self.fitacf(loadfit=loadfit, savefit=savefit, verbose=verbose)
        self.params = self.guess + Parameters.fromvalues(self.func, self.output.beta, self.output.sd_beta)
        return self.params
    
    def fitacf(self, loadfit=False, savefit=False, fitsuffix='fit.pickle', verbose=True):
        if loadfit and os.path.isfile(self.path):
            self.output = load(self.path)
            if verbose:
                print 'Fit loaded from', self.path, '\n'
        else:
            ac = self.acf.ac
            if self.acf.G:
                ac = [y-1 for y in self.acf.ac]
            lags_sd = len(self.acf.lags)*[1e-99]
            # if ySd is not provided, assume equal weigths
            ac_sd = self.acf.ac_sd
            if ac_sd == None:
                ac_sd = len(self.acf.ac)*[1]
            data = odr.RealData(x=self.acf.lags[self.fitstart:], y=ac[self.fitstart:], sx=lags_sd[self.fitstart:], sy=ac_sd[self.fitstart:]) # sx=len(acfx)*[1e-99], sy=np.sqrt(yWeight)**-1)# sy=len(acfy)*[1e-99])
            model = odr.Model(self.func)
            fit = odr.ODR(data, model, self.guess.getfitvalues(self.func), ifixb=self.getfixed(), maxit=5000, job=10)    
            self.output = fit.run()
            # save pickle
            if savefit and self.path != None:
                save(self.path, self.output)
                if verbose:
                    print 'Fit saved to', self.path, '\n'
        if verbose:
            self.printSummary(self.output)
        return self.output
    
    def getfixed(self):
        fitnames = Parameters.getfitnames(self.func)
        fixed = [1]*len(fitnames)
        for name in self.fixed:
            if name in fitnames:
                fixed[fitnames.index(name)] = 0
        return fixed

    def printSummary(self, output):
        p = output.beta 	# 'beta' is an array of the parameter estimates
        cov = output.cov_beta   # parameter covariance matrix
        uncertainty = output.sd_beta # parameter standard uncertainties
        guess = self.guess.getfitvalues(self.func)
        if len(output.stopreason) > 0:
            print "ODR algorithm stop reason: " + output.stopreason[0]
        if self.acf.lags != None:
            print "\nFit {0} Data points from file: {1}".format(len(self.acf.lags),"[filename]")
        if self.func != None:
            print "To Model :"
            print "  " + inspect.getsource(self.func)
        if self.guess != None:
            print "Estimated parameters and uncertainties"
            for i in range(len(p)) :
                print ("   p[{0}] = {1:10.5g} +/- {2:10.5g}"+
                       "          (Starting guess: {3:10.5g})").\
                        format(i,p[i],uncertainty[i],guess[i])
        print "\nCorrelation Matrix :",
        for i,row in enumerate(cov):
            print
            for j in range(len(p)) :
                print "{0:< 8.3g}".format(cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])),
        print '\n'

class Parameters:
    names = ('Number Particles', 'Structure Parameter', 'Diffusion Time 1',
    'Diffusion Time 2', 'Diffusion Time 3', 'Particle Fraction 1',
    'Triplet Fraction', 'Triplet Time',
    'Chamber / Position', 'Counts per Molecule', 'Count Rate', 'Correlation',
    'Bleach Rate')
    
    def __init__(self, values=None, errors=None):
        self.values = {name: None for name in Parameters.names}
        self.errors = {name: None for name in Parameters.names}
        if values != None:
            for name in values.keys():
                self.values[name] = values[name]
        if errors != None:
            for name in errors.keys():
                self.errors[name] = errors[name]
    
    def __add__(self, p2):
        '''Create new Parameters instance, where every value other than None is that of p2.'''
        p3 = Parameters(self.values, self.errors)
        for name in self.names:
            if p2.values[name] != None:
                p3.values[name] = p2.values[name]
            if p2.errors[name] != None:
                p3.errors[name] = p2.errors[name]
        return p3

    def getfitvalues(self, func):
        fitvalues = []
        for name in self.getfitnames(func):
            value = self.values[name]
            if value != None:
                fitvalues.append(value)
            elif name.startswith('Diffusion Time'): # if we need more diffusion times
                fitvalues.append(self.values['Diffusion Time 1'])
            elif name.startswith('Particle Fraction'): # if we need more particle fractions
                fitvalues.append(.5)
            elif name == 'Bleach Rate': # Cc2results does not give bleach rate
                fitvalues.append(0)
        return fitvalues
    
    def getfiterrors(self, func):
        return list(self.errors[name] for name in self.getfitnames(func))
    
    @classmethod
    def fromvalues(cls, func, values, errors=None):
        '''Return a Parameters instance generated from values (iterable).'''
        names = Parameters.getfitnames(func)
        values = {name: values[i] for i, name in enumerate(names)}
        if errors != None:
            errors = {name: errors[i] for i, name in enumerate(names)}
        return cls(values, errors)

    @classmethod
    def getfitnames(cls, func):
        return {
                funcs.diffusion3D: Parameters.names[0:3],
                funcs.multiComponent2: Parameters.names[0:4] + (Parameters.names[5],),
                funcs.multiComponent3: Parameters.names[0:5],
                funcs.bleach: Parameters.names[0:3] + (Parameters.names[5],),
                funcs.triplet: Parameters.names[0:3] + Parameters.names[6:8],
                funcs.tripletbleach: Parameters.names[0:3] + Parameters.names[6:8] + (Parameters.names[12],),
                funcs.tripletmulti2: Parameters.names[0:4] + Parameters.names[5:8]
                }.get(func)
        
        
        
        
















