import matplotlib.pyplot as plt
import numpy as np
from math import log10, floor, sqrt
import scipy as sp
from scipy import stats

#########################################################################
# funzione per arrotondare con un certo numero di cifre significative
#########################################################################
def PrintResult(name,mean,sigma,digits,unit):
    mean = round(mean,digits)
    sigma = round(sigma,digits)
    nu = sigma / mean
    result = (name+" = ({0} +/- {1} ) ".format(mean,sigma)+unit+" [{0:.2f}%]".format(nu*100))
    print (result)
    #return ""
    
##########################################################################
# funzioni nuove
##########################################################################
    
def my_round(x):
    i=0
    while x*pow(10,i)<10:
        i+=1 
        if x==0:
            return 3
    return i

def linear_chi(x,y,sigy,m,c):
    chiq=np.sum(((y-m*x-c)/sigy)**2)
    ni=x.size-2
    return chiq,ni

def change(ris):
    lista=list(ris)
    
    if len(lista)==1:
        if lista[0]=='g':
            new='k'+''.join(lista)
            return new,1/1000
        else:
            new=''.join(lista)
            return new,1
    elif lista[1]=='g':
        new=''.join(lista)
        return new,1
    
    elif lista[0]=='m':
        lista.remove(lista[0])
        new=''.join(lista)
        return new,1/1000
    
    elif lista[0]=='c':
        lista.remove(lista[0])
        new=''.join(lista)
        return new,1/100
    
    elif lista[0]=='d':
        lista.remove(lista[0])
        new=''.join(lista)
        return new,1/10
    
    elif lista[0]=='k':
        lista.remove(lista[0])
        new=''.join(lista)
        return new,1000
def istogreve(data,binfac,titolox,titoloy,titolo):
    binsize = np.std(data,ddof=1)*binfac # half of standard deviation
    interval = data.max() - data.min()
    nbins = int(interval / binsize)
    
    counts , bins , patches = plt.hist(data,bins=nbins,fill=False)
    plt.xlabel(titolox)
    plt.ylabel(titoloy)
    plt.title(titolo)
    plt.show
    
#########################################################################
# funzioni per fare il fit lineare
#########################################################################
#
def my_mean(x, w):
    return np.sum( x*w ) / np.sum( w )

def my_cov(x, y, w):
    return my_mean(x*y, w) - my_mean(x, w)*my_mean(y, w)

def my_var(x, w):
    return my_cov(x, x, w)

def my_line(x, m=1, c=0):
    return m*x + c

def y_estrapolato(x, m, c, sigma_m, sigma_c, cov_mc):
    y = m*x + c
    uy = np.sqrt(np.power(x, 2)*np.power(sigma_m, 2) +
                   np.power(sigma_c, 2) + 2*x*cov_mc ) 
    return y, uy

def lin_fit(x, y, sd_y, xlabel="x [ux]", ylabel="y [uy]", xm=0., xM=1., ym=0., yM=1., 
            verbose=True, plot=False, setrange=False, plus=False):

    #pesi
    w_y = np.power(sd_y.astype(float), -2) 
    
    #m
    m = my_cov(x, y, w_y) / my_var(x, w_y)
    var_m = 1 / ( my_var(x, w_y) * np.sum(w_y) )
    
    #c
    c = my_mean(y, w_y) - my_mean(x, w_y) * m
    var_c = my_mean(x*x, w_y)  / ( my_var(x, w_y) * np.sum(w_y) )
    
    #cov
    cov_mc = - my_mean(x, w_y) / ( my_var(x, w_y) * np.sum(w_y) ) 
   
    #rho
    rho_mc = cov_mc / ( sqrt(var_m) * sqrt(var_c) )

    if (verbose):
        
        print ('m         = ', m.round(4))
        print ('sigma(m)  = ', np.sqrt(var_m).round(4))
        print ('c         = ', c.round(4))
        print ('sigma(c)  = ', np.sqrt(var_c).round(4))
        print ('cov(m, c) = ', cov_mc.round(4))
        print ('rho(m, c) = ', rho_mc.round(4))
        print ('\n')
        print ('x segnato = ', my_mean(x, w_y).round(4))
        print ('y segnato = ', my_mean(y, w_y).round(4))
        print ('x^2 segna = ', my_mean(x*x, w_y).round(4))
        print ('xy segnato= ', my_mean(x*y,w_y).round(4))
        print ('Var[x]    = ', (my_mean(x*x, w_y)-my_mean(x, w_y)**2).round(4))
        print ('Cov[x,y]  = ', (my_mean(x*y,w_y)-my_mean(x, w_y)*my_mean(y, w_y)).round(4))
        print ('pesi      = ', np.sum(w_y).round(4))
    if (plot):
        
        # rappresento i dati
        plt.errorbar(x, y, yerr=sd_y, xerr=0, ls='', marker='.', 
                     color="black",label='dati')

        # costruisco dei punti x su cui valutare la retta del fit              
        xmin = float(np.min(x)) 
        xmax = float(np.max(x))
        xmin_plot = xmin-.2*(xmax-xmin)
        xmax_plot = xmax+.2*(xmax-xmin)
        if (setrange):
            xmin_plot = xm
            xmax_plot = xM  
        x1 = np.linspace(xmin_plot, xmax_plot, 100)
        y1 = my_line(x1, m, c)
        
        # rappresento la retta del fit
        plt.plot(x1, y1, linestyle='-.', color="green", label='fit')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Fit lineare')
        plt.xlim(xmin_plot,xmax_plot)
        if (setrange):
            plt.ylim(ym,yM)
        
        # rappresento le incertezze sulla retta 
        y1_plus_1sigma = y1+3*y_estrapolato(x1, m, c, np.sqrt(var_m), np.sqrt(var_c), cov_mc)[1]
        y1_minus_1sigma = y1-3*y_estrapolato(x1, m, c, np.sqrt(var_m), np.sqrt(var_c), cov_mc)[1]         
        plt.plot(x1,y1_plus_1sigma, linestyle='-', color="orange", label=r'fit $\pm 3\sigma$')
        plt.plot(x1,y1_minus_1sigma, linestyle='-', color="orange")
        
        plt.grid()
        
        plt.legend()
        
    if (plus):
        xseg=my_mean(x, w_y)
        yseg=my_mean(y, w_y)
        xqseg=my_mean(x*x, w_y)
        xyseg=my_mean(x*y,w_y)
        varx=my_mean(x*x, w_y)-my_mean(x, w_y)**2
        covxy=my_mean(x*y,w_y)-my_mean(x, w_y)*my_mean(y, w_y)
        pesi=np.sum(w_y)
        
        return m, np.sqrt(var_m), c, np.sqrt(var_c), cov_mc, rho_mc,xseg,yseg,xqseg,xyseg,varx,covxy,pesi
        
    return m, np.sqrt(var_m), c, np.sqrt(var_c), cov_mc, rho_mc
    
