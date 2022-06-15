%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from math import log10, floor, sqrt
import scipy as sp
from scipy import stats

########################
# Scatterplot/Errorbar #
########################
def scatterplot_errorbar (x, y, w_y, 
                          scatter_label, x_label, y_label, path, 
                          size=(10,8), n_ax=111, resolution=100, save=True, nplots=1):
    fig = plt.figure(figsize = size, dpi = resolution)
    #Modalità SINGLE-PLOT
    if nplots == 1:
        ax = plt.subplot(n_ax)
        #Scatter 
        ax.errorbar (x, y, yerr = w_y, 
                     linestyle = ' ', ecolor = 'cyan', fmt = '.', color = 'blue',
                     label = scatter_label)
        #Axes
        ax.xaxis.set_label_text(x_label)
        ax.yaxis.set_label_text(y_label)
        #Customization
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.legend(loc='best')
    
    #Modalità MULTI-PLOT
    else: 
        for i in range (0, nplots):
            ax = plt.subplot(n_ax[i])
            #Scatter 
            ax.errorbar (x[i], y[i], yerr = w_y[i], 
                         linestyle = ' ', ecolor = 'cyan', fmt = '.', color = 'blue',
                         label = scatter_label[i])
            #Axes
            ax.xaxis.set_label_text(x_label[i])
            ax.yaxis.set_label_text(y_label[i])
            #Customization
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.legend(loc='best')
    
    #Opzioni di salvataggio
    if (save):
        fig = plt.savefig(path)

    return fig
#############################################
# Funzione per la grafica di un fit lineare #
#############################################
def LinearFit_plot (x, y, u_y,  
                    data_label, plot_label, x_label, y_label, path, 
                    size=(10,8), n_ax=111, resolution=100, save=True, nplots=1):
    fig = plt.figure (figsize=size, dpi=resolution)  
    
    if nplots==1:
        E_m, U_m, E_c, U_c, COV_mc, chi2_ev, rho, res = Linear_Fit (x, y, u_y)
        ax = plt.subplot(n_ax)

        ax.errorbar ( x, y, yerr = u_y,
                      linestyle = ' ', ecolor = 'cyan', fmt = '.', color = 'blue',
                      label = data_label)
    
        #Curva di Regressione
        xline  = np.linspace ( np.min(x), np.max(x), 1000)
        yline  = y_lin(xline, E_m, E_c)
        
        ax.plot ( xline, yline, 
                  linestyle = '--', color = 'red',
                  label = plot_label)
        #Retta di estrapolazione
        #ax.vlines (np.min(dt_1), 0, np.exp(E_m*np.min(dt_1)+ E_c), 
        #           linestyle = 'dashdot', color= 'black', alpha = 0.5)
        
        #ax.set_xlim(0,10)
        ax.xaxis.set_label_text(x_label)
        ax.yaxis.set_label_text(y_label) 
        #Customization
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        #ax.spines['left'].set_position('zero')
        ax.legend(loc='best')
    
    #Modalità MULTI-PLOT
    else:
        for i in range (0, nplots):
        
        E_m[i], U_m[i], E_c[i], U_c[i], COV_mc[i], chi2_ev[i], rho[i], res[i] = Linear_Fit (x[i], y[i], u_y[i])
        ax = plt.subplot(n_ax[i])

        ax.errorbar ( x[i], y[i], yerr = u_y[i],
                      linestyle = ' ', ecolor = 'cyan', fmt = '.', color = 'blue',
                      label = data_label[i])
    
        #Curva di Regressione
        xline  = np.linspace ( np.min(x[i]), np.max(x[i]), 1000)
        yline  = y_lin(xline[i], E_m[i], E_c[i])
        
        ax.plot ( xline, yline, 
                  linestyle = '--', color = 'red',
                  label = plot_label[i])
        #Retta di estrapolazione
        #ax.vlines (np.min(dt_1), 0, np.exp(E_m*np.min(dt_1)+ E_c), 
        #           linestyle = 'dashdot', color= 'black', alpha = 0.5)
        
        #ax.set_xlim(0,10)
        ax.xaxis.set_label_text(x_label[i])
        ax.yaxis.set_label_text(y_label[i]) 
        #Customization
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        #ax.spines['left'].set_position('zero')
        ax.legend(loc='best')
        
    if (save):
        fig = plt.savefig(path)
    
    ######PACCARE TUTTE LE COSE UTILI DEL FIT
    return fig

#########################################################################
#                             Funzioni arrotondamento cifre significative
#########################################################################
def PrintResult(name, mean, sigma, digits, unit):
    mean = round(mean,digits)
    sigma = round(sigma,digits)
    nu = sigma / mean
    result = (name+" = ({0} +/- {1} ) ".format(mean,sigma)+unit+" [{0:.2f}%]".format(nu*100))
    print (result)
    #return ""
    
#########################################################################
#                                                Funzioni per fit lineare
#########################################################################

#########################
# Funzione media pesata #
#########################
def weighted_mean (x, w):
    return (np.sum(x*w)/(np.sum(w))
##########################################
# Funzione covarianza per il fit lineare #
##########################################
def lin_cov(x, y, w):
    return weighted_mean(x*y, w) - weighted_mean(x, w)*wieghted_mean(y, w)
########################################
# Funzione varianza per il fit lineare #
########################################
def lin_var(x, w):
    return weighted_cov(x, x, w)
############################
# Funzioni residui lineari #
############################
def res_lin (x, y, E_m, E_c, w, verbose=True):
    if (verbose):
        for i in range (0, len(y)):
            print ('e_{:} = {:}\n'.format(i+1, (y[i] - E_m*x[i] - E_c)/w[i]) )
        return (y - E_m*x - E_c)/w
    else:
        return (y - E_m*x - E_c)/w    
#######################
# Funzione chi quadro #
#######################
def chi2_lin (x, y, E_m, E_c, w):
    return np.sum( ( (y - E_m*x - E_c)/w )**2 )            
##################
# Funzione linea #
##################
def my_line(x, m, c):
    return (m*x + c)
##############################
# Incertezza sugli inviluppi #
##############################
def y_unc(xP, u_m, u_c, cov_mc):
    return np.sqrt(np.power(xP, 2)*np.power(u_m, 2) +
                   np.power(u_c, 2) + 2*xP*cov_mc )
########################
# Funzione Fit Lineare #
########################
def Linear_Fit (x, y, w, verbose_1=False, verbose_2=False, verbose_res = False, verbose_fit=True):
    
    # Per il fit lineare si utilizzano i pesi quadratici, 
    # ma è più veloce calcolarli una sola volta e pesarli già al quadrato
    
    #sums (1/w^2)
    w2_y = np.power(w, -2)
    sum_s_w = np.sum (w2_y)
    
    #valori pesati
    x_w  = weighted_mean (x, w2_y)
    y_w  = weighted_mean (y, w2_y)
    xy_w = weighted_mean (x*y, w2_y)
    x2_w = weighted_mean (x*x, w2_y) 
    # Stampa: Oggetti base
    if (verbose_1):
        print ('s_tot = {:}\n'.format(np.sqrt(1/sum_s_w))+
               'x_w = {:}\n'.format(x_w)+
               'y_w = {:}\n'.format(y_w)+
               'xy_w = {:}\n'.format(xy_w)+
               'x2_w = {:}\n'.format(x2_w))
            
    #varianza e covarianza pesata
    wvar_x  = x2_w - (x_w**2)
    wcov_xy = xy_w - x_w*y_w 
    # Stampa: Varianze "Pesate"
    if (verbose_2):
        print ('wVAR_x = {:}\n'.format(wvar_x)+
               'wCOV_xy = {:}\n'.format(wcov_xy))
    #m
    E_m = wcov_xy/wvar_x
    VAR_m = 1/(sum_s_w * wvar_x)
    U_m = np.sqrt(VAR_m)
    #c
    E_c = y_w - E_m * x_w
    VAR_c = (x2_w)/(sum_s_w * wvar_x)
    U_c = np.sqrt(VAR_c)
    #COV[m,c]
    COV_mc = -x_w/(sum_s_w * wvar_x)
    #chi2
    chi2 = chi2_lin(x, y, E_m, E_c, w)  
    #rho
    rho_mc = COV_mc / (U_m * U_c)
    #residui
    if (verbose_res):
        res = res_lin (x, y, E_m, E_c, w, verbose=True)
    else:
        res = res_lin (x, y, E_m, E_c, w, verbose=False)        
    # Stampa: "Risultati fit"
    if (verbose_fit):
        print ('E_m = {:.4}'.format(E_m)+'\n'+
               'VAR_m = {:.4}'.format(VAR_m)+'\n'+
               'U_m = {:.4}'.format(U_m)+'\n'+
               'E_c = {:.4}'.format(E_c)+'\n'+
               'VAR_c = {:.4}'.format(VAR_c)+'\n'+
               'U_c = {:.4}'.format(U_c)+'\n'+
               'chi2 = {:.4}'.format(chi2)+'\n'
               'rho_mc = {:.4}'.format(rho_mc)+'\n')
    
    return (E_m, U_m, E_c, U_c, COV_mc, chi2, rho_mc, res)
            
#########################################################################
#                                           Funzioni per fit esponenziale
#########################################################################

def Exponential_Fit (x, y, w, verbose_1_exp=False, verbose_2_exp=False, verbose_res_exp = False, verbose_fit_exp=True):
    
    # Per il fit esponenziale, si ha la relazione: ln(y) = m*x + c. 
    # La relazione diventa lineare su ln(y). Per ottenerla è necessario:
    
    # (1) tramutare le y in logaritmi. In questa parte si deve valutare la differenza dei valori con l'elemeto minimo, 
    # sia se l'array y è in ordine decrescente che viceversa 
    if ( y[0]>y[-1]):
       dy = np.abs(y[:-1]-y[-1])
       x_fit = x[:-1] 
       w_fit = w[:-1]
    else:
       dy = np.abs(y[1:]-y[0])
       x_fit = x[1:]
       w_fit = w[1:]
    ln_y = np.log(dy)
    
    # (2) tramutare le incertezze su y sulle incertezze di ln_y
    w_ln = 1/dy*w_fit

    # (3) eseguire un fit linere con i valori trovati
    return Linear_Fit (x_fit, ln_y, w_ln, verbose_1=verbose_1_exp, verbose_2=verbose_2_exp, 
                       verbose_res=verbose_res_exp, verbose_fit=verbose_fit_exp)
            
           
#########################################################################
#                                              Funzioni per le incertezze
#########################################################################          

############################
# Funzione verifica sigma #
###########################
def verify_seq(w_x, w_y, m):
    Y = w_y
    X = (m*w_x)
    print ('X : \n', X, '\nY : \n', Y, '\n')
    reiteration = False
    for i in range (0, len(Y)):
        if (X[i] >= 0.1*Y[i]):
            temp_s_eq = np.sqrt(Y**2+X**2)
            reiteration = True
    if (reiteration):
        print ('S_eq = {:.4}'.format(temp_s_eq))
        S_EQ = temp_seq
        return S_EQ
    else:
        print ('Non è necessario reiterare la sigma')
        return w_y
##########################
# Funzione sigma teorica #
##########################       
def s_th (x, y, E_m, E_c) :
    s = np.sqrt( np.sum( (y-E_m*x-E_c)**2 )/ (len(y)-2))
    print ('S_th ={:.5}'.format(s))
    return s
###############################
# Funzione errore sistematico #
###############################
def errore_sistematico (u_calc, u_th):
    E = np.sqrt(np.abs(u_calc**2-u_th**2))
    print ('Errore sistematico :\n', E)
    return E
            