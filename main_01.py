# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import warnings
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
import time
import scipy.optimize as opt
import sys

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check

    
    DISTRIBUTIONS = [
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.burr12, st.cauchy,st.chi,st.chi2,st.cosine,
        st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldnorm,st.genlogistic,st.genexpon, st.genpareto,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace, st.levy, st.levy_l,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal, st.norminvgauss,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy, st.trapz
    ] # These are crappy distributions: st.levy_stable, st.dweibull, st.gennorm, st.foldcauchy, st.dgamma,  

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)

def best_fit_cdf(prob, data, ax=None):
    """Model data by finding best fit distribution to data"""

    # Distributions to check
    
    DISTRIBUTIONS = [st.alpha,st.anglit,st.arcsine,st.argus,st.beta,st.betaprime,st.bradford,
        st.burr,st.burr12,st.cauchy,st.chi,st.chi2,st.cosine,st.crystalball,st.dgamma,st.dweibull,
        st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.genlogistic,st.gennorm,st.genpareto,st.genexpon,st.genextreme,
        st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.genhyperbolic,st.geninvgauss,
        st.gompertz,st.gumbel_r,st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,
        st.hypsecant,st.invgamma,st.invgauss,st.invweibull,st.johnsonsb,st.johnsonsu,st.kappa4,
        st.kappa3,st.ksone,st.kstwo,st.kstwobign,st.laplace,st.laplace_asymmetric,st.levy,st.levy_l,
        st.levy_stable,st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.loguniform,st.lomax,
        st.maxwell,st.mielke,st.moyal,st.nakagami,st.ncx2,st.ncf,st.nct,st.norm,st.norminvgauss,
        st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.rayleigh,st.rice,
        st.recipinvgauss,st.semicircular,st.skewcauchy,st.skewnorm,st.studentized_range,st.t,
        st.trapezoid,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy,]

    x_cont = np.linspace(0.01, 0.99, 10000)

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = opt.curve_fit(distribution.cdf,data,prob, p0=[0,1])[0]

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted CDF and error with fit in distribution
                cdf = distribution.cdf(data, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(prob - cdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(distribution.cdf(x_cont, loc=loc, scale=scale, *arg), x_cont).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

# This function later to be merged with make_pdf
def extract_pdf(dist, params, x, q=True):
    """Generate distributions for given volumes or probabilities"""

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Build PDF and turn into pandas Series
    if q:
        y = dist.cdf(x, loc=loc, scale=scale, *arg)
        cdf = pd.Series(y, x)
    else:
        y = dist.ppf(x, loc=loc, scale=scale, *arg)
        cdf = pd.Series(x, y)
    p90 = round(dist.ppf(0.9, loc=loc, scale=scale, *arg), 2)
    p50 = round(dist.ppf(0.5, loc=loc, scale=scale, *arg), 2)
    p10 = round(dist.ppf(0.1, loc=loc, scale=scale, *arg), 2)

    return cdf[cdf.index>0], p90, p50, p10

def volume(arr):
    
    aggregate = 0
    result = []
    
    for x in arr:
        buff = np.random.choice(x)
        result.append(buff)
        aggregate += buff

    result.append(aggregate)
        
    return(result)

def vol_to_dist(arr, num_sampl=500):
    
    # aggregate = 0
    result = []
    prob = np.array([0.1, 0.5, 0.9])
    x_cont = np.linspace(0.01, 0.99, num_sampl)
    plt.figure(figsize=(12,8))

    for y, x in arr:
        
        best_fit_name, best_fit_params = best_fit_cdf(prob, x)
        best_dist = getattr(st, best_fit_name)
        (extract, p10, p50, p90) = extract_pdf(best_dist, best_fit_params, x_cont, q=False)
        
        orig = pd.DataFrame({'data':x, 'prob':prob})
        ax = orig.plot('data', 'prob', kind='scatter', s=500, alpha=0.5, 
                       color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
        extract.plot(lw=2, label='CDF', legend=True)
        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
        dist_str = '{}({})'.format(best_fit_name, param_str)
        
        ax.set_title(y + u' Input data with best fit distribution \n' + dist_str, fontsize = 16)
        ax.set_xlabel(u'Volume, MMstb', fontsize = 16)
        ax.set_ylabel('Probability', fontsize = 16)
        plt.show()
        
        result.append(extract.index.to_numpy())
        
    return(result)

if __name__ == '__main__':
    
    start_time = time.time()
    
    #Change humber of simulations if required
    num_simulations = 10000

    df = pd.read_csv(r'..\bab3.csv')
    array_input = df.iloc[:, [3, 4, 5]].to_numpy()
    
    df['Reference'] = df['Well']+" "+df['Sand']
    dataset = pd.DataFrame(columns=df['Reference'].tolist())
    dataset['Volume'] = None
    
    array_names = df.Reference.to_numpy()
    array_zip = zip(array_names, array_input)
    
    array = vol_to_dist(array_zip)
    
    for i in range(num_simulations):
        dataset.loc[len(dataset)] = volume(array)

    # print(dataset)
    # sys.exit()
    data_set_name = r'Volumetric'

    
    data = dataset['Volume']
    
    # Plot for comparison
    plt.figure(figsize=(12,8))
    
    # ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5, color=list(matplotlib.rcParams['axes.prop_cycle'])[1]['color'])
    ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
    # Save plot limits
    dataYLim = ax.get_ylim()
    
    # Find best fit distribution
    best_fit_name, best_fit_params = best_fit_distribution(data, 1000, ax)
    best_dist = getattr(st, best_fit_name)
    
    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_title(data_set_name + u' Data\n All Fitted Distributions')
    ax.set_xlabel(u'Volume, MMstb')
    ax.set_ylabel('Frequency')
    
    # Make PDF with best params 
    pdf = make_pdf(best_dist, best_fit_params)
    
    #Extract data based on volumes
    volumes = dataset['Volume'].unique()
    volumes.sort()
    (extract, p10, p50, p90) = extract_pdf(best_dist, best_fit_params, volumes)
    
    textstr = '\n'.join((
    r'$P10=%.2f MMstb$' % p10,
    r'$P50=%.2f MMstb$' % p50,
    r'$P90=%.2f MMstb$' % p90))
    
    # Display
    fig = plt.figure(figsize=(12,8))
    ax = pdf.plot(lw=2, label='PDF')#, legend=True, , ax=ax
    ax.text(0.05, 0.75, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox = dict(alpha = 0.5)) # facecolor = 'blue', 
    data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', ax=ax)#, legend=True
    ax2 = ax.twinx()
    extract.plot(lw=2, label='CDF', ax=ax2, color='g')#, legend=True
    
    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)
    
    ax.set_title(data_set_name + u' data with best fit distribution \n' + dist_str)
    ax.set_xlabel(u'Volume, MMstb')
    ax.set_ylabel('Frequency')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(bottom=0, top=1)
    # ax.set_ylim(ax.get_ylim())
    # fig = plt.figure()
    # fig.legend(loc="best")
    # ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))
    ax2.grid(None)
    fig.legend(loc=(0.75, 0.75))
    
    extract.index.name = 'Volume'
    extract.name = 'Probability'
    extract = extract.to_frame().reset_index()
    
    combos = pd.DataFrame(np.stack(np.meshgrid(*array_input), axis=-1).reshape(-1, len(array_input)), 
                          columns=array_names.tolist())
    combos['Volume']=combos.sum(axis=1, numeric_only=True)
    combos = pd.merge(combos, extract, on='Volume', how='outer').sort_values(by=['Volume'])
    combos.reset_index(inplace=True, drop=True)
    combos['Probability'] = combos.set_index('Volume')['Probability'].interpolate('index').values
    combos.dropna(inplace=True)
    combos.drop_duplicates(inplace=True)
    
    combos[(((combos.Probability>=0.49) & (combos.Probability<=0.51)) | 
             ((combos.Probability>=0.09) & (combos.Probability<=0.11)) |
             ((combos.Probability>=0.89) & 
              (combos.Probability<=0.91)))].to_csv('out_b3.csv')
    combos.to_csv('out_b3_all.csv')
    
    print("--- %s seconds ---" % (time.time() - start_time))
