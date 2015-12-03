import numpy as np
import cPickle
import pdb, sys, os, time
import matplotlib.pyplot as plt
from bayes.pyhm_dev import pyhm
from bayes.gps_dev.gps import gp_class, gp_routines, spgp_routines
import irac_kernels
from planetc import transit
import utils




def get_chain_from_walkers( walker_chains, acor_integs, ncorr_burn=3, lc_type='white' ):
    nchains = len( walker_chains )
    acor = np.zeros( nchains )
    nsteps = np.zeros( nchains )
    nwalkers = np.zeros( nchains )
    for i in range( nchains ):
        walker_chain = walker_chains[i]
        nsteps[i], nwalkers[i] = np.shape( walker_chain['logp'] )
        keys = walker_chain.keys()
        keys.remove( 'logp' )
        npar = len( keys )
        acor_vals = np.zeros( npar )
        for j in range( npar ):
            acor_vals[j] = acor_integs[i][keys[j]]
        acor[i] = np.max( np.abs( acor_vals ) )
    y = nsteps/acor
    if y.min()<ncorr_burn:
        print '\nChains only run for {0:.2f}x correlation times'.format( y.min() )
        pdb.set_trace()
    else:
        acor = acor.max()
        nburn = int( np.round( ncorr_burn*acor ) )
        chain_dicts = []
        chain_arrs = []
        for i in range( nchains ):
            chain_i = pyhm.collapse_walker_chain( walker_chains[i], nburn=nburn )
            if lc_type=='white':
                try:
                    chain_i['incl'] = np.rad2deg( np.arccos( chain_i['b']/chain_i['aRs'] ) )
                except:
                    pass
            elif lc_type=='spec':
                pass
            else:
                pdb.set_trace()
            chain_dicts += [ chain_i ]
        grs = pyhm.gelman_rubin( chain_dicts, nburn=0, thin=1 )
        chain = pyhm.combine_chains( chain_dicts, nburn=nburn, thin=1 )        
    chain['tdepth'] = chain['RpRs']**2.
    grs['tdepth'] = grs['RpRs']
    return chain, grs, nchains, nwalkers, nsteps, acor, nburn


def run_gp_mcmc_emcee( gp_mle_filepath, gp_chains_filepath, channel='ch1', dataset_type='primary', \
                       nchains=2, nwalkers=150, nsteps=150, ncorr_burn=3, orbpar_priors={} ):

    ifile = open( gp_mle_filepath )
    z = cPickle.load( ifile )
    ifile.close()

    data = z['data']
    syspars = z['syspars']
    mle_results = z['mle_results']
    #pretune = z['mcmc_pretune']
    T0_approx = z['T0_approx']
    syspars['T0'] = T0_approx

    if ( channel=='ch1' )+( channel=='ch2' ):
        covpars = { 'At':mle_results['At'], 'Lt':1./mle_results['iLt'], 'Axy':mle_results['Axy'], \
                    'Lxy':np.array( [ 1./mle_results['iLx'], 1./mle_results['iLy'] ] ), \
                    'beta':mle_results['beta'] }
    elif ( channel=='ch3' )+( channel=='ch4' ):
        pdb.set_trace()

    mbundle, gp = get_gp_fixedcov_mbundle( data, syspars, covpars, channel=channel, \
                                           orbpar_priors=orbpar_priors, dataset_type=dataset_type )


    # This is currently taken from MultiBandRoutines:
    import TransitFitting
    z = TransitFitting.MultiBand()
    z.lc_type = 'white'
    z.mbundle = mbundle
    z.par_ranges = {}
    for key in mle_results.keys():
        mlev = mle_results[key]
        z.par_ranges[key] = pyhm.Gaussian( key, mu=mlev, sigma=(1e-3)*np.abs( mlev ) )
    z.nchains = nchains
    z.nsteps = nsteps
    z.nwalkers = nwalkers
    z.mcmc()

    chain, grs, nchains, nwalkers, nsteps, acor, nburn = get_chain_from_walkers( z.walker_chains, z.acor_integs, \
                                                                                 ncorr_burn=ncorr_burn, \
                                                                                 lc_type=z.lc_type )
    y = pyhm.chain_properties( chain, nburn=0, thin=None, print_to_screen=True )    

    npar = len( y['median'].keys() )
    outstr = '#Param Med Unc_l34 Unc_u34 Fixed'
    if dataset_type=='primary':
        outstr += '\nRpRs {0:.6f} {1:.6f} {2:.6f} 0'\
                  .format( y['median']['RpRs'], np.abs( y['l34']['RpRs'] ), y['u34']['RpRs'] )
        outstr += '\naRs {0:.3f} {1:.3f} {2:.3f} 0'\
                  .format( y['median']['aRs'], np.abs( y['l34']['aRs'] ), y['u34']['aRs'] )
        outstr += '\nb {0:.3f} {1:.3f} {2:.3f} 0'\
                  .format( y['median']['b'], np.abs( y['l34']['b'] ), y['u34']['b'] )
        outstr += '\nincl {0:.3f} {1:.3f} {2:.3f} 0'\
                  .format( y['median']['incl'], np.abs( y['l34']['incl'] ), y['u34']['incl'] )
    elif dataset_type=='secondary':
        outstr += '\nSecDepth {0:.6f} {1:.6f} {2:.6f} 0'\
                  .format( y['median']['SecDepth'], np.abs( y['l34']['SecDepth'] ), y['u34']['SecDepth'] )
    outstr += '\nT0 {0:.6f} {1:.6f} {2:.6f} 0'\
              .format( T0_approx+y['median']['delT'], np.abs( y['l34']['delT'] ), y['u34']['delT'] )
    if dataset_type=='primary':
        outstr += '\necc {0:.4f} 0 0 1'.format( syspars['ecc'] )
        outstr += '\nomega {0:.4f} 0 0 1'.format( syspars['omega'] )
    elif dataset_type=='secondary':
        outstr += '\nRpRs {0:.4f} 0 0 1'.format( syspars['RpRs'] )
        outstr += '\nb {0:.4f} 0 0 1'.format( syspars['b'] )
        outstr += '\naRs {0:.4f} 0 0 1'.format( syspars['aRs'] )
        outstr += '\nincl {0:.4f} 0 0 1'.format( syspars['incl'] )
        outstr += '\necc {0:.4f} 0 0 1'.format( syspars['ecc'] )
        outstr += '\nomega {0:.4f} 0 0 1'.format( syspars['omega'] )
    print outstr    

    output = { 'data':data, 'syspars':syspars }
    output['T0_approx'] = T0_approx
    output['mle_results'] = z.mle_refined#mle_results
    output['mcmc_results'] = {}
    output['mcmc_results']['walker_chains'] = z.walker_chains
    output['mcmc_results']['chain_properties'] = y
    output['mcmc_results']['grs'] = grs
    output['mcmc_results']['ncorr_burn'] = ncorr_burn
    if os.path.isdir( os.path.dirname( gp_chains_filepath ) )==False:
        os.makedirs( os.path.dirname( gp_chains_filepath ) )
    ofile = open( gp_chains_filepath, 'w' )
    cPickle.dump( output, ofile )
    ofile.close()
    gp_chains_filepath_str = gp_chains_filepath.replace( '.pkl', '.txt' )
    ofile = open( gp_chains_filepath_str, 'w' )
    ofile.write( outstr )
    ofile.close()
    print '\nSaved:\n{0}\n{1}'.format( gp_chains_filepath, gp_chains_filepath_str )

    return output


def run_gp_mcmc_chains( gp_pretune_filepath, gp_chains_filepath, channel='ch1', dataset_type='primary', \
                        nchains=5, nsteps_increment=10000, nburn_frac=0.5, orbpar_priors={} ):

    ifile = open( gp_pretune_filepath )
    z = cPickle.load( ifile )
    ifile.close()

    data = z['data']
    syspars = z['syspars']
    mle_results = z['mle_results']
    pretune = z['mcmc_pretune']
    T0_approx = z['T0_approx']
    syspars['T0'] = T0_approx

    if ( channel=='ch1' )+( channel=='ch2' ):
        covpars = { 'At':mle_results['At'], 'Lt':1./mle_results['iLt'], 'Axy':mle_results['Axy'], \
                    'Lxy':np.array( [ 1./mle_results['iLx'], 1./mle_results['iLy'] ] ), \
                    'beta':mle_results['beta'] }
    elif ( channel=='ch3' )+( channel=='ch4' ):
        pdb.set_trace()

    mbundle, gp = get_gp_fixedcov_mbundle( data, syspars, covpars, channel=channel, \
                                           orbpar_priors=orbpar_priors, dataset_type=dataset_type )
    mcmcs = []

    for i in range( nchains ):
        mcmc = pyhm.MCMC( mbundle )
        mcmc.assign_step_method( pyhm.BuiltinStepMethods.MetropolisHastings )
        mcmc.step_method.assign_proposal_distribution( pyhm.BuiltinProposals.diagonal_gaussian )
        step_sizes = {}
        for key in mcmc.model.free.keys():
            mle = mle_results[key]
            med = pretune['meds'][key]
            sig = pretune['sigs'][key]
            within_prior = False
            while within_prior==False:
                mcmc.model.free[key].value = mle + 3*sig*np.random.randn()
                if np.isfinite( mcmc.model.free[key].logp() ):
                    within_prior = True
            step_sizes[key] = pretune['step_sizes'][key]
        mcmc.step_method.proposal_distribution.proposal_kwargs['step_sizes'] = step_sizes
        mcmcs += [ mcmc ]

    converged = False
    counter = 0
    nsteps_min = 10000
    while converged==False:
        counter += 1
        # Cycle through the chains:
        for i in range( nchains ):
            print '\nSampling chain {0} of {1}...'.format( i+1, nchains )
            mcmcs[i].sample( nsteps=nsteps_increment, verbose=1, pickle_chain=None, \
                             overwrite_existing_chains=False )

        # Check GR values to see if converged:
        chains = []
        for i in range( nchains ):
            chains += [ mcmcs[i].chain ]
        nsteps = counter*nsteps_increment 
        nburn = int( np.round( nsteps*nburn_frac ) )
        grs = pyhm.gelman_rubin( chains, nburn=nburn, thin=1 )
        dgrs = {}
        max_dgr = 0.0
        print '\nGelman-Rubin values after {0} steps:'.format( nsteps )
        for key in grs.keys():
            dgrs[key] = np.abs( grs[key]-1 )
            if dgrs[key]>max_dgr:
                max_dgr = dgrs[key]
                print key, grs[key]
        print '(require minimum {0} steps to have been taken before finishing)'.format( nsteps_min )
        if ( max_dgr<0.01 )*( nsteps>nsteps_min ):
            converged = True
    print 'Chains converged in {0} steps'.format( nsteps )
    combined_chain = pyhm.combine_chains( chains, nburn=nburn, thin=1 )
    if dataset_type=='primary':
        combined_chain['incl'] = np.rad2deg( np.arccos( combined_chain['b']/combined_chain['aRs'] ) )
    y = pyhm.chain_properties( combined_chain, nburn=0, thin=None, print_to_screen=False )

    npar = len( y['median'].keys() )
    outstr = '#Param Med Unc_l34 Unc_u34 Fixed'
    if dataset_type=='primary':
        outstr += '\nRpRs {0:.6f} {1:.6f} {2:.6f} 0'\
                  .format( y['median']['RpRs'], np.abs( y['l34']['RpRs'] ), y['u34']['RpRs'] )
        outstr += '\naRs {0:.3f} {1:.3f} {2:.3f} 0'\
                  .format( y['median']['aRs'], np.abs( y['l34']['aRs'] ), y['u34']['aRs'] )
        outstr += '\nb {0:.3f} {1:.3f} {2:.3f} 0'\
                  .format( y['median']['b'], np.abs( y['l34']['b'] ), y['u34']['b'] )
        outstr += '\nincl {0:.3f} {1:.3f} {2:.3f} 0'\
                  .format( y['median']['incl'], np.abs( y['l34']['incl'] ), y['u34']['incl'] )
    elif dataset_type=='secondary':
        outstr += '\nSecDepth {0:.6f} {1:.6f} {2:.6f} 0'\
                  .format( y['median']['SecDepth'], np.abs( y['l34']['SecDepth'] ), y['u34']['SecDepth'] )
    outstr += '\nT0 {0:.6f} {1:.6f} {2:.6f} 0'\
              .format( T0_approx+y['median']['delT'], np.abs( y['l34']['delT'] ), y['u34']['delT'] )
    if dataset_type=='primary':
        outstr += '\necc {0:.4f} 0 0 1'.format( syspars['ecc'] )
        outstr += '\nomega {0:.4f} 0 0 1'.format( syspars['omega'] )
    elif dataset_type=='secondary':
        outstr += '\nRpRs {0:.4f} 0 0 1'.format( syspars['RpRs'] )
        outstr += '\nb {0:.4f} 0 0 1'.format( syspars['b'] )
        outstr += '\naRs {0:.4f} 0 0 1'.format( syspars['aRs'] )
        outstr += '\nincl {0:.4f} 0 0 1'.format( syspars['incl'] )
        outstr += '\necc {0:.4f} 0 0 1'.format( syspars['ecc'] )
        outstr += '\nomega {0:.4f} 0 0 1'.format( syspars['omega'] )
    print outstr    

    output = { 'data':data, 'syspars':syspars }
    output['T0_approx'] = T0_approx
    output['mle_results'] = mle_results
    output['mcmc_chains'] = chains
    output['mcmc_nburn'] = nburn
    if os.path.isdir( os.path.dirname( gp_chains_filepath ) )==False:
        os.makedirs( os.path.dirname( gp_chains_filepath ) )
    ofile = open( gp_chains_filepath, 'w' )
    cPickle.dump( output, ofile )
    ofile.close()
    gp_chains_filepath_str = gp_chains_filepath.replace( '.pkl', '.txt' )
    ofile = open( gp_chains_filepath_str, 'w' )
    ofile.write( outstr )
    ofile.close()
    print '\nSaved:\n{0}\n{1}'.format( gp_chains_filepath, gp_chains_filepath_str )

    return output


def run_gp_mcmc_pretune( mle_ifilepath, pretune_ofilepath, channel='ch1', dataset_type='primary', \
                         tune_interval=42, nsteps=50000, nburn=30000, orbpar_priors={} ):
    
    ifile = open( mle_ifilepath )
    z = cPickle.load( ifile )
    ifile.close()

    data = z['data']
    syspars = z['syspars']
    mle_results = z['mle_results']
    T0_approx = z['T0_approx']
    syspars['T0'] = T0_approx
    
    if ( channel=='ch1' )+( channel=='ch2' ):
        covpars = { 'At':mle_results['At'], 'Lt':1./mle_results['iLt'], 'Axy':mle_results['Axy'], \
                    'Lxy':np.array( [ 1./mle_results['iLx'], 1./mle_results['iLy'] ] ), \
                    'beta':mle_results['beta'] }
    elif ( channel=='ch3' )+( channel=='ch4' ):
        pdb.set_trace()

    mbundle, gp = get_gp_fixedcov_mbundle( data, syspars, covpars, channel=channel, \
                                           orbpar_priors=orbpar_priors, dataset_type=dataset_type )
    mcmc = pyhm.MCMC( mbundle )
    mcmc.assign_step_method( pyhm.BuiltinStepMethods.MetropolisHastings )
    mcmc.step_method.assign_proposal_distribution( pyhm.BuiltinProposals.diagonal_gaussian )
    step_sizes = {}
    for key in mcmc.model.free.keys():
        mcmc.model.free[key].value = mle_results[key]
        step_sizes[key] = (5e-3)*np.abs( mle_results[key] )
    mcmc.step_method.proposal_distribution.proposal_kwargs['step_sizes'] = step_sizes
    nfree = len( mcmc.model.free.keys() )
    ntune_iterlim = nfree*10000
    mcmc.pretune( ntune_iterlim=ntune_iterlim, tune_interval=tune_interval, nconsecutive=3, verbose=1 )
    mcmc.sample( nsteps=nsteps, verbose=1, pickle_chain=None, overwrite_existing_chains=True )
    meds = {}
    sigs = {}
    step_sizes = {}
    for key in mcmc.model.free.keys():
        trace = mcmc.chain[key][nburn:]
        meds[key] = np.median( trace )
        sigs[key] = np.std( trace )
        step_sizes[key] = mcmc.step_method.proposal_distribution.proposal_kwargs['step_sizes'][key]
    
    output = { 'data':data, 'syspars':syspars }
    output['T0_approx'] = T0_approx
    output['mle_results'] = mle_results
    output['mcmc_pretune'] = {}
    output['mcmc_pretune']['meds'] = meds
    output['mcmc_pretune']['sigs'] = sigs
    output['mcmc_pretune']['step_sizes'] = step_sizes

    if os.path.isdir( os.path.dirname( pretune_ofilepath ) )==False:
        os.makedirs( os.path.dirname( pretune_ofilepath ) )
    ofile = open( pretune_ofilepath, 'w' )
    cPickle.dump( output, ofile )
    ofile.close()
    print '\nSaved: {0}'.format( pretune_ofilepath )
    
    return output

def run_gp_mle( data, syspars, mle_ofilepath, initial_ranges={}, channel='ch1', dataset_type='primary', \
                prefit_ntrials=3, prefit_nbins=250, orbpar_priors={}, make_plot=True ):
    """
    """

    T0_approx = syspars['T0']

    # Create the binned dataset:
    datab = {}
    for key in data.keys():
        xb, yb, stdvs, npb = utils.bin_1d( data['bjd'], data[key], nbins=prefit_nbins )
        ixs = npb>=1
        datab[key] = yb[ixs]
    datab['uncs'] /= np.sqrt( npb[ixs] )

    # Get the model bundles:
    mbundleb, gpb = get_gp_freecov_mbundle( datab, syspars, dataset_type=dataset_type, orbpar_priors=orbpar_priors )
    mbundle, gp = get_gp_freecov_mbundle( data, syspars, dataset_type=dataset_type, orbpar_priors=orbpar_priors )

    mpb = pyhm.MAP( mbundleb )
    mcmcb = pyhm.MCMC( mbundleb )
    mcmcb.assign_step_method( pyhm.BuiltinStepMethods.MetropolisHastings )
    mcmcb.step_method.assign_proposal_distribution( pyhm.BuiltinProposals.diagonal_gaussian )
    mles = {}
    for key in mpb.model.free.keys():
        mles[key] = np.zeros( prefit_ntrials )
    mles['logp'] = np.zeros( prefit_ntrials )
    verbose = 1
    for i in range( prefit_ntrials ):
        invalid = False
        t1 = time.time()
        print '\nRunning pre-fit trial {0} of {1}:'.format( i+1, prefit_ntrials )
        for key in mpb.model.free.keys():
            try:
                mpb.model.free[key].value = initial_ranges[key].random()
            except:
                mpb.model.free[key].value = initial_ranges[key]
            print key, mpb.model.free[key].value, mpb.model.free[key].logp()
            if np.isfinite( mpb.model.free[key].logp() )==False:
                invalid = True
        if invalid==True:
            print '\n\nInvalid starting point\n\n'
            pdb.set_trace()
        mpb.fit( maxfun=10000, maxiter=10000 )
        mles['logp'][i] = mpb.logp()
        for key in mpb.model.free.keys():
            mles[key][i] = mpb.model.free[key].value

        ##################################
        step_sizes = {}
        for key in mcmcb.model.free.keys():
            step_sizes[key] = np.abs( mpb.model.free[key].value )*0.1#(1e-3)
        mcmcb.step_method.proposal_distribution.proposal_kwargs['step_sizes'] = step_sizes
        n_steps = 3000
        n_burn = int( 0.5*n_steps )
        n_free = len( mcmcb.model.free.keys() )
        n_tune_iterlim = n_free*1000000

        print '\nPre-tuning for short MCMC chain...'
        if verbose>0:
            verbose_tune = True
        else:
            verbose_tune = False
        try:
            mcmcb.pretune( ntune_iterlim=n_tune_iterlim, tune_interval=30, nconsecutive=3, verbose=verbose_tune )
            mcmcb.sample( nsteps=n_steps, verbose=verbose_tune, pickle_chain=None )
            print '\nTuned step-sizes:'
            for key in mpb.model.free.keys():
                mpb.model.free[key].value = np.median( mcmcb.chain[key][n_burn:] )
                step_sizes = mcmcb.step_method.proposal_distribution.proposal_kwargs['step_sizes']
                print '{0} --> {1} x abs( mp.value )'\
                      .format( key, step_sizes[key]/np.abs( mpb.model.free[key].value ) )
            mpb.fit()

            for key in mpb.model.free.keys():
                mles[key][i] = mpb.model.free[key].value
            mles['logp'][i] = mpb.logp()

        except:
            print '\nTuning failed... skipping to next trial (fitting.py module)\n'
            mles['logp'][i] = -np.inf
            continue

        t2 = time.time()
        print 'Finished. Fit took {0:.2f} minutes.'.format( ( t2-t1 )/60. )
    ixs = np.isfinite( mles['logp'] )
    if ixs.sum()<2:
        print '\n\n!! Fits seemed to fail too many times !!\n\n'
        pdb.set_trace()
    print '\n{0}\nBest results from time-binned GP:'.format( 50*'#' )
    ix = np.argmax( mles['logp'] )
    for key in mles.keys():
        if key=='logp':
            continue
        else:
            print key, mles[key][ix]

    mp = pyhm.MAP( mbundle )
    for key in mp.model.free.keys():
        mp.model.free[key].value = mles[key][ix]
    print '\nRunning MLE for unbinned GP...'
    t1 = time.time()
    mp.fit()
    t2 = time.time()
    print 'Finished (took {0:.2f} minutes)'.format( ( t2-t1 )/60. )
    print '\nFinal MLE results for unbinned GP model:'
    for key in mp.model.free.keys():
        print key, mp.model.free[key].value
    output = { 'data':data, 'syspars':syspars }
    output['mle_results'] = {}
    for key in mp.model.free.keys():
        output['mle_results'][key] = mp.model.free[key].value

    z = eval_mle_bestfit( data, gp, output['syspars'], output['mle_results'], T0_approx, \
                          dataset_type=dataset_type, channel=channel )
    output['T0_approx'] = T0_approx
    output['syspars_fit'] = z[0]
    output['psignal_fit'] = z[1]
    output['systematics_fit'] = z[2]
    output['model_unc'] = z[3]    

    if os.path.isdir( os.path.dirname( mle_ofilepath ) )==False:
        os.makedirs( os.path.dirname( mle_ofilepath ) )
    ofile = open( mle_ofilepath, 'w' )
    cPickle.dump( output, ofile )
    ofile.close()
    print '\nSaved: {0}'.format( mle_ofilepath )

    if make_plot==True:
        plt.ioff()
        fig = plt.figure( figsize=[10,10] )
        figh = 0.25
        figw = 0.8
        xlow = 0.15
        ylow1 = 0.7
        ylow2 = ylow1-0.05-figh
        ylow3 = ylow2-0.05-figh
        ax1 = fig.add_axes( [ xlow, ylow1, figw, figh ] )        
        ax2 = fig.add_axes( [ xlow, ylow2, figw, figh ], sharex=ax1 )
        ax3 = fig.add_axes( [ xlow, ylow3, figw, figh ], sharex=ax1 )
        ax1.errorbar( data['bjd'], data['flux'], yerr=output['model_unc'], fmt='.k' )
        model = output['psignal_fit']*output['systematics_fit']
        ax1.plot( data['bjd'], model, '-r' )
        #ax1.fill_between( data['bjd'], model-output['model_unc'], model+output['model_unc'], \
        #                  color=0.6*np.ones( 3 ) )
        ax2.errorbar( data['bjd'], data['flux']/output['systematics_fit'], \
                      yerr=output['model_unc'], fmt='.k' )
        ax2.plot( data['bjd'], output['psignal_fit'], '-r' )
        resids = data['flux']-model
        delt = data['bjd'].max() - data['bjd'].min()
        binw = 3./60./24.
        nbins = int( np.round( delt/binw ) ) 
        bjdb, residsb, stdvs, npb = utils.bin_1d( data['bjd'], resids, nbins=nbins )
        bjdb, uncsb, stdvs, npb = utils.bin_1d( data['bjd'], output['model_unc'], nbins=nbins )
        ixs = npb>1
        yerr = uncsb[ixs]/np.sqrt( npb[ixs] )
        ax3.errorbar( bjdb[ixs], residsb[ixs], yerr=yerr, fmt='ok', zorder=2 )
        ax3.plot( data['bjd'], resids, '.', color=[0.7,0.7,0.7], zorder=1 )
        ax3.axhline( 0, color='r', ls='-', zorder=0 )
        x1 = data['bjd'].min() - 10./60./24.
        x2 = data['bjd'].max() + 10./60./24.
        ax1.set_xlim( [ x1, x2 ] )
        label_fs = 14
        fig.text( 0.05, ylow1+0.5*figh, 'Rel Flux', fontsize=label_fs, rotation=90., \
                  horizontalalignment='right', verticalalignment='center' )
        fig.text( 0.05, ylow2+0.5*figh, 'Rel Flux', fontsize=label_fs, rotation=90., \
                  horizontalalignment='right', verticalalignment='center' )
        fig.text( 0.05, ylow3+0.5*figh, 'Resids', fontsize=label_fs, rotation=90., \
                  horizontalalignment='right', verticalalignment='center' )
        fig.text( 0.5, 0.03, 'BJD_UTC', fontsize=label_fs, rotation=0., \
                  horizontalalignment='center', verticalalignment='bottom' )
        figpath = mle_ofilepath.replace( '.pkl', '.pdf' )
        fig.savefig( figpath )
        plt.ion()
        print 'Saved: {0}'.format( figpath )

    return output


def eval_mle_bestfit( data, gp, syspars, mle_results, T0_approx, dataset_type='primary', channel='ch1' ):

    bjd = data['bjd']
    tv = data['tv']
    xv = data['xv']
    yv = data['yv']
    flux = data['flux']
    uncs = data['uncs']
    ndat = len( flux )

    if dataset_type=='primary':
        syspars['RpRs'] = mle_results['RpRs']
        syspars['aRs'] = mle_results['aRs']
        syspars['b'] = mle_results['b']
        syspars['incl'] = np.rad2deg( np.arccos( syspars['b']/syspars['aRs'] ) )
    else:
        syspars['SecDepth'] = mle_results['SecDepth']
    syspars['T0'] = T0_approx + mle_results['delT']
    psignal = transit.ma02_aRs( bjd, **syspars )
    resids = flux.flatten()/psignal.flatten()
    residsm = np.median( resids )
    resids -= residsm

    if ( channel=='ch1' )+( channel=='ch2' ):
        #mle_results['At']=0#delete
        gp.cpars = { 'Axy':mle_results['Axy'], 'At':mle_results['At'], 'Lt':1./mle_results['iLt'], \
                     'Lxy':np.array( [ 1./mle_results['iLx'], 1./mle_results['iLy'] ] ) }
    elif ( channel=='ch3' )+( channel=='ch4' ):
        pdb.set_trace()

    gp.dtrain = np.reshape( resids, [ ndat, 1 ] )
    gp.etrain = ( 1+mle_results['beta']-1e-6 )*uncs
    mu, sig = gp.predictive( xnew=gp.xtrain, enew=gp.etrain )
    systematics = mu.flatten() + residsm

    return syspars, psignal, systematics, sig.flatten()


def mpfit_xypoly( data, syspars, channel='', ramp_type=None, x_poly=2, y_poly=2, xy_cross=1, make_plot=False ):

    T0_approx = syspars['T0']

    if ( channel=='ch1' )+( channel=='ch2' ):

        xv = data['xv']
        yv = data['yv']
        ndat = len( data['flux'] )
        offset = np.ones( [ ndat, 1 ] )
        basis = [ offset ] 
        for i in range( x_poly ):
            basis += [ xv**( i+1 ) ]
        for i in range( y_poly ):
            basis += [ yv**( i+1 ) ]
        if xy_cross==1:
            basis += [ xv*yv ]
        basis = np.column_stack( basis )
        if syspars['tr_type']=='primary':
            results = mpfit_linmodel_primary( data, syspars, basis, make_plot=make_plot )
        elif syspars['tr_type']=='secondary':
            results = mpfit_linmodel_secondary( data, syspars, basis, make_plot=make_plot )
        else:
            pdb.set_trace()

    return results



def mpfit_pld( data, syspars, pix_timeseries, channel='', t_poly=2, make_plot=False ):

    ndat, npix = np.shape( pix_timeseries )
    pix_norm_v = np.sum( pix_timeseries, axis=1 )
    for i in range( npix ):
        pix_timeseries[:,i] = pix_timeseries[:,i]/pix_norm_v

    if ( channel!='ch1' )*( channel!='ch2' ):
        pdb.set_trace() # this routine is only intended for ch1 and ch2
    else:
        ndat, npix = np.shape( pix_timeseries )
        means = np.mean( pix_timeseries, axis=0 )
        stdvs = np.std( pix_timeseries, axis=0 )
        for i in range( npix ):
            pix_timeseries[:,i] = ( pix_timeseries[:,i]-means[i] )/stdvs[i]
        offset = np.ones( [ ndat, 1 ] )
        if t_poly>0:
            t_poly_basis = np.zeros( [ ndat, t_poly ] )
            for i in range( t_poly ):
                t_poly_basis[:,i] = data['tv']**(i+1)
            basis = np.column_stack( [ offset, pix_timeseries, t_poly_basis ] )
        else:
            basis = np.column_stack( [ offset, pix_timeseries ] )
        if syspars['tr_type']=='primary':
            results = mpfit_linmodel_primary( data, syspars, basis )
        elif syspars['tr_type']=='secondary':
            results = mpfit_linmodel_secondary( data, syspars, basis, make_plot=make_plot )
        else:
            pdb.set_trace()
    pdb.set_trace()
    return None


def mpfit_linmodel_primary( data, syspars, basis, make_plot=False ):

    T0_approx = syspars['T0']
    ndat, ncoeffs = np.shape( basis )
    coeffs_init = np.zeros( ncoeffs )
    coeffs_init[0] = 1.0
    def resids_func( pars, fjac=None, x=None, y=None, err=None ):
        if ( pars[0]<0 )+( pars[2]<0 ):
            status = 0
            resids = (1e10)*np.ones( ndat )
            return status, resids
        else:
            syspars['RpRs'] = pars[0]
            syspars['T0'] = T0_approx + pars[1]
            #syspars['T0'] = 2456252.85107 + pars[1]
            syspars['aRs'] = pars[2]
            syspars['b'] = pars[3]
            syspars['incl'] = np.rad2deg( np.arccos( syspars['b']/syspars['aRs'] ) )
            psignal = transit.ma02_aRs( data['bjd'], **syspars )
            coeffs = pars[4:4+ncoeffs]
            systematics = np.dot( basis, coeffs )
            resids = data['flux'].flatten() - psignal.flatten()*systematics.flatten()
            resids /= err
            status = 0
            pderive = None
            if pars[0]>0.3:
                plt.figure()
                plt.errorbar(data['bjd'],data['flux'],fmt='ok',yerr=err)
                plt.plot(data['bjd'],data['flux'],'or')
                plt.plot(data['bjd'],psignal*systematics,'-r')
                plt.plot(data['bjd'],psignal,'-b')
                plt.axvline(syspars['T0'])
                pdb.set_trace()
            return status, resids
    transit_init = np.array( [ syspars['RpRs'], 0.0, syspars['aRs'], syspars['b'] ] )
    pars_init = np.concatenate( [ transit_init, coeffs_init ] )
    fa = { 'x':data['bjd'], 'y':data['flux'], 'err':data['uncs'] }
    results = utils.mpfit( resids_func, pars_init, functkw=fa, quiet=1 )

    output = {}
    output['pars_fit'] = {}
    output['pars_err'] = {}
    keys = [ 'RpRs', 'delT', 'aRs', 'b' ]
    for i in range( ncoeffs ):
        keys += [ 'c{0}'.format( i ) ]
    npar = len( results.params )
    for i in range( npar ):
        output['pars_fit'][keys[i]] = results.params[i]
        output['pars_err'][keys[i]] = results.perror[i]
    output['T0_approx'] = T0_approx
    syspars['RpRs'] = results.params[0]
    syspars['T0'] = T0_approx + results.params[1]
    syspars['aRs'] = results.params[2]
    syspars['b'] = results.params[3]
    syspars['incl'] = np.rad2deg( np.arccos( syspars['b']/syspars['aRs'] ) )
    psignal = transit.ma02_aRs( data['bjd'], **syspars )
    coeffs = results.params[4:4+ncoeffs]
    systematics = np.dot( basis, coeffs )
    resids = data['flux'].flatten() - psignal.flatten()*systematics.flatten()
    chi2 = np.sum( ( resids/data['uncs'] )**2. )
    npar = len( results.params )
    bic = chi2 + npar*np.log( ndat )
    output['rms'] = np.sqrt( np.mean( resids**2. ) )
    output['chi2'] = chi2
    ndof = ndat-npar
    output['redchi2'] = chi2/float( ndof )
    output['bic'] = bic
    output['psignal'] = psignal
    output['systematics'] = systematics
    output['data'] = data
    print '\nRpRs = {0:.5f} +/- {1:.5f}'.format( results.params[0], results.perror[0] )
    print 'delT = {0:.10f} +/- {1:.10f} sec'.format( results.params[1]*24*60*60, results.perror[1]*24*60*60 )
    print 'aRs = {0:.3f} +/- {1:.3f}'.format( results.params[2], results.perror[2] )
    print 'b = {0:.3f} +/- {1:.3f}'.format( results.params[3], results.perror[3] )

    if make_plot==True:
        plt.figure()
        plt.plot(data['bjd'],data['flux']/systematics,'.k')
        plt.plot(data['bjd'],psignal,'-r')
        while syspars['T0']<data['bjd'].min():
            syspars['T0'] += syspars['P']
        if syspars['T0']>data['bjd'].max():
            pdb.set_trace() # tmid not found correctly
        plt.axvline(syspars['T0'])
        pdb.set_trace()
    return output

def mpfit_linmodel_secondary( data, syspars, basis, make_plot=False ):
    T0_approx = syspars['T0']
    ndat, ncoeffs = np.shape( basis )
    coeffs_init = np.zeros( ncoeffs )
    coeffs_init[0] = 1.0
    def resids_func( pars, fjac=None, x=None, y=None, err=None ):
        #if ( pars[0]<0 )+( pars[2]<0 ):
        #    status = 0
        #    resids = (1e10)*np.ones( ndat )
        #    return status, resids
        #else:
        if 1:
            syspars['SecDepth'] = pars[0]
            syspars['T0'] = T0_approx + pars[1]
            #syspars['T0'] = 2456252.85107 + pars[1]
            #syspars['aRs'] = pars[2]
            #syspars['b'] = pars[3]
            #syspars['incl'] = np.rad2deg( np.arccos( syspars['b']/syspars['aRs'] ) )
            psignal = transit.ma02_aRs( data['bjd'], **syspars )            
            #print 'aaaaa', pars[0], psignal.min(), psignal.max(), len(data['bjd'])
            #print 'ccccccc', len( data['bjd'] )
            if 0:

                plt.figure()
                print syspars['T0'], pars[1]
                bjd2=np.r_[data['bjd'].min()-0.6*syspars['P']:data['bjd'].max()+0.6*syspars['P']:1j*700]
                syspars['tr_type'] = 'both'
                psignal2=transit.ma02_aRs( bjd2, **syspars )
                syspars['tr_type'] = 'secondary'
                psignal3=transit.ma02_aRs( bjd2, **syspars )
                plt.plot(data['bjd'],data['flux'],'.c')
                plt.plot(bjd2,psignal2,'-r')
                plt.plot(bjd2,psignal2,'--k')
                plt.plot(data['bjd'],psignal,'-g')
                pdb.set_trace()
            coeffs = pars[2:2+ncoeffs]
            systematics = np.dot( basis, coeffs )
            resids = data['flux'].flatten() - psignal.flatten()*systematics.flatten()
            resids /= err
            status = 0
            pderive = None
            if pars[0]>0.3:
                plt.figure()
                plt.errorbar(data['bjd'],data['flux'],fmt='ok',yerr=err)
                plt.plot(data['bjd'],data['flux'],'or')
                plt.plot(data['bjd'],psignal*systematics,'-r')
                plt.plot(data['bjd'],psignal,'-b')
                plt.axvline(syspars['T0'])
                pdb.set_trace()
            return status, resids
    #syspars['SecDepth']=0.002
    transit_init = np.array( [ syspars['SecDepth'], 0.0 ] )
    pars_init = np.concatenate( [ transit_init, coeffs_init ] )
    fa = { 'x':data['bjd'], 'y':data['flux'], 'err':data['uncs'] }
    results = utils.mpfit( resids_func, pars_init, functkw=fa, quiet=1 )

    #output = {}
    #output['pars_fit'] = results.params
    output = {}
    output['pars_fit'] = {}#results.params
    output['pars_err'] = {}#results.params
    keys = [ 'SecDepth', 'delT' ]
    for i in range( ncoeffs ):
        keys += [ 'c{0}'.format( i ) ]
    npar = len( results.params )
    for i in range( npar ):
        output['pars_fit'][keys[i]] = results.params[i]
        output['pars_err'][keys[i]] = results.perror[i]
    output['T0_approx'] = T0_approx
    syspars['SecDepth'] = results.params[0]
    syspars['T0'] = T0_approx + results.params[1]
    psignal = transit.ma02_aRs( data['bjd'], **syspars )
    coeffs = results.params[2:2+ncoeffs]
    systematics = np.dot( basis, coeffs )
    resids = data['flux'].flatten() - psignal.flatten()*systematics.flatten()
    chi2 = np.sum( ( resids/data['uncs'] )**2. )
    npar = len( results.params )
    bic = chi2 + npar*np.log( ndat )
    output['rms'] = np.sqrt( np.mean( resids**2. ) )
    output['chi2'] = chi2
    ndof = ndat-npar
    output['redchi2'] = chi2/float( ndof )
    output['bic'] = bic
    output['psignal'] = psignal
    output['systematics'] = systematics
    output['data'] = data
    print '\nSecDepth = {0:.5f} +/- {1:.5f}'.format( results.params[0], results.perror[0] )
    print 'delT = {0:.10f} +/- {1:.10f} sec'.format( results.params[1]*24*60*60, results.perror[1]*24*60*60 )

    if make_plot==True:
        plt.figure()
        tb, fb, stdvs, npb = utils.bin_1d( data['bjd'], data['flux']/systematics, nbins=100 )
        ixs = npb>1
        plt.plot(data['bjd'],data['flux']/systematics,'.c')
        plt.plot(tb[ixs], fb[ixs], 'ok')
        plt.plot(data['bjd'],psignal,'-r')
        while syspars['T0']<data['bjd'].min():
            syspars['T0'] += syspars['P']
        plt.axhline( 1-results.params[0], ls='-', color='g' )
        plt.axhline( 1-results.params[0]+results.perror[0], ls='--', color='g' )
        plt.axhline( 1-results.params[0]-results.perror[0], ls='--', color='g' )
        #if syspars['T0']>data['bjd'].max():
        #    pdb.set_trace() # tmid not found correctly
        #plt.axvline(syspars['T0'])
        pdb.set_trace()
    return output


def get_gp_fixedcov_mbundle( data, syspars, covpars, channel='ch1', ramp_type=None, \
                             orbpar_priors={}, dataset_type='primary' ):

    delT = pyhm.Uniform( 'delT', lower=-2./24., upper=2./24. )
    if dataset_type=='primary':
        if orbpar_priors=={}:
            orbpar_priors = None
        RpRs = pyhm.Uniform( 'RpRs', lower=0.5*syspars['RpRs'], upper=1.5*syspars['RpRs'] )
        if orbpar_priors==None:
            aRs = pyhm.Uniform( 'aRs', lower=0.5*syspars['aRs'], upper=1.5*syspars['aRs'] )
            b = pyhm.Uniform( 'b', lower=0.5*syspars['b'], upper=1.5*syspars['b'] )
        else:
            aRs = orbpar_priors['aRs']
            b = orbpar_priors['b']
        transitvars = { 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT } # these can be free or fixed
        parents, gp_logp, gp_obj = gp_fixedcov_model_primary( data, syspars, transitvars, covpars, channel, \
                                                              ramp_type, verbose=1 )
    elif dataset_type=='secondary':
        SecDepth = pyhm.Uniform( 'SecDepth', lower=-0.01, upper=0.01 )
        transitvars = { 'SecDepth':SecDepth, 'delT':delT } # these can be free or fixed
        parents, gp_logp, gp_obj = gp_fixedcov_model_secondary( data, syspars, transitvars, covpars, channel, \
                                                                ramp_type, verbose=1 )
    elif dataset_type=='phasecurve':
        pdb.set_trace()

    mbundle = parents.copy()
    for key in parents.keys():
        mbundle[key] = parents[key]
    mbundle['gp_logp'] = gp_logp

    return mbundle, gp_obj


def get_gp_freecov_mbundle( data, syspars, channel='ch1', ramp_type=None, \
                            orbpar_priors={}, dataset_type='primary' ):
    """
    """

    delT = pyhm.Uniform( 'delT', lower=-2./24., upper=2./24. )
    if dataset_type=='primary':
        if orbpar_priors=={}:
            orbpar_priors = None
        RpRs = pyhm.Uniform( 'RpRs', lower=0.5*syspars['RpRs'], upper=1.5*syspars['RpRs'] )
        if orbpar_priors==None:
            aRs = pyhm.Uniform( 'aRs', lower=0.5*syspars['aRs'], upper=1.5*syspars['aRs'] )
            b = pyhm.Uniform( 'b', lower=0.5*syspars['b'], upper=1.5*syspars['b'] )
        else:
            aRs = orbpar_priors['aRs']
            b = orbpar_priors['b']
        transitvars = { 'RpRs':RpRs, 'aRs':aRs, 'b':b, 'delT':delT } # these can be free or fixed
        parents, gp_logp, gp_obj = gp_freecov_model_primary( data, syspars, transitvars, channel, \
                                                             ramp_type, verbose=1 )
    elif dataset_type=='secondary':
        SecDepth = pyhm.Uniform( 'SecDepth', lower=-0.01, upper=0.01 )
        transitvars = { 'SecDepth':SecDepth, 'delT':delT } # these can be free or fixed
        parents, gp_logp, gp_obj = gp_freecov_model_secondary( data, syspars, transitvars, channel, \
                                                               ramp_type, verbose=1 )
    elif dataset_type=='phasecurve':
        pdb.set_trace()

    mbundle = parents.copy()
    for key in parents.keys():
        mbundle[key] = parents[key]
    mbundle['gp_logp'] = gp_logp

    return mbundle, gp_obj


def gp_fixedcov_model_primary( data, syspars, transitvars, covpars, channel, ramp_type, verbose=0 ):

    T0_approx = syspars['T0']
    ndat = len( data['flux'] )
    gp = gp_class.gp( which_type='full' )
    gp.mfunc = None
    gp.mpars = {}

    if ( channel=='ch1' )+( channel=='ch2' ):
        gp.xtrain = np.column_stack( [ data['tv'], data['xv'], data['yv'] ] )
        gp.cfunc = irac_kernels.txy_kernel_numexpr
        gp.cpars = { 'Axy':covpars['Axy'], 'Lxy':covpars['Lxy'], 'At':covpars['At'], 'Lt':covpars['Lt'] }
        gp.etrain = ( 1+covpars['beta']-1e-6 )*data['uncs']
    elif ( channel=='ch3' )+( channel=='ch4' ):
        # TODO = will use the ramp_type argument to determine appropriate cpars for kernel
        pdb.set_trace()
    else:
        pdb.set_trace()
    
    parents = transitvars
    cov_kwpars = gp.prep_fixedcov()
    @pyhm.stochastic( observed=True )
    def gp_logp( value=data['flux'], parents=parents ):
        def logp( value, parents=parents ):
            tp1 = time.time()
            syspars['RpRs'] = parents['RpRs']
            syspars['aRs'] = parents['aRs']
            syspars['b'] = parents['b']
            syspars['incl'] = np.rad2deg( np.arccos( syspars['b']/syspars['aRs'] ) )
            syspars['T0'] = T0_approx + parents['delT']
            tp2 = time.time()
            psignal = transit.ma02_aRs( data['bjd'], **syspars )
            tp3 = time.time()
            resids = value.flatten()/psignal.flatten()
            residsm = np.median( resids )
            resids -= residsm
            resids = np.reshape( resids, [ ndat, 1 ] )
            logp_value = gp.logp_fixedcov( resids=resids, kwpars=cov_kwpars )
            tp4 = time.time()
            if verbose>1:
                print '\nPrimary transit:'
                print 'RpRs={0:.5f}'.format( parents['RpRs'] )
                print 'delT={0:.1f} seconds'.format( parents['delT']*24*60*60 )
                print 'aRs={0:.3f}'.format( parents['aRs'] )
                print 'incl={0:.3f} deg'.format( syspars['incl'] )
                print 'logp={0:.3f}'.format( logp_value )
                tp5 = time.time()
                print 'Evaluation time = {0:.2f} sec'.format( tp4-tp1 )
            return logp_value

    return parents, gp_logp, gp


def gp_fixedcov_model_secondary( data, syspars, transitvars, covpars, channel, ramp_type, verbose=0 ):

    T0_approx = syspars['T0']
    ndat = len( data['flux'] )
    gp = gp_class.gp( which_type='full' )
    gp.mfunc = None
    gp.mpars = {}

    if ( channel=='ch1' )+( channel=='ch2' ):
        gp.xtrain = np.column_stack( [ data['tv'], data['xv'], data['yv'] ] )
        gp.cfunc = irac_kernels.txy_kernel_numexpr
        gp.cpars = { 'Axy':covpars['Axy'], 'Lxy':covpars['Lxy'], 'At':covpars['At'], 'Lt':covpars['Lt'] }
        gp.etrain = ( 1+covpars['beta']-1e-6 )*data['uncs']
    elif ( channel=='ch3' )+( channel=='ch4' ):
        # TODO = will use the ramp_type argument to determine appropriate cpars for kernel
        pdb.set_trace()
    else:
        pdb.set_trace()
    
    parents = transitvars
    cov_kwpars = gp.prep_fixedcov()
    @pyhm.stochastic( observed=True )
    def gp_logp( value=data['flux'], parents=parents ):
        def logp( value, parents=parents ):
            tp1 = time.time()
            syspars['SecDepth'] = parents['SecDepth']
            syspars['T0'] = T0_approx + parents['delT']
            tp2 = time.time()
            psignal = transit.ma02_aRs( data['bjd'], **syspars )
            tp3 = time.time()
            resids = value.flatten()/psignal.flatten()
            residsm = np.median( resids )
            resids -= residsm
            resids = np.reshape( resids, [ ndat, 1 ] )
            logp_value = gp.logp_fixedcov( resids=resids, kwpars=cov_kwpars )
            tp4 = time.time()
            if verbose>1:
                print '\nPrimary transit:'
                print 'SecDepth={0:.5f}'.format( parents['SecDepth'] )
                print 'delT={0:.1f} seconds'.format( parents['delT']*24*60*60 )
                print 'logp={0:.3f}'.format( logp_value )
                tp5 = time.time()
                print 'Evaluation time = {0:.2f} sec'.format( tp4-tp1 )
            return logp_value

    return parents, gp_logp, gp



def gp_freecov_model_primary( data, syspars, transitvars, channel, ramp_type, verbose=0 ):

    T0_approx = syspars['T0']
    ndat = len( data['flux'] )
    gp = gp_class.gp( which_type='full' )
    gp.mfunc = None
    gp.mpars = {}

    if ( channel=='ch1' )+( channel=='ch2' ):

        gp.xtrain = np.column_stack( [ data['tv'], data['xv'], data['yv'] ] )
        gp.cfunc = irac_kernels.txy_kernel_numexpr
        #At = 0
        At = pyhm.Gamma( 'At', alpha=1, beta=1e4 )
        iLt = pyhm.Uniform( 'iLt', lower=0, upper=1000 )
        #Axy = pyhm.Uniform( 'Axy', lower=0, upper=1000 )
        Axy = pyhm.Gamma( 'Axy', alpha=1, beta=1e4 )
        iLx = pyhm.Uniform( 'iLx', lower=0, upper=1000 )
        iLy = pyhm.Uniform( 'iLy', lower=0, upper=1000 )
        beta = pyhm.Uniform( 'beta', lower=-0.9, upper=2 )

        covpars = { 'At':At, 'iLt':iLt, 'Axy':Axy, 'iLx':iLx, 'iLy':iLy, 'beta':beta }
        parents = transitvars.copy()
        parents.update( covpars )
        @pyhm.stochastic( observed=True )
        def gp_logp( value=data['flux'], parents=parents ):
            def logp( value, parents=parents ):
                tp1 = time.time()
                syspars['RpRs'] = parents['RpRs']
                syspars['aRs'] = parents['aRs']
                syspars['b'] = parents['b']
                syspars['incl'] = np.rad2deg( np.arccos( syspars['b']/syspars['aRs'] ) )
                syspars['T0'] = T0_approx + parents['delT']
                tp2 = time.time()
                psignal = transit.ma02_aRs( data['bjd'], **syspars )
                tp3 = time.time()
                resids = value.flatten()/psignal.flatten()
                residsm = np.median( resids )
                resids -= residsm
                gp.cpars = { 'Axy':parents['Axy'], 'Lxy':np.array( [ 1./parents['iLx'], 1./parents['iLy'] ] ), \
                             'At':parents['At'], 'Lt':1./parents['iLt'] }
                gp.dtrain = np.reshape( resids, [ ndat, 1 ] )
                gp.etrain = ( 1+parents['beta']-1e-6 )*data['uncs']
                if 0:
                    plt.figure()
                    plt.errorbar(data['bjd'],value,yerr=gp.etrain,fmt='ok')
                    mu, sig = gp.predictive( xnew=gp.xtrain, enew=gp.etrain )
                    systematics = mu.flatten() + residsm
                    plt.plot(data['bjd'],systematics*psignal,'-r')
                    pdb.set_trace()
                logp_value = gp.logp_builtin()
                tp4 = time.time()
                if verbose>1:
                    print '\nPrimary transit:'
                    print 'Axy={0:.2f}percent, At={1:.0f}ppm, beta={2:.4f}'\
                          .format( parents['Axy']*(1e2), parents['At']*(1e6), parents['beta'] )
                    print 'Lx={0:.2f} pixels, Ly={1:.2f} pixels, Lt={2:.2f} minutes'\
                          .format( 1./parents['iLx'], 1./parents['iLy'], 1./parents['iLt'] )
                    print 'RpRs={0:.5f}'.format( parents['RpRs'] )
                    print 'delT={0:.1f} seconds'.format( parents['delT']*24*60*60 )
                    print 'aRs={0:.3f}'.format( parents['aRs'] )
                    print 'incl={0:.3f} deg'.format( syspars['incl'] )
                    print 'logp={0:.3f}'.format( logp_value )
                    tp5 = time.time()
                    print 'Evaluation time = {0:.2f} sec'.format( tp4-tp1 )
                    print len(value), T0_approx, 'kkkkkkkkk', parents['delT']
                return logp_value

    elif ( channel=='ch3' )+( channel=='ch4' ):

        pdb.set_trace() # todo 

    return parents, gp_logp, gp


def gp_freecov_model_secondary( data, syspars, transitvars, channel, ramp_type, verbose=0 ):

    T0_approx = syspars['T0']
    ndat = len( data['flux'] )
    gp = gp_class.gp( which_type='full' )
    gp.mfunc = None
    gp.mpars = {}

    if ( channel=='ch1' )+( channel=='ch2' ):

        gp.xtrain = np.column_stack( [ data['tv'], data['xv'], data['yv'] ] )
        gp.cfunc = irac_kernels.txy_kernel_numexpr
        #At = 0
        At = pyhm.Gamma( 'At', alpha=1, beta=1e4 )
        iLt = pyhm.Uniform( 'iLt', lower=0, upper=1000 )
        #Axy = pyhm.Uniform( 'Axy', lower=0, upper=1000 )
        Axy = pyhm.Gamma( 'Axy', alpha=1, beta=1e4 )
        iLx = pyhm.Uniform( 'iLx', lower=0, upper=1000 )
        iLy = pyhm.Uniform( 'iLy', lower=0, upper=1000 )
        #beta = pyhm.Uniform( 'beta', lower=-0.9, upper=2 )
        beta = pyhm.Uniform( 'beta', lower=0., upper=2 )

        covpars = { 'At':At, 'iLt':iLt, 'Axy':Axy, 'iLx':iLx, 'iLy':iLy, 'beta':beta }
        parents = transitvars.copy()
        parents.update( covpars )
        @pyhm.stochastic( observed=True )
        def gp_logp( value=data['flux'], parents=parents ):
            def logp( value, parents=parents ):
                tp1 = time.time()
                syspars['SecDepth'] = parents['SecDepth']
                syspars['T0'] = T0_approx + parents['delT']
                tp2 = time.time()
                psignal = transit.ma02_aRs( data['bjd'], **syspars )
                tp3 = time.time()
                resids = value.flatten()/psignal.flatten()
                residsm = np.median( resids )
                resids -= residsm
                gp.cpars = { 'Axy':parents['Axy'], 'Lxy':np.array( [ 1./parents['iLx'], 1./parents['iLy'] ] ), \
                             'At':parents['At'], 'Lt':1./parents['iLt'] }
                gp.dtrain = np.reshape( resids, [ ndat, 1 ] )
                gp.etrain = ( 1+parents['beta']-1e-6 )*data['uncs']
                if 0:
                    plt.figure()
                    plt.errorbar(data['bjd'],value,yerr=gp.etrain,fmt='ok')
                    mu, sig = gp.predictive( xnew=gp.xtrain, enew=gp.etrain )
                    systematics = mu.flatten() + residsm
                    plt.plot(data['bjd'],systematics*psignal,'-r')
                    pdb.set_trace()
                logp_value = gp.logp_builtin()
                tp4 = time.time()
                if verbose>1:
                    print '\nPrimary transit:'
                    print 'Axy={0:.2f}percent, At={1:.0f}ppm, beta={2:.4f}'\
                          .format( parents['Axy']*(1e2), parents['At']*(1e6), parents['beta'] )
                    print 'Lx={0:.2f} pixels, Ly={1:.2f} pixels, Lt={2:.2f} minutes'\
                          .format( 1./parents['iLx'], 1./parents['iLy'], 1./parents['iLt'] )
                    print 'SecDepth={0:.5f}'.format( parents['SecDepth'] )
                    print 'delT={0:.1f} seconds'.format( parents['delT']*24*60*60 )
                    print 'logp={0:.3f}'.format( logp_value )
                    tp5 = time.time()
                    print 'Evaluation time = {0:.2f} sec'.format( tp4-tp1 )
                    print len(value), T0_approx, 'kkkkkkkkk', parents['delT']
                return logp_value

    elif ( channel=='ch3' )+( channel=='ch4' ):

        pdb.set_trace() # todo 

    return parents, gp_logp, gp



def prep_data( lightcurve, data, target, channel, dataset_type, signal, syspars ):

    T0_lit = syspars['T0']

    bjd = data[:,0]
    x = data[:,1]
    y = data[:,2]
    noisepix = data[:,3]
    bg_ppix = data[:,4]
    flux = data[:,5]
    stdv = data[:,6]
    npb = data[:,7]
    goodbad = data[:,8]
    ixs = ( goodbad==1 )
    bjd = bjd[ixs]
    x = x[ixs]
    y = y[ixs]
    noisepix = noisepix[ixs]
    bg_ppix = bg_ppix[ixs]
    flux = flux[ixs]
    stdv = stdv[ixs]
    npb = npb[ixs]

    # Determine the transit/eclipse mid-time:
    if signal=='transit':
        syspars['tr_type'] = 'primary'
    elif ( signal=='eclipse' )+( signal=='eclipse_1' )+( signal=='eclipse_2' ):
        syspars['tr_type'] = 'secondary'
    T0_lit, toff = get_T0( bjd, T0_lit, syspars['P'], signal )

    # If the data is a phase observation, cut out a subsection
    # containing the transit/eclipse:
    if ( dataset_type!='transit' )*( dataset_type!='eclipse' ):
        ixs = cut_sub( bjd, T0_lit )
        bjd = bjd[ixs]
        x = x[ixs]
        y = y[ixs]
        noisepix = noisepix[ixs] 
        flux = flux[ixs]
        stdv = stdv[ixs]
        npb = npb[ixs]
    ndat = len( flux )
    tvar = bjd - bjd.min()
    tvar /= tvar.max()-tvar.min()
    sigw = np.sqrt( flux )/np.sqrt( npb )
    flux_med = np.median( flux )
    flux /= flux_med
    sigw /= flux_med
    x_med = np.median( x )
    x_std = np.std( x )
    xvar = ( x - x_med )/x_std
    y_med = np.median( y )
    y_std = np.std( y )
    yvar = ( y - y_med )/y_std

    z = {}
    z['bjd'] = bjd
    z['x'] = x
    z['y'] = y
    z['noisepix'] = noisepix
    z['bg_ppix'] = bg_ppix
    z['flux'] = flux
    z['sigw'] = sigw
    z['xvar'] = xvar
    z['yvar'] = yvar
    z['tvar'] = tvar
    z['T0_lit'] = T0_lit
    z['toff'] = toff

    return z
