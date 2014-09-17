import pdb, os, sys, warnings, time
import fitsio
import atpy
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.interpolate
import scipy.spatial
from pyraf import iraf # MAKE THIS OPTIONAL
iraf.digiphot()
iraf.apphot()

"""

Top-level routines that constitute the basic
steps of the pipeline reduction:
  read_headers()
  centroids()
  preclean()
  ap_phot()
  bg_subtract()
  save_table()

Routines that actually contain the implementation
of the various centroiding algorithms:
  fluxweight_centroid()
  gauss1d_centroid()
  gauss2d_centroid()
  iraf_centroid()

Routines that extract background pixels according
to the various methods:
  get_annulus_circle_pixs()
  get_corner_pixs()
  get_custom_mask_pixs()

And utility functions:
  cut_subarray()

"""


def read_headers( irac ):
    """
    Reads information from the FITS headers, including the
    readnoise, gain etc. Also creates an array containing
    the BJDs at mid-exposure, based on information contained
    in the headers.
    """
    
    nfits = len( irac.fitsfiles )
    midtimes = np.empty( 0, dtype=float )
    exptimes = np.empty( 0, dtype=float )
    gains = np.empty( 0, dtype=float )
    readnoises = np.empty( 0, dtype=float )
    fluxconvs = np.empty( 0, dtype=float )
    obsblocks = np.empty( 0, dtype=int )
    print '\nReading headers for frame mid-times, readnoise, gain etc'

    for i in range( nfits ):

        fitsfile = os.path.join( irac.ddir, irac.fitsfiles[i] )
        if irac.verbose>1:
            print fitsfile
        hdu = fitsio.FITS( fitsfile, 'r' )
        header = hdu[0].read_header()
        hdu.close()
        if header['NAXIS']==3:
            naxis3 = header['NAXIS3']
        else:
            naxis3 = 1
        framtime = header[irac.framtime_kw]/60./60./24.
        midtimes_i = np.zeros( naxis3 )
        exptimes_i = np.zeros( naxis3 )
        readnoises_i = np.zeros( naxis3 )
        gains_i = np.zeros( naxis3 )
        fluxconvs_i = np.zeros( naxis3 )

        # Observing block:
        bjd_start = header[irac.bjd_kw] + 2400000.5
        for j in range( naxis3 ):

            # Calculate mid-exposure times and read values
            # for integration/exposure time, readnoise, gain
            # and MJy-to-electrons conversion factors:
            midtimes_i[j] = bjd_start + (0.5 + j)*framtime
            exptimes_i[j] = header[irac.exptime_kw]
            readnoises_i[j] = header[irac.readnoise_kw]
            gains_i[j] = header[irac.gain_kw]
            fluxconvs_i[j] = header[irac.fluxconv_kw]

        midtimes = np.concatenate( [ midtimes, midtimes_i ] )
        exptimes = np.concatenate( [ exptimes, exptimes_i ] )
        readnoises = np.concatenate( [ readnoises, readnoises_i ] )
        gains = np.concatenate( [ gains, gains_i ] )
        fluxconvs = np.concatenate( [ fluxconvs, fluxconvs_i ] )
        obsblocks_i = i + np.zeros( naxis3, dtype=int )
        obsblocks = np.concatenate( [ obsblocks, obsblocks_i ] )

    # Take unique values of exptime, readnoise, gain and fluxconv
    # rather than carrying value for each image, because these
    # values should basically be the same for each image:
    irac.bjd = midtimes
    if np.diff( exptimes ).max()==0:
        irac.exptime = exptimes[0]
    else:
        warnings.warn( 'exptimes not all identical - taking median' )
        irac.exptime = np.median( extimes )
    if np.diff( readnoises ).max()==0:
        irac.readnoise = readnoises[0]
    else:
        warnings.warn( 'readnoises not all identical - taking median' )
        irac.readnoise = np.median( readnoises )
    if np.diff( gains ).max()==0:
        irac.gain = gains[0]
    else:
        warnings.warn( 'gains not all identical - taking median' )
        irac.gain = np.median( gains )
    if np.diff( fluxconvs ).max()==0:
        irac.fluxconv = fluxconvs[0]
    else:
        warnings.warn( 'fluxconvs not all identical - taking median' )
        irac.fluxconv = np.median( fluxconvs )
    irac.frame_obsblocks = obsblocks
    print 'Done.'

    return None


def centroids( irac ):
    """
    Calculates the centroids for each image. Requires an
    attribute 'centroid_kwargs' in the form of a dictionary
    containing a string specifying which centroiding method
    is to be used along with any required keyword arguments.
    Also flags as bad images where the peak flux changes by
    more than 50% from the previous image.
    """

    # Count the number of sub-images stored
    # within each fits file:
    irac.nfits = len( irac.fitsfiles )
    irac.nsub = np.zeros( irac.nfits, dtype=int )
    for i in range( irac.nfits ):
        if irac.verbose>1:
            print 'Reading in fits file {0} of {1}...'\
                  .format( i+1, irac.nfits )
        hdu = fitsio.FITS( irac.fitsfiles[i], 'r' )
        #dims = hdu[0].info['dims'] # worked with fitsio v0.9.0
        dims = hdu[0].get_info()['dims'] # works with fitsio v0.9.5
        hdu.close()
        if len( dims )==2:
            irac.nsub[i] = 1
        elif len( dims )==3:
            irac.nsub[i] = dims[0]
        else:
            pdb.set_trace() #shouldn't happen
    irac.nframes = np.sum( irac.nsub )

    # Start off assuming all frames are good,
    # unless told otherwise:
    if irac.goodbad==None:
        irac.goodbad = np.ones( irac.nframes )

    # Number of frames flagged as good
    # to start with:
    nstart = np.sum( irac.goodbad )

    # Determine whether we'll be using 4- or 5-sigma
    # clipping depending on the number of frames x pixels
    # we have:
    if np.sum( irac.goodbad )>1e5:
        nsigma_clip = 5.
    else:
        nsigma_clip = 4.

    # Arguments for centroiding algorithm:
    method = irac.centroid_kwargs['method']
    boxwidth = irac.centroid_kwargs['boxwidth']
    maxshift = 2
    if boxwidth%2==0:
        boxwidth += 1
    if method=='fluxweight':
        irac.xy_fluxweight = np.zeros( [ irac.nframes, 2 ] )
        irac.xy_method = 'fluxweight'
    elif method=='gauss1d':
        irac.xy_gauss1d = np.zeros( [ irac.nframes, 2 ] )
        irac.xy_method = 'gauss1d'
    elif method=='gauss2d':
        irac.xy_gauss2d = np.zeros( [ irac.nframes, 2 ] )
        irac.xy_method = 'gauss2d'
    elif method=='iraf':
        irac.xy_iraf = np.zeros( [ irac.nframes, 2 ] )
        irac.xy_method = 'iraf'

    # Guess xy-coordinates for first frame:
    xguess = irac.init_xy[0]
    yguess = irac.init_xy[1]
    xyguess = np.array( [ xguess, yguess ] )
    if irac.verbose>0:
        print '\nCalculating centroids within a {0}x{0} pixel box'\
              .format( boxwidth )

    # Apply centroiding algorithm one image at a time:
    suspicious_ks = []
    for i in range( irac.nfits ):

        fitsfile = os.path.join( irac.ddir, irac.fitsfiles[i] )
        if irac.verbose>1:
            print fitsfile

        hdu = fitsio.FITS( irac.fitsfiles[i], 'r' )
        #data = hdu[0].read_image() # worked with fitsio v0.9.0
        data = hdu[0].read() # works with fitsio v0.9.5
        hdu.close()

        # Account for the possibility that the fits
        # image could be comprised of subframes:
        for j in range( irac.nsub[i] ):

            if irac.nsub[i]==1:
                fullarray = data
                k = i
            else:
                fullarray = data[j,:,:]
                k = np.sum( irac.nsub[:i] ) + j

            if irac.verbose>1:
                print 'Centroiding frame {0} of {1} using {2}...'\
                      .format( k+1, irac.nframes, method )

            # Check if current frame has already been
            # flagged as bad:
            if irac.goodbad[k]==0:
                continue

            # Before proceeding further, estimate and subtract
            # the sky background:
            ny, nx = np.shape( fullarray )
            xfull = np.arange( nx )
            yfull = np.arange( ny )
            xmesh, ymesh = np.meshgrid( xfull, yfull )

            ## NOTE: This is untested, but I don't think that we want
            ## to just blindly take the pixel coordinates of the brightest
            ## pixel in each frame; instead, we want to use whatever is
            ## currently recorded as xyguess; if this is the first frame,
            ## xyguess will be whatever was entered manually; if it not the
            ## first frame, then it will be whatever the xy coordinates
            ## determine in the last frame that was not flagged as bad were
            #yix, xix = np.unravel_index( np.argmax( fullarray ), \
            #                             np.shape( fullarray ) )
            #xguess = xfull[xix]
            #yguess = yfull[yix]
            if irac.bg_kwargs['method']=='annulus_circle':
                bg_pixs = get_annulus_circle_pixs( fullarray, xmesh, ymesh, xyguess, \
                                                   irac.ap_radius, \
                                                   irac.bg_kwargs['annulus_inedge'], \
                                                   irac.bg_kwargs['annulus_width'] )
            elif irac.bg_kwargs['method']=='corners':
                bg_pixs = get_corner_pixs( fullarray, xmesh, ymesh, xyguess, \
                                           irac.ap_radius, \
                                           irac.bg_kwargs['ncorner'] )
            elif irac.bg_kwargs['method']=='custom_mask':
                bg_pixs = get_custom_mask_pixs( fullarray, xmesh, ymesh, xyguess, \
                                                irac.ap_radius, \
                                                irac.bg_kwargs['mask_array'] )
            else:
                pdb.set_trace() # no other methods implemented yet
                
            # Two-pass sigma clipping:
            ixs = np.isfinite( bg_pixs )
            bg_pixs = bg_pixs[ixs]
            bg_med = np.median( bg_pixs )
            bg_stdv = np.std( bg_pixs )
            delta_sigmas = ( bg_pixs - bg_med )/bg_stdv
            ixs_keep_1 = ( delta_sigmas < nsigma_clip )
            bg_med = np.median( bg_pixs[ixs_keep_1] )
            bg_stdv = np.std( bg_pixs[ixs_keep_1] )
            delta_sigmas = ( bg_pixs[ixs_keep_1] - bg_med )/bg_stdv
            ixs_keep_2 = ( delta_sigmas < nsigma_clip )
            bg_pixs = bg_pixs[ixs_keep_1][ixs_keep_2]

            # Extract a single value for the background:
            if irac.bg_kwargs['value']=='median':
                bg = np.median( bg_pixs )
            elif irac.bg_kwargs['value']=='mean':
                bg = np.mean( bg_pixs )
            else:
                pdb.set_trace() # none implemented yet

            # Subtract the background:
            fullarray -= bg

            # Identify the box surrounding the starting guess xy-coordinates:
            subarray, xsub, ysub = cut_subarray( fullarray, xguess, yguess, boxwidth )
            xrefined, yrefined = fluxweight_centroid( subarray, xsub, ysub )
            subarray, xsub, ysub = cut_subarray( fullarray, xrefined, yrefined, boxwidth )
            # NOTE: non-finite pixel values might screw things up here...
            # might be necessary to deal with this sort of thing at some point...

            if ( k==0 ):
                # If it's the first frame, locate the peak 
                # pixel in the initial subarray and use this
                # as the center pixel in the refined subarray:
                peak_prev = np.max( subarray )
            else:
                peak_curr = np.max( subarray )
                delpeak = abs( peak_curr/float( peak_prev ) - 1. )
                if abs( peak_prev - peak_curr )>0.5*abs( peak_prev ):
                    print '\nDiscarding suspicious frame due to >50%'
                    print 'frame-to-frame change in peak pixel value:'
                    fitsfile_name = os.path.basename( fitsfile )
                    if irac.nsub[i]==1:
                        print '--> frame {0}'.format( fitsfile_name )
                    else:
                        print '--> frame {0} in {1}'.format( j+1, fitsfile_name )
                    print 'due to >50% frame-to-frame change in peak pixel value'
                    irac.goodbad[k] = 0
                    peak_prev = peak_curr
                    suspicious_ks += [ k ]
                else:
                    peak_prev = peak_curr

            if method=='fluxweight':
                x0, y0 = fluxweight_centroid( subarray, xsub, ysub )
                irac.xy_fluxweight[k,0] = x0
                irac.xy_fluxweight[k,1] = y0
            elif method=='gauss1d':
                x0, y0, wx, wy = gauss1d_centroid( subarray, xsub, ysub, irac.channel )
                irac.xy_gauss1d[k,0] = x0
                irac.xy_gauss1d[k,1] = y0
            elif method=='gauss2d':
                x0, y0, wx, wy = gauss2d_centroid( subarray, xsub, ysub, irac.channel )
                irac.xy_gauss2d[k,0] = x0
                irac.xy_gauss2d[k,1] = y0
            elif method=='iraf':
                xguess, yguess = fluxweight_centroid( subarray, xsub, ysub )
                xyguess = np.array( [ xguess, yguess ] )
                x0, y0 = iraf_centroid( irac.adir, fitsfile, xyguess, j, boxwidth )
                irac.xy_iraf[k,0] = x0
                irac.xy_iraf[k,1] = y0
                    
            # Check how much the pointing has changed since previous frame:
            npixshift_max = 2
            if ( abs( x0 - xguess )<npixshift_max ) and ( abs( y0 - yguess )<npixshift_max ):
                # If it was a small amount, use the current
                # guess as the starting guess for next frame:
                xguess = x0
                yguess = y0
                xyguess = np.array( [ x0, y0 ] )
            else:
                # Otherwise, assume it was an anomaly and
                # use the value from the previous frame as
                # the starting guess for the next frame:
                warnings.warn( '>{0} pixel jump in centroid - something probably wrong...'\
                               .format( npixshift_max ) )

    if len( suspicious_ks )>1:
        print '\nNumber of images between successive suspicious images:'
        print np.diff( np.array( suspicious_ks ) )
        print '\n'

    # Stand-alone text files containing the centroids:
    header = 'x, y'
    output_fmt = '%.6f %.6f'
    if method=='fluxweight':
        ofilename = os.path.join( irac.adir, 'xy_fluxweight.coords' )
        np.savetxt( ofilename, irac.xy_fluxweight, fmt=output_fmt, header=header )
    elif method=='gauss1d':
        ofilename = os.path.join( irac.adir, 'xy_gauss1d.coords' )
        np.savetxt( ofilename, irac.xy_gauss1d, fmt=output_fmt, header=header )
    elif method=='gauss2d':
        ofilename = os.path.join( irac.adir, 'xy_gauss2d.coords' )
        np.savetxt( ofilename, irac.xy_gauss2d, fmt=output_fmt, header=header )
    elif method=='iraf':
        ofilename = os.path.join( irac.adir, 'xy_iraf.coords' )
        np.savetxt( ofilename, irac.xy_iraf, fmt=output_fmt, header=header )
    if np.sum( irac.goodbad )<nstart:
        print 'Flagged {0:d} of {1:d} frames as bad due to large pointing shifts or other problems'\
              .format( int( nstart-np.sum( irac.goodbad ) ), irac.nframes )

    if irac.verbose>0:
        print 'Saved xy coords in {0}'.format( ofilename )
        print 'Done.'
    
    return None

    
def fluxweight_centroid( subarray, xsub, ysub ):
    """
    Used to also have fluxweight2d, but realised that
    is mathematically identical and nearly 50% slower.
    """

    # Negative flux values bias the output, so
    # set minimum value to be zero:
    subarray -= subarray.min()
    # Calculate the flux-weighted mean:
    marginal_x = np.sum( subarray, axis=0 )
    marginal_y = np.sum( subarray, axis=1 )
    fluxsumx = np.sum( marginal_x )
    x0 = np.sum( xsub*marginal_x )/fluxsumx
    fluxsumy = np.sum( marginal_y )
    y0 = np.sum( ysub*marginal_y )/fluxsumy

    return x0, y0


def gauss1d_centroid( subarray, xsub, ysub, channel ):
    """
    NOTE: Unfortunately this routine is >10 times slower than the IDL gcntrd.pro routine.
    One obvious thing is that gcntrd fixes the width of the Gaussian (i.e. doesn't fit
    for it). But probably more importantly, it seems to implement an analytic solution
    for the least squares result, whereas I use the scipy optimisation routines. I haven't
    yet worked out how the gcntrd routine is actually doing this, so I'm sticking with
    the scipy optimisation routines for now.
    """

    # Negative flux values bias the output, so
    # set minimum value to be zero:
    subarray -= subarray.min()

    marg_x = np.sum( subarray, axis=0 )
    marg_y = np.sum( subarray, axis=1 )
    x0 = xsub[ np.argmax( marg_x ) ]
    y0 = ysub[ np.argmax( marg_y ) ]

    A0_x = marg_x.min()
    A0_y = marg_y.min()    
    B0_x = marg_x.max() - A0_x
    B0_y = marg_y.max() - A0_y
    if channel==1:
        w0 = 0.5*1.66/1.221
    elif channel==2:
        w0 = 0.5*1.72/1.213
    elif channel==3:
        w0 = 0.5*1.88/1.222
    elif channel==4:
        w0 = 0.5*1.98/1.220
    elif channel==None:
        w0 = 0.5*1.5/1.220
    else:
        print '{0} for channel not understood'.format( channel )
        pdb.set_trace()

    nf = 10
    mx = int( len( xsub )*nf )
    my = int( len( ysub )*nf )
    xsub_f = np.arange( xsub.min()-0.5, xsub.max()+0.5, 1./nf )
    ysub_f = np.arange( ysub.min()-0.5, ysub.max()+0.5, 1./nf )

    def func_x( pars ):
        A = pars[0]
        B = pars[1]
        mu = pars[2]
        w = pars[3]
        gaussfit_f = A + B * np.exp( -0.5*( ( ( ( xsub_f - mu )/w )**2. ) ) )
        gaussfit = rebin1d( gaussfit_f, nf )
        return marg_x - gaussfit
    pars0 = np.array( [ A0_x, B0_x, x0, w0 ] )
    pars_fit_x = scipy.optimize.leastsq( func_x, pars0, ftol=1e-3 )[0]
    mux_fit = pars_fit_x[2]
    wx_fit = pars_fit_x[3]
    
    def func_y( pars ):
        A = pars[0]
        B = pars[1]
        mu = pars[2]
        w = pars[3]
        gaussfit_f = A + B * np.exp( -0.5*( ( ( ( ysub_f - mu )/w )**2. ) ) )
        gaussfit = rebin1d( gaussfit_f, nf )
        return marg_y - gaussfit
    pars0 = np.array( [ A0_y, B0_y, y0, w0 ] )
    pars_fit_y = scipy.optimize.leastsq( func_y, pars0, ftol=1e-3 )[0]
    muy_fit = pars_fit_y[2]
    wy_fit = pars_fit_y[3]

    return mux_fit, muy_fit, wx_fit, wy_fit


def gauss2d_centroid(  subarray, xsub, ysub, channel ):
    """
    """

    # Negative flux values bias the output, so
    # set minimum value to be zero:
    subarray -= subarray.min()

    fluxsum = np.sum( subarray )
    xmesh, ymesh = np.meshgrid( xsub, ysub )
    mux0 = np.sum( xmesh*subarray )/fluxsum
    muy0 = np.sum( ymesh*subarray )/fluxsum
    A0 = np.median( subarray )
    B0 = np.max( subarray )
    if channel==1:
        w0 = 0.5*1.66/1.221
    if channel==2:
        w0 = 0.5*1.72/1.213
    if channel==3:
        w0 = 0.5*1.88/1.222
    if channel==4:
        w0 = 0.5*1.98/1.220
    elif channel==None:
        w0 = 0.5*1.5/1.220
    rot0 = 0.0
    xvals = xmesh.flatten()
    yvals = ymesh.flatten()
    zdata = subarray.flatten()
    def func( pars ):
        A = pars[0] # background level
        B = pars[1] # peak amplitude
        wx = pars[2] # first axis width
        wy = pars[3] # second axis width
        mux = pars[4] # horizontal mean coordinate
        muy = pars[5] # vertical mean coordinate
        zmodel = A + B*np.exp( -0.5*( ( ( xvals - mux )/wx)**2. + \
                                      ( ( yvals - muy )/wy)**2. ) )
        return zdata - zmodel
    pars0 = np.array( [ A0, B0, w0, w0, mux0, muy0 ] )
    pars_fit = scipy.optimize.leastsq( func, pars0, ftol=1e-2 )[0]
    wx_fit = pars_fit[2]
    wy_fit = pars_fit[3]
    mux_fit = pars_fit[4]
    muy_fit = pars_fit[5]

    return mux_fit, muy_fit, wx_fit, wy_fit

def gauss2d_centroid_WITH_ROTATION(  subarray, xsub, ysub, channel ):
    """
    """

    # Negative flux values bias the output, so
    # set minimum value to be zero:
    subarray -= subarray.min()

    fluxsum = np.sum( subarray )
    xmesh, ymesh = np.meshgrid( xsub, ysub )
    mux0 = np.sum( xmesh*subarray )/fluxsum
    muy0 = np.sum( ymesh*subarray )/fluxsum
    A0 = np.median( subarray )
    B0 = np.max( subarray )
    if channel==1:
        w0 = 0.5*1.66/1.221
    if channel==2:
        w0 = 0.5*1.72/1.213
    if channel==3:
        w0 = 0.5*1.88/1.222
    if channel==4:
        w0 = 0.5*1.98/1.220
    elif channel==None:
        w0 = 0.5*1.5/1.220
    rot0 = 0.0
    xvals = xmesh.flatten()
    yvals = ymesh.flatten()
    zdata = subarray.flatten()
    def func( pars ):
        A = pars[0] # background level
        B = pars[1] # peak amplitude
        wx = pars[2] # first axis width
        wy = pars[3] # second axis width
        mux = pars[4] # horizontal mean coordinate
        muy = pars[5] # vertical mean coordinate
        rot = pars[6] # rotation of axes relative to vertical-horizontal
        sinrot = np.sin( rot )
        cosrot = np.cos( rot )            
        sin2rot = np.sin( 2*rot )
        a = ( cosrot**2. )/( 2*( wx**2. ) ) + \
            ( sinrot**2. )/( 2*( wy**2. ) )
        b = ( -sin2rot )/( 4*( wx**2. ) ) + \
            ( sin2rot )/( 4*( wy**2. ) )
        c = ( sinrot**2. )/( 2*( wx**2. ) ) + \
            ( cosrot**2. )/( 2*( wy**2. ) )
        zmodel = A + B*np.exp( -( a*( ( xvals-mux )**2. ) + \
                                  2*b*( xvals-mux )*( yvals-muy ) + \
                                  c*( ( yvals-muy )**2. ) ) )
        return zdata - zmodel
    pars0 = np.array( [ A0, B0, w0, w0, mux0, muy0, rot0 ] )
    pars_fit = scipy.optimize.leastsq( func, pars0, ftol=1e-2 )[0]
    wx_fit = pars_fit[2]
    wy_fit = pars_fit[3]
    mux_fit = pars_fit[4]
    muy_fit = pars_fit[5]

    return mux_fit, muy_fit, wx_fit, wy_fit


def iraf_centroid( adir, fitsfile, xymid, jth_sub, boxwidth ):
    """
    Uses IRAF implementation of the fluxweight method for
    determining the centroid coordinates. This is done by
    setting the calgorithm parameter to 'centroid'.
    """

    # Turn interaction off:
    iraf.phot.interactive = 'no'
    iraf.phot.verify = 'no'

    # Prepare algorithm settings:
    iraf.centerpars.cbox = boxwidth
    #iraf.centerpars.calgorithm = 'gauss' # fits 1D gaussians to marginal profiles;
                                          # requires FWHMs of gaussians to be specified
    iraf.centerpars.calgorithm = 'centroid' # weighted means of marginals

    # Save the initial guess coordinates in an
    # external file, as required by IRAF:
    coord_tempfilename = os.path.join( adir, 'temp.coords' )
    coord_tempfile = open( coord_tempfilename, 'w' )
    coord_string = '{0:.2f} {1:.2f}'.format( xymid[0]+0.5, xymid[1]+0.5 )
    coord_tempfile.write( coord_string )
    coord_tempfile.close()

    # Prepare the filename that the IRAF output
    # will be sent to:
    mag_tempfilename = os.path.join( adir, 'temp.mag' )
    if os.path.isfile( mag_tempfilename ):
        os.remove( mag_tempfilename )

    # Run the IRAF centroiding algorithm:
    ext = '[*,*,{0}]'.format( int( jth_sub + 1 ) )
    fitsfile = fitsfile+ext
    iraf.phot( image=fitsfile, coords=coord_tempfilename, output=mag_tempfilename )

    # Extract the centroid result from the IRAF
    # output and save to another text file:
    coord_newfilename = os.path.join( adir, 'new.coords' )
    coord_newfile = open( coord_newfilename, 'w' )
    iraf.txdump( mag_tempfilename, fields='xcenter,ycenter', expr='yes', Stdout=coord_newfile )

    # Extract the centroid result:
    d = np.loadtxt( coord_newfilename )
    x0 = d[0] - 0.5
    y0 = d[1] - 0.5

    # Tidy up temporary files:
    os.remove( coord_newfilename )
    os.remove( mag_tempfilename )
    os.remove( coord_tempfilename )

    return x0, y0

    
def preclean( irac, iters=2 ):
    """
    NEED TO DECIDE WHERE THIS FITS INTO THE OVERALL PIPELINE.
    FOR INSTANCE, IT MAKES SENSE FOR IT TO BE RUN AFTER THE
    CENTROIDING, BUT PERHAPS IT THEN MAKES SENSE FOR THE
    CENTROIDING TO BE RE-RUN AGAIN ... POSSIBLY NOT WORTH IT.
    """

    # Start off assuming all frames are good,
    # unless told otherwise:
    if irac.goodbad==None:
        irac.goodbad = np.ones( irac.nframes )

    nslide = 30

    # Number of frames flagged as good
    # to start with:
    nstart = np.sum( irac.goodbad )

    # Determine whether we'll be using 4- or 5-sigma
    # clipping depending on the number of frames x pixels
    # we have:
    if np.sum( irac.goodbad )>1e5:
        nsigma_clip = 5.
    else:
        nsigma_clip = 4.
    print '\nPrecleaning xy shifts: {0} iterations of {1}-sigma clipping'\
          .format( iters, nsigma_clip )
    print '(centroids over {0} unflagged frames)'\
          .format( int( np.sum( irac.goodbad ) ) )

    # Retrieve the xy chip coordinates
    # that have already been determined
    # by the centroids() routine:
    print '\nUsing centroids determined by {0} method...'\
          .format( irac.xy_method )
    if irac.xy_method=='fluxweight':
        xy = irac.xy_fluxweight
    elif irac.xy_method=='gauss1d':
        xy = irac.xy_gauss1d
    elif irac.xy_method=='gauss2d':
        xy = irac.xy_gauss2d
    elif irac.xy_method=='iraf':
        xy = irac.xy_iraf
    else:
        raise AttributeError( 'xy_method attribute not recognised' )

    # Slide across all frames in the dataset,
    # iterating more than once if requested:
    ix1 = 0
    ix2 = nslide
    for iteration in range( iters ):
        for i in range( irac.nfits ):
            for j in range( irac.nsub[i] ):

                # Work out the current image number:
                k = np.sum( irac.nsub[:i] ) + j
                if irac.goodbad[k]==0:
                    continue

                if ( k<nslide )*( irac.nframes>nslide ):
                    # Less trailing:
                    ix1 = 0
                    ix2 = k+nslide+1
                elif ( k>=nslide )*( k<=irac.nframes-nslide ):
                    # Equal trailing and ahead:
                    ix1 = k - nslide
                    ix2 = k + nslide + 1
                elif ( k>irac.nframes-nslide )*( irac.nframes>nslide ):
                    # Less ahead
                    ix1 = k - nslide
                    ix2 = irac.nframes
                else:
                    pdb.set_trace() # shouldn't happen

                # xy values in block behind current point:
                xybehind = xy[ix1:k,:]
                gbbehind = ( irac.goodbad[ix1:k]==1 )

                # xy values in block ahead of current point:
                xyahead = xy[k+1:ix2+1,:]
                gbahead = ( irac.goodbad[k+1:ix2]==1 )

                # Combine the behind+ahead surrounding points:
                xycompare = np.vstack( [ xybehind, xyahead ] )
                gbcompare = np.concatenate( [ gbbehind, gbahead ] )
                xycompare = xycompare[ gbcompare, : ]

                # Calculate the median and scatter of surrounding points:
                xymed = np.median( xycompare, axis=0 )
                xystdv = np.std( xycompare, axis=0 )
                nsigma = np.max( np.abs( xy[k,:] - xymed )/xystdv )

                # If current point is outlier, flag it as bad:
                if nsigma>=nsigma_clip:
                    if irac.verbose>0:
                        if irac.nsub[i]==1:
                            print '--> discarding frame {0}'\
                                  .format( os.path.basename( irac.fitsfiles[i] ) )
                        else:
                            print '--> discarding frame {0} in {1}'\
                                  .format( j+1, os.path.basename( irac.fitsfiles[i] ) )
                    irac.goodbad[k] = 0

    nremoved_xy = nstart - np.sum( irac.goodbad )
    if nremoved_xy>0:
        if nremoved_xy==1:
            frame_str = 'frame'
        else:
            frame_str = 'frames'
        print 'Removing {0:d} {1} due to large xy shifts...'\
              .format( int( nremoved_xy ), frame_str )
    else:
        print 'No frames removed due to large xy shifts...'

    # Second, identify frames where an anomalous
    # pixel count occurs anywhere within a subarray
    # containing the photometry aperture:
    ixs = ( irac.goodbad==1 )
    xy = xy[ixs,:]
    halfbox = irac.ap_radius
    xl = int( np.floor( np.min( xy[:,0] ) - halfbox ) )
    xu = int( np.ceil( np.max( xy[:,0] ) + halfbox ) )
    yl = int( np.floor( np.min( xy[:,1] ) - halfbox ) )
    yu = int( np.ceil( np.max( xy[:,1] ) + halfbox ) )

    # Read in the first image:
    hdu = fitsio.FITS( irac.fitsfiles[0], 'r' )
    #data = hdu[0].read_image() # worked with fitsio v0.9.0
    data = hdu[0].read() # works with fitsio v0.9.5
    if irac.nsub[0]==1:
        fullarray0 = data
    else:
        fullarray0 = data[0,:,:]
    hdu.close()

    # Account for possibility of subarray being
    # right near the chip edge:
    naxis1 = np.shape( fullarray0 )[1]
    naxis2 = np.shape( fullarray0 )[0]
    if xl<0: xl = 0
    if yl<0: yl = 0
    if xu<0: xu = 0
    if yu<0: yu = 0
    if xl>naxis1-1: xl = naxis1 - 1
    if yl>naxis2-1: yl = naxis2 - 1
    if xu>naxis1-1: xu = naxis1 - 1
    if yu>naxis2-1: yu = naxis2 - 1

    # Cut the subarray for the first frame and
    # determine an appropriate sigma clip:
    frame0_sub = fullarray0[yl:yu+1,xl:xu+1]
    naxis1_sub = np.shape( frame0_sub )[1]
    naxis2_sub = np.shape( frame0_sub )[0]
    npixs_tot = naxis1_sub*naxis2_sub*np.sum( irac.goodbad )
    if npixs_tot<2e6:
        nsigma_clip = 4.
    else:
        nsigma_clip = 5.
    print '\nPrecleaning pixel values: {0} iterations of {1}-sigma clipping'\
          .format( iters, nsigma_clip )
    print '({0} pixels over {1} unflagged frames)'\
          .format( int( npixs_tot ), int( np.sum( irac.goodbad ) ) )

    # Iterate the process more than once if requested:
    nstart = np.sum( irac.goodbad )
    for iteration in range( iters ):

        # Initialise the train of subarrays that will be shifted across
        # one frame at a time:
        slide_block = np.zeros( [ naxis2_sub, naxis1_sub, 2*nslide + 1 ] )
        slide_gbs = np.zeros( 2*nslide + 1 )
        cs = np.cumsum( irac.nsub )
        ix = np.arange( irac.nfits )[cs>=nslide+1][0]
        break_loop = False
        for i in range( ix+1 ):
            hdu = fitsio.FITS( irac.fitsfiles[i], 'r' )
            #fits_data_i = hdu[0].read_image() # worked with fitsio v0.9.0
            fits_data_i = hdu[0].read() # works with fitsio v0.9.5
            hdu.close()
            for j in range( irac.nsub[i] ):
                k = np.sum( irac.nsub[:i] ) + j
                if irac.nsub[i]==1:
                    fullarrayk = fits_data_i
                else:
                    fullarrayk = fits_data_i[j,:,:]
                slide_block[:,:,k+nslide] = fullarrayk[yl:yu+1,xl:xu+1]
                slide_gbs[k+nslide] = irac.goodbad[k]
                if k==nslide:
                    j_lead = j
                    break_loop = True
                    break
            if break_loop==True:
                i_lead = i
                break

        # Now shift the train across one frame at a time, identifying
        # all frames that contain anomalous pixel counts and flagging
        # them as bad:
        for i in range( irac.nfits ):

            hdu = fitsio.FITS( irac.fitsfiles[i], 'r' )
            #data_i = hdu[0].read_image() # worked with fitsio v0.9.0
            data_i = hdu[0].read() # works with fitsio v0.9.5
            hdu.close()

            for j in range( irac.nsub[i] ):

                # The train sliding is done up here at the top of the 
                # loop, unless we're on the first frame:
                k = np.sum( irac.nsub[:i] ) + j
                if ( k>0 ):
                    # Slide the train across a frame:
                    slide_block[:,:,0:2*nslide] = slide_block[:,:,1:2*nslide+1]
                    slide_gbs[0:2*nslide] = slide_gbs[1:2*nslide+1]
                    # Find the next frame in the train, assuming there
                    # are some frames ahead of the leading edge:
                    if k+nslide+1<irac.nframes:
                        if irac.nsub[i_lead]==1:
                            # It's easy to move ahead if there are no
                            # sub-images within the current fits file:
                            i_lead_new += 1
                        else:
                            # If the current fits file is divided
                            # into sub-images, we must check if there
                            # are any ahead of the current sub-image:
                            if j_lead<irac.nsub[i_lead]-1:
                                i_lead_new = i_lead
                                j_lead += 1
                            else:
                                i_lead_new += 1
                                j_lead = 0
                        # Only read in a new fits file if it's not
                        # the one we already have open:
                        if i_lead_new!=i_lead:
                            hdu = fitsio.FITS( irac.fitsfiles[i_lead_new], 'r' )
                            data_lead = hdu[0].read_image()
                            #data_lead = hdu[0].read_image() # worked with fitsio v0.9.0
                            data_lead = hdu[0].read() # works with fitsio v0.9.5
                            hdu.close()
                            i_lead = i_lead_new
                        else:
                            data_lead = data_i
                        # Define the updated leading image
                        if irac.nsub[i_lead]==1:
                            fullarray_lead = data_lead
                        else:
                            fullarray_lead = data_lead[j_lead,:,:]
                        # Trim the leading image and add it to the
                        # sliding train:
                        k_lead = np.sum( irac.nsub[:i_lead] ) + j_lead
                        slide_block[:,:,2*nslide] = fullarray_lead[yl:yu+1,xl:xu+1]
                        slide_gbs[2*nslide] = irac.goodbad[k_lead]

                # Now that the sliding train is updated, check if
                # the current frame has been flagged as bad:
                fcurrent = slide_block[:,:,nslide]
                if slide_gbs[nslide]==0:
                    continue

                # The ix1 and ix2 indices control how many frames
                # before and after the current frame are included
                # for comparison; the default is nslide, but this
                # will decrease close to the beginning and end due
                # to edge overlap:
                if k<nslide:
                    ix1 = nslide - k
                else:
                    ix1 = 0
                if k>irac.nframes-1-nslide:
                    ix2 = nslide + irac.nframes - k
                else:
                    ix2 = 2*nslide + 1

                # Use the ix1 and ix2 determined above to specify which
                # frames in the current train we actually want to use:
                if k==0:
                    fcompare = slide_block[:,:,k+1:ix2]
                    gbcompare = slide_gbs[k+1:ix2]
                elif k==irac.nframes-1:
                    fcompare = slide_block[:,:,ix1:nslide]
                    gbcompare = slide_gbs[ix1:nslide]
                else:
                    fbehind = slide_block[:,:,ix1:nslide]
                    gbbehind = slide_gbs[ix1:nslide]
                    fahead = slide_block[:,:,nslide+1:ix2]
                    gbahead = slide_gbs[nslide+1:ix2]
                    fcompare = np.dstack( [ fbehind, fahead ] )
                    gbcompare = np.concatenate( [ gbbehind, gbahead ] )
                if np.sum( gbcompare )<10:
                    warnings.warn( '  <10 good nearby frames to compare with... skipping' )
                    continue

                # Calculate the pixel medians and scatters for the
                # neighbouring frames:
                ixs = ( gbcompare==1 )
                fmedians = np.median( fcompare[:,:,ixs], axis=2 )
                fstdvs = np.std( fcompare[:,:,ixs], axis=2 )

                # Calculate how badly the worst pixel on the current
                # frame deviates from the values in surrounding frames:
                nsigmas = np.abs( fcurrent - fmedians )/fstdvs
                if np.max( nsigmas )>nsigma_clip:                
                    if irac.nsub[i]==1:
                        print '--> discarding frame {0}'\
                              .format( os.path.basename( irac.fitsfiles[i] ) )
                    else:
                        print '--> discarding frame {0} in {1}'\
                              .format( j+1, os.path.basename( irac.fitsfiles[i] ) )
                    irac.goodbad[k] = 0
                    
    nremoved_f = nstart - np.sum( irac.goodbad )
    if nremoved_f>0:
        if nremoved_f==1:
            frame_str = 'frame'
        else:
            frame_str = 'frames'
        print 'Removing {0:d} {1} due to discrepant pixels...'\
              .format( int( nremoved_f ), frame_str )
    else:
        print 'No frames removed due to discrepant pixels...'
    return None



def bg_subtract( irac ):
    """
    Calculates and subtracts the background from the
    raw aperture fluxes. Also calculates the theoretical
    shot noise for the resulting stellar flux.
    """

    print '\nRunning bg_subtract()...'
    print '\nUsing centroids determined by {0} method...'\
          .format( irac.xy_method )
    if irac.xy_method=='fluxweight':
        xy = irac.xy_fluxweight
    elif irac.xy_method=='gauss1d':
        xy = irac.xy_gauss1d
    elif irac.xy_method=='gauss2d':
        xy = irac.xy_gauss2d
    elif irac.xy_method=='iraf':
        xy = irac.xy_iraf
    else:
        raise AttributeError( 'xy_method either None or not recognised' )
    if xy==None:
        raise AttributeError( 'centroid xy coordinates must be calculated first' )

    if irac.goodbad==None:
        irac.preclean()
    if irac.nframes>1e5:
        nsigma_clip = 5
    else:
        nsigma_clip = 4
    irac.bg_ppix = -1*np.ones( irac.nframes )
    MJysr2electrons = irac.exptime*irac.gain/irac.fluxconv

    if irac.verbose>0:
        if irac.bg_kwargs['method']=='annulus_circle':
            print 'Calculating the background flux in annulus centered on star'
            print 'for each frame...'
        elif irac.bg_kwargs['method']=='corners':
            print 'Calculating the sky flux in {0}x{0} pixel corners of each image'\
                  .format( irac.bg_kwargs['ncorner'], irac.bg_kwargs['ncorner'] )
            print 'for each frame...'
        elif irac.bg_kwargs['method']=='custom_mask':
            print 'Calculating the sky flux from pixels in custom-defined mask for each image'
            print 'for each frame...'

    for i in range( irac.nfits ):

        # Read in the contents of the ith FITS file:
        hdu = fitsio.FITS( irac.fitsfiles[i], 'r' )
        #fits_data_i = hdu[0].read_image() # worked with fitsio v0.9.0
        fits_data_i = hdu[0].read() # works with fitsio v0.9.5
        hdu.close()

        for j in range( irac.nsub[i] ):

            # Extract the current frame:
            if irac.nsub[i]==1:
                fullarray = fits_data_i
            else:
                fullarray = fits_data_i[j,:,:]
                
            # Determine the current image number:
            k = np.sum( irac.nsub[:i] ) + j
            if irac.goodbad[k]==0:
                continue

            if i==0:
                naxis1 = np.shape( fullarray )[1]
                naxis2 = np.shape( fullarray )[0]
                xpixs = range( naxis1 )
                ypixs = range( naxis2 )
                xmesh, ymesh = np.meshgrid( xpixs, ypixs )
            if irac.goodbad[k]==0:
                continue
            if k%500==0:
                print '... up to frame {0} of {1} (in {2})'\
                      .format( k+1, irac.nframes, os.path.basename( irac.fitsfiles[i] ) )
            xycentroid = xy[k,:]
            if irac.bg_kwargs['method']=='annulus_circle':
                annulus_inedge = irac.bg_kwargs['annulus_inedge']
                annulus_width = irac.bg_kwargs['annulus_width']
                bg_pixs = get_annulus_circle_pixs( fullarray, xmesh, ymesh, xycentroid, \
                                                   irac.ap_radius, \
                                                   irac.bg_kwargs['annulus_inedge'], \
                                                   irac.bg_kwargs['annulus_width'] )
            elif irac.bg_kwargs['method']=='corners':
                bg_pixs = get_corner_pixs( fullarray, xmesh, ymesh, xycentroid, \
                                           irac.ap_radius, \
                                           irac.bg_kwargs['ncorner'] )
            elif irac.bg_kwargs['method']=='custom_mask':
                bg_pixs = get_corner_pixs( fullarray, xmesh, ymesh, xycentroid, \
                                           irac.ap_radius, \
                                           irac.bg_kwargs['custom_mask'] )

            # Two-pass sigma clipping:
            bg_med = np.median( bg_pixs )
            bg_stdv = np.std( bg_pixs )
            delta_sigmas = ( bg_pixs - bg_med )/bg_stdv
            ixs_keep_1 = ( delta_sigmas < nsigma_clip )
            bg_med = np.median( bg_pixs[ixs_keep_1] )
            bg_stdv = np.std( bg_pixs[ixs_keep_1] )
            delta_sigmas = ( bg_pixs[ixs_keep_1] - bg_med )/bg_stdv
            ixs_keep_2 = ( delta_sigmas < nsigma_clip )
            bg_pixs = bg_pixs[ixs_keep_1][ixs_keep_2]
                
            if irac.bg_kwargs['value']=='median':
                irac.bg_ppix[k] = np.median( bg_pixs )*MJysr2electrons
            elif irac.bg_kwargs['value']=='mean':
                irac.bg_ppix[k] = np.mean( bg_pixs )*MJysr2electrons
            else:
                pdb.set_trace() # none implemented yet
                
    print 'Done.'
        
    irac.fluxstar = -1*np.ones( irac.nframes )
    irac.shotstar = -1*np.ones( irac.nframes )
    ixs = ( irac.goodbad==1 )
    ap_bg = irac.nappixs[ixs]*irac.bg_ppix[ixs]
    irac.fluxstar[ixs] = irac.fluxraw[ixs] - ap_bg
    irac.shotstar[ixs] = np.sqrt( irac.fluxraw[ixs] + ap_bg \
                                  + irac.nappixs[ixs]*( irac.readnoise**2. ) )
    
    return None


def get_annulus_circle_pixs( pixvals, xmesh, ymesh, xymid, \
                             ap_radius, annulus_inedge, annulus_width ):
    """
    Uses a circular annulus centered centered on the aperture to  
    determine the background value. Either the mean or median pixel
    value is used, depending on which is specified. Anomalous
    pixels are discarded beforehand.
    """

    # Inner and outer edges of background annulus:
    iedge = np.max( [ ap_radius+1, annulus_inedge ] )
    oedge = annulus_inedge + annulus_width

    # Make a rough initial cut of a box containing
    # the background annulus:
    xymid = np.reshape( xymid, [ 1, 2 ] )
    ixs = ( xmesh.flatten()>xymid[0,0]-oedge-1 )*\
          ( xmesh.flatten()<xymid[0,0]+oedge+1 )*\
          ( ymesh.flatten()>xymid[0,1]-oedge-1 )*\
          ( ymesh.flatten()<xymid[0,1]+oedge+1 )
    xann = xmesh.flatten()[ixs]
    yann = ymesh.flatten()[ixs]

    # Remove the inner pixels of the rough box that 
    # don't fall within the background annulus:
    ixs = ( xann.flatten()>xymid[0,0]+iedge-1 )+\
          ( xann.flatten()<xymid[0,0]-iedge+1 )+\
          ( yann.flatten()>xymid[0,1]+iedge-1 )+\
          ( yann.flatten()<xymid[0,1]-iedge+1 )
    xann = xann.flatten()[ixs]
    yann = yann.flatten()[ixs]
    xypixs_rough = np.column_stack( [ xann, yann ] ) 

    # Distances from pixel centers within box to star coordinates:
    pixdists = scipy.spatial.distance.cdist( xypixs_rough, xymid )

    # Keep the pixels with centers that fall in the annulus:
    ixs = ( pixdists.flatten()>iedge )*( pixdists.flatten()<oedge )
    bg_pixs = pixvals.flatten()[ixs]

    return bg_pixs

def get_corner_pixs( fullarray, xmesh, ymesh, xymid, ap_radius, ncorner ):
    """
    """

    # Create background mask template:
    naxis2, naxis1 = np.shape( fullarray )
    bg_mask = np.zeros( [ naxis2, naxis1 ] )
    bg_mask[-ncorner:,:ncorner] = 1 # top left corner
    bg_mask[-ncorner:,-ncorner:] = 1 # top right corner
    bg_mask[:ncorner,:ncorner] = 1 # bottom left corner            
    bg_mask[:ncorner,-ncorner:] = 1 # bottom right corner

    # Extract the pixel values and their coordinates:
    ixs = ( bg_mask==1 )
    bg_pixs = fullarray[ixs].flatten()
    xcorners = xmesh[ixs].flatten()
    ycorners = ymesh[ixs].flatten()
    xycorners = np.column_stack( [ xcorners, ycorners ] )

    # Only keep the pixels that are outsiude the
    # photometric aperture:
    xymid = np.reshape( np.array( [ xymid[0], xymid[1] ] ), [ 1, 2 ] )
    pixdists = scipy.spatial.distance.cdist( xycorners, xymid )
    ixs = ( pixdists.flatten()>ap_radius+1 )
    bg_pixs = bg_pixs[ixs]

    return bg_pixs


def get_custom_mask_pixs( fullarray, xmesh, ymesh, xymid, ap_radius, custom_mask ):
    """
    """

    # Extract the pixels within the mask:
    ixs = ( custom_mask==1 )
    bg_pixs = fullarray[ixs].flatten()
    xmask = xmesh[ixs].flatten()
    ymask = ymesh[ixs].flatten()
    xymask = np.column_stack( [ xmask, ymask ] )

    # Only keep the pixels that are outsiude the
    # photometric aperture:
    xymid = np.reshape( np.array( [ xymid[0], xymid[1] ] ), [ 1, 2 ] )
    pixdists = scipy.spatial.distance.cdist( xymask, xymid )
    ixs = ( pixdists.flatten()>ap_radius+1 )
    bg_pixs = bg_pixs[ixs]

    return bg_pixs


def ap_phot( irac, save_pngs=False ):
    """
    Calculates raw aperture fluxes. Background subtraction
    must be done separately with the bg_subtract() routine.
    """
    
    print '\nUsing centroids determined by {0} method...'\
          .format( irac.xy_method )
    if irac.xy_method=='fluxweight':
        xy = irac.xy_fluxweight
    elif irac.xy_method=='gauss1d':
        xy = irac.xy_gauss1d
    elif irac.xy_method=='gauss2d':
        xy = irac.xy_gauss2d
    elif irac.xy_method=='iraf':
        xy = irac.xy_iraf
    else:
        raise AttributeError( 'xy_method either None or not recognised' )
    if xy==None:
        raise AttributeError( 'centroid xy coordinates must be calculated first' )
    photom_boxwidth = 2.*irac.ap_radius + 3.

    if irac.goodbad==None:
        irac.preclean()

    # Data units to electrons:
    irac.fluxraw = -1*np.ones( irac.nframes )
    irac.nappixs = -1*np.ones( irac.nframes )
    MJysr2electrons = irac.exptime * irac.gain / irac.fluxconv

    # Prepare for interpolated aperture photometry if requested:
    if irac.ninterp>1:
        print 'Doing {0:d}x interpolated aperture photometry'.format( irac.ninterp )
        dAsub = ( 1./irac.ninterp )**2.
        nfine = int( ( photom_boxwidth - 1. )*irac.ninterp )
    else:
        print 'Doing non-interpolated aperture photometry:'
    print '(aperture radius = {0} pixels)'.format( irac.ap_radius )

    
    for i in range( irac.nfits ):

        # Read in the contents of the ith FITS file:
        hdu = fitsio.FITS( irac.fitsfiles[i], 'r' )
        #fits_data_i = hdu[0].read_image() # worked with fitsio v0.9.0
        fits_data_i = hdu[0].read() # works with fitsio v0.9.5
        hdu.close()

        for j in range( irac.nsub[i] ):

            # Determine the current image number:
            k = np.sum( irac.nsub[:i] ) + j
            if k%500==0:
                print '... up to frame {0} of {1} (in {2})'\
                      .format( k+1, irac.nframes, os.path.basename( irac.fitsfiles[i] ) )
            if irac.goodbad[k]==0:
                continue

            # Extract the current frame:
            if irac.nsub[i]==1:
                fullarray = fits_data_i
            else:
                fullarray = fits_data_i[j,:,:]

            # Center of aperture:
            xcent = xy[k,0]
            ycent = xy[k,1]

            # Cut subarray from full frame:
            subarray, xsub, ysub = cut_subarray( fullarray, xcent, ycent, photom_boxwidth )
            # Bilinear interpolation of subarray:
            interpf = scipy.interpolate.RectBivariateSpline( xsub, ysub, subarray.T, kx=1, ky=1, s=0 )
            # Interpolate to a finer grid:
            xfine = np.r_[ xsub.min() : xsub.max() : 1j*nfine ]
            yfine = np.r_[ ysub.min() : ysub.max() : 1j*nfine ]
            xmeshf, ymeshf = np.meshgrid( xfine, yfine )
            zf = interpf.ev( xmeshf.flatten(), ymeshf.flatten() )
            # Calculate distances from subpixels to
            # center of aperture:
            xysubpixs = np.column_stack( [ xmeshf.flatten(), ymeshf.flatten() ] )
            xy_cent = np.reshape( np.array( [ xcent, ycent ] ), [ 1, 2 ] )
            pixdists = scipy.spatial.distance.cdist( xysubpixs, xy_cent )
            # Keep those subpixels with centers falling
            # within the aperture:
            ixs = ( pixdists.flatten()<irac.ap_radius )
            zfap = zf[ixs] * MJysr2electrons 
            nsubpixs = len( zfap )
            # Sum the subpixels falling within the aperture
            # and multiply them by the subpixel areas to get
            # the integrated flux:
            irac.fluxraw[k] = dAsub * np.sum( zfap )
            irac.nappixs[k] = dAsub * nsubpixs

            # Save png images of each photometric aperture if
            # requested:
            if save_pngs==True:
                fig = plt.figure()
                ax = fig.add_subplot( 111 )
                if irac.nsub[i]==1:
                    title = '{0}'.format( irac.fitsfiles[i] )
                else:
                    title = 'frame {0} in {1}'.format( j+1, irac.fitsfiles[i] )
                ax.set_title( title )
                imz = np.reshape( zf, [ len( xfine ), len( yfine ) ] )
                plt.imshow( imz, extent=[ xmeshf.min(), xmeshf.max(), \
                                          ymeshf.min(), ymeshf.max() ], \
                            origin='lower', interpolation='nearest' )
                plt.plot( [ xcent ], [ ycent ], 'ok' )
                plt.axvline( xcent - irac.ap_radius, c='k' )
                plt.axvline( xcent + irac.ap_radius, c='k' )
                plt.axhline( ycent - irac.ap_radius, c='k' )
                plt.axhline( ycent + irac.ap_radius, c='k' )
                cwd = os.getcwd()
                ofolder = os.path.join( cwd, 'photimages' )
                if os.path.isdir( ofolder )==False:
                    os.mkdir( ofolder )
                ndig = int( np.ceil( np.log10( irac.nframes ) ) )
                figname = os.path.join( ofolder, 'image{0:0{1}d}.png'.format( k+1, ndig ) )
                plt.savefig( figname )
                plt.close()
        
    print 'Done.'
    print 'NOTE: Raw aperture fluxes have been calculated, but'
    print 'the background must be subtracted separately with'
    print 'the bg_subtract() method.'
    return None

def cut_subarray( fullarray, xcent, ycent, boxwidth ):
    """
    Cuts a subarray from a 2D array.
    """
    
    naxis1 = np.shape( fullarray )[1]
    naxis2 = np.shape( fullarray )[0]

    # Left-hand edges of x pixels, where 0 is
    # the left-hand edge of the first column:
    xpixs = np.arange( naxis1 )

    # Lower edges of y pixels, where 0 is the
    # lower edge of the first row:
    ypixs = np.arange( naxis2 )

    # Number of pixels either side of
    # central pixel in subarray:
    delpix = int( 0.5*( boxwidth - 1 ) )

    # Cut out the subarray:
    xixs = ( xpixs>=np.floor( xcent )-delpix )*\
           ( xpixs<np.floor( xcent )+1+delpix )
    yixs = ( ypixs>=np.floor( ycent )-delpix )*\
           ( ypixs<np.floor( ycent )+1+delpix )
    subarray = fullarray[yixs,:][:,xixs]

    # Convert coordinates from pixel edges to
    # pixel centers before returning:
    xsub = xpixs[xixs] + 0.5
    ysub = ypixs[yixs] + 0.5

    return subarray, xsub, ysub


def save_table( irac, ofilename=None ):
    """
    Generates and saved a FITS output table.

    Columns:
      1. 'bjd' - Barycentric Julian Date at mid-exposure.
      2. 'fluxraw' - Integrated aperture flux before
         background subtraction, in electrons.
      3. 'fluxstar' - Integrated aperture flux after
         background subtraction, in electrons.
      4. 'shotstar' - Theoretical shot noise for fluxstar,
         in electrons.
      5. 'xcent' - x-coordinate of centroid.
      6. 'ycent' - y-coordinate of centroid.
      7. 'goodbad' - Flags for good (=1) and bad (=0) frames.
      8. 'nappixs' - Effective number of pixels in circular
         photometric aperture.
      9. 'bg_ppix' - Background value per pixel, in electrons.

    Keywords:
      1. 'centroid_method' - String describing the method
         used to calculate the centroids.
      2. centroid_kwargs - Each keyword argument for the
         centroiding method is also saved in the table.
      3. 'bg_method' - String describing the method used
         to calculate the background values.
      4. bg_kwargs - Each keywords argument for the background
         estimation method is also saved in the table.
      5. 'ap_radius' - Radius of the photometric aperture,
         used, in pixels.
      6. 'gain' - Conversion factor from DN to electrons.
      7. 'readnoise' - Read noise per pixel, in electrons.
      8. 'nframes_total' - Total number of frames.
      9. 'nframes_rejected' - Number of rejected frames.
    """
    opath = os.path.join( irac.adir, ofilename )
    otable = atpy.Table()
    otable.add_column( 'bjd', irac.bjd, \
                       description='BJD at mid-exposure', unit='days' )
    otable.add_column( 'nappixs', irac.nappixs, unit='pixel areas' )
    otable.add_column( 'bg_ppix', irac.bg_ppix, unit='electrons' )
    otable.add_column( 'fluxraw', irac.fluxraw, unit='electrons' )
    otable.add_column( 'fluxstar', irac.fluxstar, unit='electrons' )
    otable.add_column( 'shotstar', irac.shotstar, unit='electrons' )
    otable.add_column( 'goodbad', irac.goodbad, unit='1=good, 0=bad' )
    centroid_method = irac.centroid_kwargs['method']
    otable.add_keyword( 'centroid_method', centroid_method )
    if centroid_method=='gauss2d':
        otable.add_column( 'xcent', irac.xy_gauss2d[:,0], unit='pixel' )
        otable.add_column( 'ycent', irac.xy_gauss2d[:,1], unit='pixel' )
    elif centroid_method=='fluxweight':
        otable.add_column( 'xcent', irac.xy_fluxweight[:,0], unit='pixel' )
        otable.add_column( 'ycent', irac.xy_fluxweight[:,1], unit='pixel' )
    elif centroid_method=='gauss1d':
        otable.add_column( 'xcent', irac.xy_gauss1d[:,0], unit='pixel' )
        otable.add_column( 'ycent', irac.xy_gauss1d[:,1], unit='pixel' )
    for key in irac.centroid_kwargs.keys():
        if key!='method':
            otable.add_keyword( key, irac.centroid_kwargs[key] )
    otable.add_keyword( 'bg_method', irac.bg_kwargs['method'] )
    otable.add_keyword( 'ap_radius (pixel lengths)', irac.ap_radius )
    otable.add_keyword( 'gain (e/DN conversion)', irac.gain )
    otable.add_keyword( 'readnoise (e/DN conversion)', irac.gain )
    nframes = len( irac.goodbad )
    nreject = np.sum( irac.goodbad )
    otable.add_keyword( 'nframes_total', nframes )
    otable.add_keyword( 'nframes_reject', nreject )
    otable.write( opath, overwrite=True )
    print '\nSaved output table:\n{0}'.format( opath )

    return None
    
    
def rebin1d( a, binning_factor ):
    """
    Resizes a 1D array by averaging. New dimensions must be
    integral factors of the original dimensions. This routine
    was inspired by the IDL rebin routine; however, currrently
    it only has the option to bin down the input array to a
    lower resolution - it does not interpolate to higher
    resolutions (although this wouldn't be hard to add).
 
    Inputs
    ------
    a : 1D array
    binning_factors : tuple of int giving resolution decreas
    for each axis.

 
    Output
    -------
    rebinned_array : 1D array with reduced resolution.
    """
    M = len( a.flatten() )
    m = binning_factor
    a = np.reshape( a, ( M/m, m ) )
    a = np.sum( a, axis=1 )/float( m )
    return a
