import pdb, os, sys, shutil 
import numpy as np
from . import single_star_routines

class irac():
    """
    """
    def __init__( self ):
        """
        """
        self.ddir = None
        self.adir = None
        self.fitsfiles = None
        self.obsblocks = None
        self.init_xy = None
        self.bjd = None
        self.ap_radius = 3
        self.ap_radius_noisepix_params = [ 0, 0 ]
        self.noisepix_half_boxwidth = 6
        self.fluxstar = None
        self.shotstar = None
        self.fluxraw = None        
        self.nappixs = None
        self.bg_ppix = None
        self.channel = None
        self.verbose = 0
        self.ninterp = 10 
        self.centroid_kwargs = { 'method':'gauss2d', 'boxwidth':10 }
        self.bg_kwargs = { 'method':'annulus_circle', 'value':'median', \
                           'annulus_inedge':15, 'annulus_width':2 }
        self.goodbad = None
        self.goodbad_xy = None
        self.goodbad_flux = None
        self.bjd_kw = 'BMJD_OBS'
        self.readnoise_kw = 'RONOISE'
        self.readnoise = None
        self.gain_kw = 'GAIN'
        self.gain = None
        self.exptime_kw = 'EXPTIME'
        self.exptime = None
        self.fluxconv_kw = 'FLUXCONV'
        self.fluxconv = None
        self.framtime_kw = 'FRAMTIME'
        self.framtime = None
        return None

    def read_headers( self ):
        """
        Reads information from the FITS headers, including the
        readnoise, gain etc. Also creates an array containing
        the BJDs at mid-exposure, based on information contained
        in the headers.
        """
        single_star_routines.read_headers( self )
        return None

    def centroids( self ):
        """
        """
        single_star_routines.centroids( self )

    def extract_pix_timeseries( self ):
        """
        """
        single_star_routines.extract_pix_timeseries( self )

    def preclean( self ):
        """
        """
        single_star_routines.preclean( self )

    def ap_phot( self, save_pngs=False ):
        """
        """
        single_star_routines.ap_phot( self, save_pngs=save_pngs )
        return None

    def bg_subtract( self ):
        """
        """
        single_star_routines.bg_subtract( self )
        return None

    def save_table( self, ofilename=None ):
        """
        """
        single_star_routines.save_table( self, ofilename=ofilename )
