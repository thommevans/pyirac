import numpy as np
import numexpr
import scipy.spatial

def txy_kernel_numexpr( v1, v2, **cpars ):
  """
  
  """

  At = cpars['At']
  Lt = cpars['Lt']
  Axy = cpars['Axy']
  Lxy = cpars['Lxy']

  v1 = np.matrix( v1 )

  if v2==None:

      n = np.shape( v1 )[0]
      cov = ( At**2. ) + ( Axy**2. ) + np.zeros( n )
      cov = np.reshape( cov, [ n, 1 ] )

  else:
      
      v2 = np.matrix( v2 )

      t1 = v1[:,0]
      xy1 = v1[:,1:]
      t2 = v2[:,0]
      xy2 = v2[:,1:]

      # GENERAL COMMENT: numexpr should also be able to speed
      # up the scipy.spatial.distance.cdist() calls... but will
      # need to think about how to implement this...

      Dt = scipy.spatial.distance.cdist( t1, t2, 'euclidean' )

      # Without numexpr:
      #arg = np.sqrt( 3 )*Dt/Lt
      #poly_term = ( 1. + arg ) 
      #exp_term = np.exp( -arg )
      #covt = ( At**2 )*poly_term*exp_term

      # Element-wise operations are faster with numexpr:
      arg = numexpr.evaluate( 'sqrt( 3 )*Dt/Lt' )
      poly_term = numexpr.evaluate( '1.+arg' )
      exp_term = numexpr.evaluate( 'exp(-arg)' )
      covt = numexpr.evaluate( '( At**2 )*poly_term*exp_term' )

      # Cannot do matrix multiplication with numexpr:
      Mxy = np.matrix( np.diag( 1./Lxy ) )
      xy1 = xy1*Mxy
      xy2 = xy2*Mxy
      D2xy = scipy.spatial.distance.cdist( xy1, xy2, 'sqeuclidean' )
      term1 = Axy**2.
      term2 = numexpr.evaluate( 'exp( -0.5*D2xy )' )
      covxy = numexpr.evaluate( 'term1*term2' )

      cov = numexpr.evaluate( 'covt + covxy' )
      
  return cov
