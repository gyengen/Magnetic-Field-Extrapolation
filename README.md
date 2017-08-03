# Magnetic-Field-Extrapolation
Magnetic Field Extrapolation (potential) for solar active regions. LOS magnetogram data provide a basis for active region magnetic field modeling.

Input
-----

bz: Vertical or longitudinal magnetic field. 2d numpy array.

nz: Number of equally spaced grid points in the z direction (default = 30)

zscale : Sets the z-scale (1.0 = same scale as the x,y axes before the heliographic transformation). Default= 1.0


Returns
-------
magout : New magnetic field structure with the fields bxp,byp,bzp defined These are 3D arrays in x,y,z giving the x,y,z components of the force free field.
