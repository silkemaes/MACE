## Chemistry solver: equations

***Nieuwe chemie solver bouwen in PyTorch: $ChemTorch$*** --> "mean field approximation"
	==> ODEs van Fortran vertalen naar Python; zie $\texttt{/lhome/silkem/MACE/MACE/ODEs-to-python.ipynb}$

### Rate equations 
- ***Two body:*** $$k=\alpha\left(\frac{T}{300}\right)^\beta\exp\left(-\frac{\gamma}{T}\right)$$ in ${\rm cm^3 \, s^{-1}}$
- ***Cosmic rays***
	- (CP) Direct ionisation: $$k=\alpha$$ in ${\rm s^{-1}}$
	- (CR) Induced photoreaction: $$k = \alpha\left(\frac{T}{300}\right)^\beta\frac{\gamma}{1-w}$$ in ${\rm s^{-1}}$
- ***(PH) Photodissociation:*** $$k = \alpha\, \delta \exp (-\gamma A_V)$$ in ${\rm s^{-1}}$, where $\delta = {\rm RAD}$ in the Fortran chemistry code. 
	In the 1D chemistry code, the following approach is used ([Morris & Jura (1983)](https://ui.adsabs.harvard.edu/abs/1983ApJ...264..546M))
		$$A_V = \frac{N({\rm H_2})}{1.87\times 10^{21}}$$ with $$N({\rm H_2}) = \int_r^\infty n({\rm H_2})dr = n({\rm H_2})r = \frac{\dot{M}}{4\pi r v_\infty}$$
		Radiation field: $$I=\frac{I_0}{4\pi}\int e^{-\tau} = \alpha \delta$$ and $$J=\alpha \frac{M_iJ_0}{4\pi}\int {\rm d}\Omega \left(\frac{N({\rm H_2})\phi}{\sin \phi}\right)^{-1/2}$$
		--> since the radiation is shielded
			This last equation can be found by 
			$$\tau_\ell =\int_?^\infty d\ell\rho(\ell)\kappa(\ell) = \kappa(\ell) \int{\rm d}\ell \frac{\ell_0}{r^2(\ell)}$$ since $\rho(r)=\rho_0/r^2$.
			Thus, $$\tau_\ell = \kappa(\ell)\ell_0\int \frac{{\rm d \ell}}{r^2_0 + \ell^2+r_0\ell\cos\theta}$$
			If this is solved with the proper boundaries (cfr. Thomas Ceulemans, Luka, Frederik), the factor $\frac{\theta}{\sin\theta}$ comes out.

### Vier fysische parameters $$\vec{p} = (\rho, T, \delta, A_V)$$
- $\rho$ = density
- $T$ = temperature
- $\delta$ = overall outwards dilution of the radiation field, [1e-6, 1]
- $A_V$ = dust extinction in the direction of the ISM (outwards) $\sim$ optical depth [0,-ln(delta)]
*If taking into account companion photons: extra physical parameters:*
 - $W$ = geometrical dilution (inwards)
 - $\Delta A_V$ = dust extinction in the direction of the companion (inwards)

### Chemische abundanties $$\vec{n}=\vec{n}(t)$$

## Fortran code RAD ($\delta$) en AUV (Av)
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  FUNCTION TO CALCULATE RADIATION FIELD STRENGTH AT 1000 A  C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      DOUBLE PRECISION FUNCTION GETRAD(AUV)
C  AUV = RADIAL EXTINCTION AT 1000 ANGSTROMS
	
      CHARACTER*500 CLUMPMODE,TEMPMODE
      DOUBLE PRECISION AUV,CETA(6),GA(6),W(6),PI,SUM,GAMMA,TAU,
     * TCL,TEFF,EARG,RADIUS,FVOL,L,H2COL,HNR,KAPPA,GETH2,GETHNR
      DOUBLE PRECISION MH,MU,KB,CORR,H,SUM2,B,FIC,
     * T_STAR,EPSIL,R_STAR,K,A,C,TEFF_INF,TEFF_PAR,TEFF_R,
     * AUV_AV,GAMCO,ALBEDO,ZETA,A_G,X_G,Y(1000),MLOSS,V,GSTAR,
     *   RSCALE,RDUST,DELTA_AUV,R_EPSIL,GBIN
      INTEGER I,ICLUMP,ICO,ISTELLAR,IBIN

      PI = 4.*ATAN(1.)


C   SET COEFFICIENTS FOR CALCULATION OF RADIATION FIELD
      W(1) = 0.17132449
      W(2) = 0.36076157
      W(3) = 0.46791393
      W(4) = W(1)
      W(5) = W(2)
      W(6) = W(3)
      GA(1) = 0.93246951
      GA(2) = 0.66120939
      GA(3) = 0.23861919
      GA(4) = -GA(1)
      GA(5) = -GA(2)
      GA(6) = -GA(3)

      SUM = 0.0
      DO I=1,6
      CETA(I) = (PI*GA(I)+PI)/2.0
      SUM=SUM+(W(I)*(SIN(CETA(I))*EXP((-AUV*CETA(I))/SIN(CETA(I)))))
      END DO
      SUM = (PI/4.0)*SUM
      
      GETRAD = SUM
                        
      RETURN
      END
      
      
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
C  FUNCTION FOR EXTINCTION AT 1000 ANGSTROMS (NEJAD AND MILLAR)   C
CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

      DOUBLE PRECISION FUNCTION GETAUV(H2COL,RADIUS)
C  H2COL = RADIAL TOTAL H COLUMN DENSITY (OUTWARDS)
      CHARACTER*500 CLUMPMODE,TEMPMODE
      DOUBLE PRECISION H2COL,AUV_AV,GAMCO,ALBEDO,ZETA,A_G,X_G,Y(1000),
     *    GSTAR,RSCALE,RDUST,DELTA_AUV,AUV,K,H,A,B,C,RADIUS,TEFF,
     *    TEFF_R,TEFF_INF,R_STAR,PI,MH,MU,KB,T_STAR,EPSIL,
     *    FVOL,L,FIC,BFILL,CFILL,R,R_EPSIL,GBIN
      INTEGER ICO,ICLUMP,ISTELLAR,IBIN
      COMMON/BL3/ Y,X_G,A_G,ZETA,ALBEDO,GAMCO,AUV_AV,ICO,GSTAR,GBIN,
     *   RSCALE,RDUST,DELTA_AUV,ISTELLAR,IBIN
      COMMON/BLC/ PI,MH,MU,KB
      COMMON/CLM/ CLUMPMODE,ICLUMP,FVOL,L,FIC
      COMMON/BLTEMP/ R_STAR,T_STAR,EPSIL,TEMPMODE,R_EPSIL
C	Column density H2/A(V) = 1.87E21 atoms cm-2 mag-1
C	http://adsabs.harvard.edu/abs/1995A&A...293..889P

!       AUV = (AUV_AV*2.0*H2COL)/1.87E21
      AUV = (AUV_AV*H2COL)/1.87E21
      

      GETAUV = AUV

      
      
      
      RETURN
      END
      
