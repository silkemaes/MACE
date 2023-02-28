!     Python C-bindings for CS model
!     ------------------------------
!     Currently the old Fortran main has been split into two subroutines:
!     (1) setup: mostly reading in data
!     (2) solve: performing the chemical evolution (solving the ODEs)
!     This file only contains the setup
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine setup(Yi,Ki) bind(c, name="setup")
      use iso_c_binding
       
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      CHARACTER*10  SP(468),CSP(2),RE1,RE2,P1,P2,P3,P4,PARENT(200)
      CHARACTER*5   FLG
      CHARACTER*1   SR
      CHARACTER*11  JAC
      CHARACTER*500 FRATES, FSPECS, FOUTF, FOUTN, FPARENTS
      INTEGER OU
      INTEGER IWORK(1025),I,J,FSPEC,FRATE,FFRAC,FNUM,UFRAC,URATES,USPEC,
     *     NSPEC,NCONS,NPAR,ICO,ICOR,IRUN,ITOL,ITASK,ISTATE,IOPT,LRW,
     *     LIW,MF,NREAC,IS,IF,LI,LTAG,NEXTRA,IFREEZE,IANA,UANA,
     *     UPARENTS
      DOUBLE PRECISION K(10000),MASS(468),NGD,RTOL,ATOL,TSTART,Y(466),
     *     T,RWORK(2000000),TOTAL(10),B(468,300),CINP(2),TAGE(300),SH,
     *     X(10),GR,DN,TFINAL,TLAST,KJ,ACCR,HNR,PI,KB,MH,MU,PABUND(200)

!     Define common blocks     
      COMMON/BL1/ K,X,TOTAL,GR,DN,ACCR,HNR,IFREEZE
      COMMON/BL10/ MASS,SH
      COMMON/BL3/ Y,X_G,A_G,TEMP,AV,ZETA,ALBEDO,RAD
          
!     NC = Number of conserved species, NR = Number of reactions, N = Number of species (ODEs)
      PARAMETER(OU=8,NC=2,NR=6173,N=466)

!     PHYSICAL CONSTANTS
      DATA PI, MH, MU, KB / 3.1415927, 1.6605E-24, 2.2, 1.3807E-16/

      real(c_double), intent(inout) :: Yi(  466)
      real(c_double), intent(inout) :: Ki(10000)

      K = Ki
      Y = Yi

      OPEN(UNIT=11, FILE="dat/rate13steady.state")
      
!     FILE NAME RATE
      FRATES   = 'dat/rate13.rates'
!     FILE NAME SPECIES 
      FSPECS   = 'dat/dc.specs' 
!     FILE NAME OUTPUT 
      FOUTF    = 'out/dc.out' 
!     FILE NAME PARENTS
      FPARENTS = 'dat/dc.parents'

      USPEC    =  1
      UFRAC    =  3
      URATES   =  4
      UANA     =  5
      UPARENTS = 37
      
!     ANALYSE CHEMISTRY?
      IANA = 1
!     ANALYSIS TIME (IRUN)
      IRUN = 51

!     Open rate, species, and output files
      OPEN(UNIT=USPEC,    FILE=FSPECS,   STATUS='OLD')
      OPEN(UNIT=UPARENTS, FILE=FPARENTS, STATUS='OLD')
      OPEN(UNIT=UFRAC,    FILE=FOUTF)

!     Physical parameters - temperature, density, cosmic ray
!     ionisation and UV radiation field scaling factors and visual extinction
      TEMP = 10.0
      DN   =  2.0E4
      ZETA =  1.0
      RAD  =  1.0
      AV   = 10.0
!     GRAIN RADIUS (cm) (0.1 micron = 1.0E-5 cm)
      A_G = 1.0E-5
!     GRAIN NUMBER DENSITY/H2 (assuming gas/dust = 200, rho = 3.5 g/cm^3)
!     USED FOR H-ATOM ACCRETION CALCULATION
      X_G = 1.5E-12
!     HNR = 1.0 FOR DENSE CLOUD CHEMISTRY
      HNR = 1.0
!     GRAIN ALBEDO
      ALBEDO = 0.6
!     SET STICKING COEFFICIENT FOR H ATOMS
      SH = 0.3
!     H-ATOM ACCRETION RATE
      ACCR = SH*PI*(A_G**2.0)*DN*X_G*(8.0*KB*TEMP/(PI*MH))**0.5
!     SET GRAIN SURFACE FORMATION OF H2 TO CLASSICAL RATE
!     GR = 5.2E-17*(TEMP*3.33E-3)**0.5
!     ACCR = GR*DN

!     TFINAL is the end time in years
      TFINAL = 1.0E8

!     TSTART is the first output point at which abundances are printed
      TSTART = 1.0
      TSTART = TSTART*3.1536E7
      TLAST  = TFINAL*3.1536E7
      TOUT   = TSTART
      TL     = 0.0
!     IFREEZE CONTROLS WHETHER FREEZE OUT IS INCLUDED (VIA SUBROUTINE FREEZE)
      IFREEZE = 0
!     IF FREEZE OUT ONTO GRAINS IS INCLUDED SET IFREEZE = 1
!     IFREEZE = 1

      WRITE( *,*)'Temperature.. ',TEMP
      WRITE(11,*)'Temperature.. ',TEMP
      WRITE( *,*)'Density..     ',DN
      WRITE(11,*)'Density..     ',DN
      WRITE( *,*)'CR Ionisation rate scaling.. ',ZETA
      WRITE(11,*)'CR Ionisation rate scaling.. ',ZETA
      WRITE( *,*)'UV radiation field scaling.. ',RAD
      WRITE(11,*)'UV radiation field scaling.. ',RAD
      WRITE( *,*)'Visual extinction.. ',AV
      WRITE(11,*)'Visual extinction.. ',AV
      WRITE( *,*)'Reading species and initial abundances..'

!     Input section
!     -------------
!     read species file.. get species names, masses and initial abundances
!        
!     parent species tov H
!     density #/cm3
      WRITE( *,*)'Initially, relative to H2'
      WRITE(11,*)'Initially, relative to H2'
      DO 2 I = 1,N
          READ(USPEC,100)SP(I),Y(I),MASS(I)
          IF(Y(I).GT.0) THEN
              WRITE(11,100)SP(I),2*Y(I),MASS(I)
              WRITE( *,100)SP(I),2*Y(I),MASS(I)
!             CONVERT ABUNDANCES TO PER UNIT VOLUME RATHER THAN FRACTIONAL WRT H2
              Y(I)=Y(I)*DN
          ENDIF
 2    CONTINUE

      DO 3 I=1,NC
          READ(USPEC,100)SP(N+I),CINP(I),MASS(N+I)
          WRITE(11,100)SP(N+I),CINP(I),MASS(N+I)
          WRITE( *,100)SP(N+I),CINP(I),MASS(N+I)
          TOTAL(I)=CINP(I)*DN
 3    CONTINUE
      WRITE(11,170)TOTAL(1),TOTAL(2)
            
      NPAR = 1
 109  READ(UPARENTS,*,END=101) PARENT(NPAR),PABUND(NPAR)
      NPAR = NPAR + 1
      GO TO 109

 101  CONTINUE
   

!     SET INITIAL ABUNDANCES
      DO I = 1,NPAR
          DO J = 1,N+NC
              IF(SP(J).EQ.PARENT(I)) THEN
                  Y(J) = PABUND(I)*DN
!                 LOAD FIRST ELEMENT OF OUTPUT ARRAYS
!                            B(J,0) = PABUND(I)* HNR
                  WRITE(*,106) SP(J),PABUND(I)
 106              FORMAT(3X,A12,1PE8.2)
              ENDIF
          END DO
      END DO


!     READ RATE FILE AND EXTRACT RATE DATA FOR EACH REACTION
      WRITE(*,*) '...Reading rate file..............'
      CALL READR(FRATES,URATES)
      
!     CALL SUBROUTINE TO CALCULATE RATE COEFFICIENTS
      CALL RATES
    

!     FORMATS
!     -------
!     read species file and write steady state output file
 100  FORMAT(6X,A10,1X,1PE8.2,2X,F5.1)
 170  FORMAT(3X,'TOTAL(1) = ELECTRONS = ',1PE8.2,3X,'TOTAL(2) = H2 = ',
     *  1PE8.2)
!     write main output file
 114  FORMAT(//)
 116  FORMAT(3X,'TIME',7X,10(1A9,2X))
 118  FORMAT(1X,11(1PE11.3))
!     read ratefile
 102  FORMAT(6X,A9,A9,A9,A9,A9,A8,1PE8.2,1X,0PF5.2,F9.1)
 105  FORMAT(1PE11.5)
 123  FORMAT(I5,1X,A9,A9,A9,A9,A9,A8,1PE8.2,F6.2,F9.1)
 202  FORMAT(5X,A9,A9,A9,A9,A9,A5,A5,1PE8.2,1X,0PF7.2,2X,F8.1,A1,
     *     I5,I5,A5)
!     write binned.rates
 223  FORMAT(I4,1X,A9,A9,A9,A9,A9,A5,A5,1PE8.2,1X,0PF7.2,2X,F8.1,A1,
     *     I5,I5,A5)
    

      ! Copy variables
      ! Unfortunately, this is required to avoid issues with the common blocks...
      Ki = K
      Yi = Y
    
      end subroutine setup
