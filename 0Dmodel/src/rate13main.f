C
C made by rate version 1.5 on Tue, May 15 2012 at 13:24:22
C
c issues users need to be aware of:
c ---------------------------------
c  expects species file 'dc.specs' with initial abundances filled in
c  produces steady state abundance file 'rate13steady.state'
c  produces full output file 'dc.out'
c  abundances in the output files are relative to H2 
c  except that of H2 itself.
c
c  this program needs to be compiled and linked to the odes file odes.f
c  and also to the GEAR package LSODE.
c
c  for more info see http://www.udfa.net
c
      PROGRAM MAIN
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      CHARACTER*10  SP(468),CSP(2),RE1,RE2,P1,P2,P3,P4,PARENT(200)
      CHARACTER*5 FLG
      CHARACTER*1 SR
      CHARACTER*11 JAC
      CHARACTER*500 FRATES,FSPECS,FOUTF,FOUTN,FPARENTS
      INTEGER OU
      INTEGER IWORK(1025),I,J,FSPEC,FRATE,FFRAC,FNUM,UFRAC,URATES,USPEC,
     *     NSPEC,NCONS,NPAR,ICO,ICOR,IRUN,ITOL,ITASK,ISTATE,IOPT,LRW,
     *     LIW,MF,NREAC,IS,IF,LI,LTAG,NEXTRA,IFREEZE,IANA,UANA,
     *     UPARENTS
      DOUBLE PRECISION K(10000),MASS(468),NGD,RTOL,ATOL,TSTART,Y(466),
     *     T,RWORK(2000000),TOTAL(10),B(468,300),CINP(2),TAGE(300),SH,
     *     X(10),GR,DN,TFINAL,TLAST,KJ,ACCR,HNR,PI,KB,MH,MU,PABUND(200)
      COMMON/BL1/ K,X,TOTAL,GR,DN,ACCR,HNR,IFREEZE
      COMMON/BL10/ MASS,SH
      COMMON/BL3/ Y,X_G,A_G,TEMP,AV,ZETA,ALBEDO,RAD
C  NC = Number of conserved species, NR = Number of reactions, N = Number of species (ODEs)
      PARAMETER(OU=8,NC=2,NR=6173,N=466)
      EXTERNAL DIFFUN
C
C  PHYSICAL CONSTANTS
      DATA PI,MH,MU,KB/3.1415927,1.6605E-24,2.2,1.3807E-16/
C
      OPEN(UNIT=11, FILE="rate13steady.state")
      
C  RATE FILE NAME
      FRATES = 'rate13.rates'
C  SPECIES FILE NAME
      FSPECS = 'dc.specs' 
C  OUTPUT FILE NAME
      FOUTF = 'dc.out' 
      
      FPARENTS = 'dc.parents'
C
      USPEC = 1
      UFRAC = 3
      URATES = 4
      UANA = 5
      UPARENTS = 37
      
C  ANALYSE CHEMISTRY?
      IANA = 1
C  ANALYSIS TIME (IRUN)
      IRUN = 51
c      
C  OPEN RATE, SPECIES AND OUTPUT FILES
      OPEN(UNIT=USPEC, FILE=FSPECS, STATUS='OLD')
      OPEN(UNIT=UPARENTS, FILE=FPARENTS, STATUS='OLD')
      OPEN(UNIT=UFRAC, FILE=FOUTF)
C
C  Physical parameters - temperature, density, cosmic ray
c  ionisation and UV radiation field scaling factors and visual extinction
      TEMP = 10.0
      DN = 2.0E4
      ZETA = 1.0
      RAD = 1.0
      AV = 10.0
C  GRAIN RADIUS (cm) (0.1 micron = 1.0E-5 cm)
      A_G = 1.0E-5
C  GRAIN NUMBER DENSITY/H2 (assuming gas/dust = 200, rho = 3.5 g/cm^3)
C  USED FOR H-ATOM ACCRETION CALCULATION
      X_G = 1.5E-12
C	HNR = 1.0 FOR DENSE CLOUD CHEMISTRY
       HNR = 1.0
C   GRAIN ALBEDO
       ALBEDO = 0.6
C SET STICKING COEFFICIENT FOR H ATOMS
      SH = 0.3
C
C  H-ATOM ACCRETION RATE
      ACCR = SH*PI*(A_G**2.0)*DN*X_G*(8.0*KB*TEMP/(PI*MH))**0.5
C SET GRAIN SURFACE FORMATION OF H2 TO CLASSICAL RATE
C     GR = 5.2E-17*(TEMP*3.33E-3)**0.5
C     ACCR = GR*DN

c TFINAL is the end time in years
c
      TFINAL = 1.0E8
C
c TSTART is the first output point at which abundances are printed
      TSTART = 1.0
      TSTART = TSTART*3.1536E7
      TLAST = TFINAL*3.1536E7
      TOUT = TSTART
      TL = 0.0
C IFREEZE CONTROLS WHETHER FREEZE OUT IS INCLUDED (VIA SUBROUTINE FREEZE)
      IFREEZE = 0
C IF FREEZE OUT ONTO GRAINS IS INCLUDED SET IFREEZE = 1
c      IFREEZE = 1
c
      WRITE(*,*)'Temperature.. ',TEMP
      WRITE(11,*)'Temperature.. ',TEMP
      WRITE(*,*)'Density..     ',DN
      WRITE(11,*)'Density..     ',DN
      WRITE(*,*)'CR Ionisation rate scaling.. ',ZETA
      WRITE(11,*)'CR Ionisation rate scaling.. ',ZETA
      WRITE(*,*)'UV radiation field scaling.. ',RAD
      WRITE(11,*)'UV radiation field scaling.. ',RAD
      WRITE(*,*)'Visual extinction..  ',AV
      WRITE(11,*)'Visual extinction.. ',AV
      WRITE(*,*)'Reading species and initial abundances..'
C
c Input section
c -------------
C read species file.. get species names, masses and initial abundances
C
C  parent species tov H
C  density #/cm3
      WRITE(*,*)'Initially, relative to H2'
      WRITE(11,*)'Initially, relative to H2'
      DO 2 I = 1,N
         READ(USPEC,100)SP(I),Y(I),MASS(I)
         IF(Y(I).GT.0) THEN
            WRITE(*,100)SP(I),2*Y(I),MASS(I)
            WRITE(11,100)SP(I),2*Y(I),MASS(I)
C CONVERT ABUNDANCES TO PER UNIT VOLUME RATHER THAN FRACTIONAL WRT H2
            Y(I)=Y(I)*DN
         ENDIF
 2    CONTINUE
C
      DO 3 I=1,NC
         READ(USPEC,100)SP(N+I),CINP(I),MASS(N+I)
         WRITE(*,100)SP(N+I),CINP(I),MASS(N+I)
         WRITE(11,100)SP(N+I),CINP(I),MASS(N+I)
         TOTAL(I)=CINP(I)*DN
 3    CONTINUE
      WRITE(11,170)TOTAL(1),TOTAL(2)
      
      
      NPAR = 1
 109  READ(UPARENTS,*,END=101) PARENT(NPAR),PABUND(NPAR)
        NPAR = NPAR + 1
      GO TO 109
      
 101  CONTINUE
   


C  SET INITIAL ABUNDANCES
      DO I = 1,NPAR
         DO J = 1,N+NC
            IF(SP(J).EQ.PARENT(I)) THEN
            Y(J) = PABUND(I)*DN
C  LOAD FIRST ELEMENT OF OUTPUT ARRAYS
c             B(J,0) = PABUND(I)* HNR
            WRITE(*,106) SP(J),PABUND(I)
 106        FORMAT(3X,A12,1PE8.2)
            ENDIF
         END DO
      END DO

      
      
      
      
      
      
C
C  *****READ RATE FILE AND EXTRACT RATE DATA FOR EACH REACTION**********

      WRITE(*,*) '...Reading rate file..............'

      CALL READR(FRATES,URATES)
      
 
      
C  CALL SUBROUTINE TO CALCULATE RATE COEFFICIENTS
      CALL RATES
      
c
      WRITE(*,*)'Starting calculation..'
      
C *****INITIALISE INTEGRATION STEP VARIABLES****************************
C  IRUN measures the number of time steps/output points

      T = TSTART
      IRUN = 1
      NTOT = N + NC
      NSPEC = N

C *****INITIALISE DVODE SOLVER VARIABLES********************************

      LIW = NTOT + 30
      LRW = 22 + (9*NTOT) + (2*(NTOT**2))
      ITOL   = 1
      RTOL   = 1.0E-7
      ATOL   = 1.0E-20
      ITASK  = 1
      ISTATE = 1
      IOPT   = 0
      JAC    = 'DUMMYMATRIX'
      MF     = 22

C *****MAIN SOLVER LOOP*************************************************
 6    CONTINUE

C    
c set next output time
c currently uses logarithmic time steps
c
      TOUT=LOG10(TOUT)+0.1
      TOUT=10.0**TOUT
C
c call integrator
c
      CALL DVODE (DIFFUN,NSPEC,Y,T,TOUT,ITOL,RTOL,ATOL,ITASK,
     &            ISTATE,IOPT,RWORK,LRW,IWORK,LIW,JAC,MF,RPAR,IPAR)

      IF(ISTATE.LT.0) ISTATE=1
C     
c save time in array tage
c
      TAGE(IRUN)=TOUT*3.171E-08
      WRITE(*,*)IRUN,TAGE(IRUN),Y(75)/X(2)
      
C store output data in array b for analyse subroutine (before normalisation)
      DO 88 I=1,N
         B(I,IRUN)=Y(I)
 88   CONTINUE
C     
      DO 99 I=1,NC
         B(N+I,IRUN)=X(I)
 99   CONTINUE
C  *****CALL THE SUBROUTINE TO ANALYSE THE OUTPUT***********************

      IF(IANA.EQ.1.AND.IRUN.EQ.51) THEN
      WRITE(*,*) '...Analysing chemistry............'
      CALL ANALYSE(SP,B,K,HNR,ROUT,IRUN,UANA,NTOT,FRATES,URATES)
      WRITE(*,*) '...Resuming model.................'
      IANA = 0
      ENDIF
C     
C store output data in array b NOW IN NORMALISED FORM wrt total density
C
      DO 8 I=1,N
         B(I,IRUN)=Y(I)/DN
 8    CONTINUE
C     
      DO 9 I=1,NC
         B(N+I,IRUN)=X(I)/DN
         IF(I.EQ.2) B(N+I,IRUN)=X(I)
 9    CONTINUE
C
      IRUN=IRUN+1
      IF(TOUT.LE.TLAST) GO TO 6
      TL = TOUT
C     
C 


C     
 11   CONTINUE
c
c loop if not finished
c

c *********************
c
C Output routine
c --------------
C
C first save steady state species file
C     
      WRITE(11,*)'----------------------'
      DO 953 I = 1,NC
         WRITE(11,100)SP(N+I),X(I)/DN
 953  CONTINUE
      WRITE(11,*)'----------------------'
      DO 952 I = 1,N
         WRITE(11,100)SP(I),Y(I)/DN
 952  CONTINUE   
c
      CLOSE(UNIT=11)
C
 7    IRUN=IRUN-1
      IS=1
      IF=10
 13     WRITE(UFRAC,116)(SP(I),I=IS,IF)
      DO 14 I=1,IRUN
         WRITE(UFRAC,118) TAGE(I),(B(J,I),J=IS,IF)
 14   CONTINUE
      WRITE(UFRAC,116)(SP(I),I=IS,IF)
      WRITE(UFRAC,114)
      IS=IS+10
      IF=IF+10
      IF(IF.GT.N+NC) IF=N+NC
      IF(IS.LT.N+NC) GO TO 13
      CLOSE(UNIT=UFRAC)
C
c formats
c
c read species file and write steady state output file
 100  FORMAT(6X,A10,1X,1PE8.2,2X,F5.1)
 170  FORMAT(3X,'TOTAL(1) = ELECTRONS = ',1PE8.2,3X,'TOTAL(2) = H2 = ',
     *  1PE8.2)
c write main output file
 114  FORMAT(//)
 116  FORMAT(3X,'TIME',7X,10(1A9,2X))
 118  FORMAT(1X,11(1PE11.3))
c read ratefile

 102  FORMAT(6X,A9,A9,A9,A9,A9,A8,1PE8.2,1X,0PF5.2,F9.1)
 105  FORMAT(1PE11.5)
 123  FORMAT(I5,1X,A9,A9,A9,A9,A9,A8,1PE8.2,F6.2,F9.1)
 202  FORMAT(5X,A9,A9,A9,A9,A9,A5,A5,1PE8.2,1X,0PF7.2,2X,F8.1,A1,
     *     I5,I5,A5)
c write binned.rates
 223  FORMAT(I4,1X,A9,A9,A9,A9,A9,A5,A5,1PE8.2,1X,0PF7.2,2X,F8.1,A1,
     *     I5,I5,A5)
      STOP
      END
