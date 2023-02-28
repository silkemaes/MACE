!     Python C-bindings for CS model
!     ------------------------------
!     Currently the old Fortran main has been split into two subroutines:
!     (1) setup: mostly reading in data
!     (2) solve: performing the chemical evolution (solving the ODEs)
!     This file only contains the solve
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine solve(Yi,Ki) bind(c, name="solve")
      use iso_c_binding
            
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      CHARACTER*10  SP(468),CSP(2),RE1,RE2,P1,P2,P3,P4,PARENT(200)
      CHARACTER*5   FLG
      CHARACTER*1   SR
      CHARACTER*11  JAC
      CHARACTER*500 FRATES,FSPECS,FOUTF,FOUTN,FPARENTS
      INTEGER OU
      INTEGER IWORK(1025),I,J,FSPEC,FRATE,FFRAC,FNUM,UFRAC,URATES,USPEC,
     *     NSPEC,NCONS,NPAR,ICO,ICOR,IRUN,ITOL,ITASK,ISTATE,IOPT,LRW,
     *     LIW,MF,NREAC,IS,IF,LI,LTAG,NEXTRA,IFREEZE,IANA,UANA,
     *     UPARENTS
      DOUBLE PRECISION K(10000),MASS(468),NGD,RTOL,ATOL,TSTART,Y(466),
     *     T,RWORK(2000000),TOTAL(10),B(468,300),CINP(2),TAGE(300),SH,
     *     X(10),GR,DN,TFINAL,TLAST,KJ,ACCR,HNR,PI,KB,MH,MU,PABUND(200)
          
!     NC = Number of conserved species, NR = Number of reactions, N = Number of species (ODEs)
      PARAMETER(OU=8,NC=2,NR=6173,N=466)
          
      EXTERNAL DIFFUN

!     PHYSICAL CONSTANTS
      DATA PI, MH, MU, KB / 3.1415927, 1.6605E-24, 2.2, 1.3807E-16/
      
!     Define common blocks
      COMMON/BL1/ K,X,TOTAL,GR,DN,ACCR,HNR,IFREEZE
      COMMON/BL10/ MASS,SH
      COMMON/BL3/ Y,X_G,A_G,TEMP,AV,ZETA,ALBEDO,RAD

      real(c_double), intent(inout) :: Yi( 466)
      real(c_double), intent(inout) :: Ki(10000)
      
      Y = Yi
      K = Ki

      WRITE(*,*)'Starting calculation..'
      
!     *****INITIALISE INTEGRATION STEP VARIABLES****************************
!     IRUN measures the number of time steps/output points
      T     = TSTART
      IRUN  = 1
      NTOT  = N + NC
      NSPEC = N

!     *****INITIALISE DVODE SOLVER VARIABLES********************************
      LIW    = NTOT + 30
      LRW    = 22 + (9*NTOT) + (2*(NTOT**2))
      ITOL   = 1
      RTOL   = 1.0E-7
      ATOL   = 1.0E-20
      ITASK  = 1
      ISTATE = 1
      IOPT   = 0
      JAC    = 'DUMMYMATRIX'
      MF     = 22
      
!     *****MAIN SOLVER LOOP*************************************************
6     CONTINUE

!     set next output time
!     currently uses logarithmic time steps
      TOUT = LOG10(TOUT)+0.1
      TOUT = 10.0**TOUT

!     call integrator
      CALL DVODE (DIFFUN,NSPEC,Y,T,TOUT,ITOL,RTOL,ATOL,ITASK,
     &            ISTATE,IOPT,RWORK,LRW,IWORK,LIW,JAC,MF,RPAR,IPAR)

      IF(ISTATE.LT.0) ISTATE=1
     
!     save time in array tage
      TAGE(IRUN)=TOUT*3.171E-08
      WRITE(*,*)IRUN,TAGE(IRUN),Y(75)/X(2)
      
!     store output data in array b for analyse subroutine (before normalisation)
      DO 88 I=1,N
          B(I,IRUN)=Y(I)
 88   CONTINUE

      DO 99 I=1,NC
          B(N+I,IRUN)=X(I)
 99   CONTINUE
 
 
! !     *****CALL THE SUBROUTINE TO ANALYSE THE OUTPUT***********************
!       IF(IANA.EQ.1.AND.IRUN.EQ.51) THEN
!           WRITE(*,*) '...Analysing chemistry............'
!           CALL ANALYSE(SP,B,K,HNR,ROUT,IRUN,UANA,NTOT,FRATES,URATES)
!           WRITE(*,*) '...Resuming model.................'
!           IANA = 0
!       ENDIF

!     store output data in array b NOW IN NORMALISED FORM wrt total density
      DO 8 I=1,N
          B(I,IRUN)=Y(I)/DN
 8    CONTINUE

      DO 9 I=1,NC
          B(N+I,IRUN)=X(I)/DN
          IF(I.EQ.2) B(N+I,IRUN)=X(I)
 9    CONTINUE

!     loop if not finished
      IRUN = IRUN+1
      IF(TOUT.LE.TLAST) GO TO 6
         
      TL = TOUT
         
 11   CONTINUE


      ! Explicit copies required to avoid fuss with common blocks
      Yi = Y
      Ki = K

      end subroutine solve
