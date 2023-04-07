      SUBROUTINE ANALYSE(SP,B,K,HNR,TOUT,IRUN,LOUT,N2,FRATES,URATES)
C N2=NO OF SPECIES, NREAC=NO OF REACTIONS,B=RELATIVE ABUNDANCES
C K=REACTION RATES, HNR=H2 NO DENSITY(OR WHATEVER YOUR ABUNDANCES
C ARE RELATIVE TO) LOUT=OUTPUT FILE NUMBER
C FRATES = REACTION RATE FILE NAME
C IURATE = RATE FILE UNIT NUMBER
C NB. ARRAY SIZE OF B(SPECIES NUMBER,TIME STEP NUMBER) MUST BE RIGHT
C  OTHER (1-D) ARRAYS MUST MERELY BE LARGE ENOUGH
C  CHECK COMMON BL4 DEFINITION AND RATE FILE SEPARATOR/FORMAT

      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DOUBLE PRECISION K
      CHARACTER*500 FRATES,SDATA,FANA,TOKEN(100)
      CHARACTER*10 SP(468),RE1(10000),RE2(10000),PR1(10000),
     *            PR2(10000),PR3(10000),PR4(10000),
     *            RD1(10000),RD2(10000),PD1(10000),PD2(10000),
     *            RP1(10000),RP2(10000),PP1(10000),PP2(10000)
      CHARACTER*1 SEP
      DIMENSION D(10000),P(10000),NDR(10000),NPR(10000),B(468,300),
     *          K(10000),INDX(10000),ALF(10000,5),BET(10000,5),
     *          GAM(10000,5),TLOWER(10000,5),TUPPER(10000,5)
      INTEGER NTR(10000),RTYPE(10000),URATES
      COMMON/BL4/ ALF,BET,GAM,TLOWER,TUPPER,RTYPE,NTR,NREAC,ICOR

      FANA = 'analyse.out'

      OPEN(UNIT=LOUT, FILE=FANA)
      OPEN(UNIT=URATES, FILE=FRATES, STATUS='OLD')

C  TOKEN (RATE FILE DATA SUBSTRING SEPARATOR)
      SEP = ':'

      WRITE(LOUT,25)IRUN
   25 FORMAT(1X,'MAIN FORMATION AND DESTRUCTION PROCESSES AT IRUN = ',
     * I4,2X,'TIME STEPS')


      DO 1 NR=1,NREAC

      READ(URATES,101) SDATA
 101  FORMAT(A)

C  INITIALISE TOKEN COUNTER
      I = 0

C  BEGIN TOKENISATION OF THE DATA STRING 
 102  I = I + 1
C  FIND TOKEN POSITION
         L = INDEX(SDATA,SEP) - 1
C  POSITIVE LENGTH TOKEN
         IF (L.GT.0) THEN
            TOKEN(I) = SDATA(:L)
            L = L + 1
C  ZERO-LENGTH TOKEN
         ELSE IF (L.EQ.0) THEN
            TOKEN(I) = ""
            L = 1
C  NO MORE TOKENS FOUND         
         ELSE IF (L.EQ.-1) THEN
            TOKEN(I) = SDATA(1:)
            L = 0
         END IF
C  REMOVE PREVIOUS TOKEN FROM DATA STRING
         SDATA = SDATA(L+1:)
C  CONTINUE LOOPING UNTIL NO MORE TOKENS REMAIN
      IF (I.LE.100.AND.L.GT.0) GO TO 102 

C  IF LESS THAN 14 TOKENS WERE READ, REACTION DATA IS INCOMPLETE
      IF(I.LT.14) CALL DIE
     *   ("Unable to interpret reaction string -- check rate file     ")


C  STORE INDEX, REACTANTS AND PRODUCTS
      READ(TOKEN(1),*) INDX(NR)
      RE1(NR) = TOKEN(3)
      RE2(NR) = TOKEN(4)
      PR1(NR) = TOKEN(5)
      PR2(NR) = TOKEN(6)
      PR3(NR) = TOKEN(7)
      PR4(NR) = TOKEN(8)

    1 CONTINUE


      NSPEC=0
      DO 2 I=1,N2
      NSPEC=NSPEC+1
      DTOT=0.0
      PTOT=0.0
      WRITE(LOUT,100)NSPEC,SP(I)
  100 FORMAT(/,10X,I3,4X,"*",A12,10X)
C      WRITE(*,101,ADVANCE="NO") SP(I)
      ID=0
      IP=0
      DO 3 J=1,NREAC
      DO 4 M=1,N2
      IF(RE1(J).EQ.SP(M)) Y1=B(M,IRUN)
      IF(RE2(J).EQ.SP(M)) Y2=B(M,IRUN)
    4 CONTINUE
      RMULT=0.0
      IF(RE1(J).EQ.SP(I).OR.RE2(J).EQ.SP(I)) THEN
      IF(RE1(J).EQ.SP(I)) RMULT=1.0
      IF(RE2(J).EQ.SP(I)) RMULT=RMULT+1.0
      IF(PR1(J).EQ.SP(I)) RMULT=RMULT-1.0
      IF(PR2(J).EQ.SP(I)) RMULT=RMULT-1.0
      IF(PR3(J).EQ.SP(I)) RMULT=RMULT-1.0
      IF(PR4(J).EQ.SP(I)) RMULT=RMULT-1.0
      ID=ID+1
      IF(RE2(J).EQ.'CRP'.OR.RE2(J).EQ.'PHOTON'.OR.RE2(J).EQ.
     *      'CRPHOT') THEN
      D(ID)=K(J)*Y1
      ELSE
      D(ID)=K(J)*Y1*Y2*HNR*RMULT
      END IF
      DTOT=DTOT+D(ID)
      NDR(ID)=INDX(J)
      RD1(ID)=RE1(J)
      RD2(ID)=RE2(J)
      PD1(ID)=PR1(J)
      PD2(ID)=PR2(J)
      END IF
      RMULT=0.0
      IF(PR1(J).EQ.SP(I).OR.PR2(J).EQ.SP(I).OR.
     *   PR3(J).EQ.SP(I)) THEN
      IF(PR1(J).EQ.SP(I)) RMULT=RMULT+1.0
      IF(PR2(J).EQ.SP(I)) RMULT=RMULT+1.0
      IF(PR3(J).EQ.SP(I)) RMULT=RMULT+1.0
      IF(PR4(J).EQ.SP(I)) RMULT=RMULT+1.0
      IF(RE1(J).EQ.SP(I)) RMULT=RMULT-1.0
      IF(RE2(J).EQ.SP(I)) RMULT=RMULT-1.0
      IP=IP+1
      IF(RE2(J).EQ.'PHOTON'.OR.RE2(J).EQ.'CRP'.OR.RE2(J).EQ.
     *     'CRPHOT') THEN
      P(IP)=K(J)*Y1*RMULT
      ELSE
      P(IP)=K(J)*Y1*Y2*HNR*RMULT
      END IF
      PTOT=PTOT+P(IP)
      NPR(IP)=INDX(J)
      RP1(IP)=RE1(J)
      RP2(IP)=RE2(J)
      PP1(IP)=PR1(J)
      PP2(IP)=PR2(J)
      END IF
    3 CONTINUE
      DO 5 IL=1,ID
      DPC=100*(D(IL)/DTOT)
      IF(DPC.GT.5.0) THEN
      WRITE(LOUT,300)NDR(IL),RD1(IL),RD2(IL),PD1(IL),PD2(IL),-NINT(DPC)
      END IF
    5 CONTINUE
      DO 6 IL=1,IP
      PPC=100*(P(IL)/PTOT)
      IF(PPC.GT.5.0) THEN
      WRITE(LOUT,300)NPR(IL),RP1(IL),RP2(IL),PP1(IL),PP2(IL),NINT(PPC)
      END IF
    6 CONTINUE
      WRITE(LOUT,400)DTOT,PTOT
    2 CONTINUE
  400 FORMAT(10X,'DRATE= ',E9.3,5X,'PRATE= ',E9.3)
  300 FORMAT(2X,I4,4(1X,A12),I4,'%')
  
      CLOSE(LOUT)
      
      WRITE(*,501) FANA
  501 FORMAT(/,' ...Main formation and destruction reactions written to',
     *   1X,A20)
     
      RETURN
      END