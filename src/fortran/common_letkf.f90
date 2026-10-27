MODULE common_letkf
!=======================================================================
!
! [PURPOSE:] Local Ensemble Transform Kalman Filter (LETKF) core.
!
!   letkf_core : one LETKF analysis step at a single grid point.
!
! [ALGORITHM:] Hunt et al. 2007, Physica D 230, 112-126.
!
! [DEPENDENCY:] common_tools, common_mtx, BLAS (dgemm).
!
!=======================================================================
  USE common_tools
  USE common_mtx
  IMPLICIT NONE
  PUBLIC :: letkf_core

CONTAINS

!=======================================================================
SUBROUTINE letkf_core(ne, nobsl, hdxb, rloc, dep, parm_infl, &
                       trans, transm, pao, minfl)
!=======================================================================
!
!  INPUT
!    ne              : ensemble size
!    nobsl           : number of observations at this grid point
!    hdxb(nobsl,ne)  : H * ensemble perturbations  (obs-space)
!    rloc(nobsl)     : localised observation error (oerr / weight)
!    dep(nobsl)      : observation departure  yo - H(xb_mean)
!    parm_infl       : covariance inflation parameter (rho >= 1)
!    minfl           : minimum allowed inflation
!
!  OUTPUT
!    trans(ne,ne)    : ensemble weight perturbation matrix  T
!    transm(ne)      : ensemble weight mean vector          w_a
!    pao(ne,ne)      : analysis ensemble covariance in ensemble space
!
!=======================================================================
  IMPLICIT NONE
  INTEGER,      INTENT(IN)    :: ne, nobsl
  REAL(r_size), INTENT(IN)    :: hdxb(nobsl, ne)
  REAL(r_size), INTENT(IN)    :: rloc(nobsl)
  REAL(r_size), INTENT(IN)    :: dep(nobsl)
  REAL(r_size), INTENT(INOUT) :: parm_infl
  REAL(r_size), INTENT(OUT)   :: trans(ne, ne)
  REAL(r_size), INTENT(OUT)   :: transm(ne)
  REAL(r_size), INTENT(OUT)   :: pao(ne, ne)
  REAL(r_size), INTENT(IN)    :: minfl

  REAL(r_size) :: hdxb_rinv(nobsl, ne)
  REAL(r_size) :: eivec(ne, ne)
  REAL(r_size) :: eival(ne)
  REAL(r_size) :: pa(ne, ne)
  REAL(r_size) :: work1(ne, ne)
  REAL(r_size) :: work2(ne, nobsl)
  REAL(r_size) :: work3(ne)
  REAL(r_size) :: rho
  INTEGER      :: i, j, nrank

  ! ---- no observations: return inflated identity ----------------------
  IF (nobsl == 0) THEN
    trans  = 0.0d0
    transm = 0.0d0
    pao    = 0.0d0
    DO i = 1, ne
      trans(i,i) = SQRT(parm_infl)
      pao(i,i)   = parm_infl / REAL(ne-1, r_size)
    END DO
    RETURN
  END IF

  ! ---- enforce minimum inflation --------------------------------------
  IF (minfl > 0.0d0 .AND. parm_infl < minfl) parm_infl = minfl
  rho = 1.0d0 / parm_infl

  ! ---- hdxb * R^{-1}  (R-localisation: rloc already = oerr/weight) ---
  DO j = 1, ne
    DO i = 1, nobsl
      hdxb_rinv(i,j) = hdxb(i,j) / rloc(i)
    END DO
  END DO

  ! ---- C = hdxb^T R^{-1} hdxb  (ne x ne) -----------------------------
  CALL dgemm('T','N', ne, ne, nobsl, &
             1.0d0, hdxb_rinv, nobsl, hdxb, nobsl, &
             0.0d0, work1, ne)

  ! ---- C + (ne-1)/rho * I  (add inflation) ----------------------------
  DO i = 1, ne
    work1(i,i) = work1(i,i) + REAL(ne-1, r_size) * rho
  END DO

  ! ---- eigendecomposition of C ----------------------------------------
  nrank = ne
  CALL mtx_eigen(1, ne, work1, eival, eivec, nrank)

  ! ---- Pa = eivec * diag(1/eival) * eivec^T  --------------------------
  DO j = 1, ne
    DO i = 1, ne
      work1(i,j) = eivec(i,j) / eival(j)
    END DO
  END DO
  CALL dgemm('N','T', ne, ne, ne, &
             1.0d0, work1, ne, eivec, ne, &
             0.0d0, pa, ne)

  ! ---- Pa * hdxb_rinv^T  (ne x nobsl) ---------------------------------
  CALL dgemm('N','T', ne, nobsl, ne, &
             1.0d0, pa, ne, hdxb_rinv, nobsl, &
             0.0d0, work2, ne)

  ! ---- w_a = Pa * hdxb_rinv^T * dep  (mean weight vector) ------------
  work3 = 0.0d0
  DO j = 1, nobsl
    DO i = 1, ne
      work3(i) = work3(i) + work2(i,j) * dep(j)
    END DO
  END DO
  transm = work3

  ! ---- T = eivec * diag(sqrt((ne-1)/eival)) * eivec^T  ---------------
  DO j = 1, ne
    rho = SQRT(REAL(ne-1, r_size) / eival(j))
    DO i = 1, ne
      work1(i,j) = eivec(i,j) * rho
    END DO
  END DO
  CALL dgemm('N','T', ne, ne, ne, &
             1.0d0, work1, ne, eivec, ne, &
             0.0d0, trans, ne)

  pao = pa

END SUBROUTINE letkf_core

END MODULE common_letkf
