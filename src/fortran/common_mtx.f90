MODULE common_mtx
!=======================================================================
!
! [PURPOSE:] Eigenvalue decomposition for real symmetric matrices.
!
!   mtx_eigen : eigendecomposition via LAPACK dsyev.
!               Eigenvalues returned in DESCENDING order.
!               Near-zero eigenvalues (below machine-epsilon threshold)
!               are set to zero; nrank_eff counts positive ones.
!
! [DEPENDENCY:] LAPACK (dsyev) — provided by conda's openblas or lapack
!               package; linked via -llapack in build_fortran.sh.
!
!=======================================================================
  USE common_tools
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: mtx_eigen

CONTAINS

!=======================================================================
SUBROUTINE mtx_eigen(imode, n, a, eival, eivec, nrank_eff)
!=======================================================================
! Eigenvalue decomposition of a real symmetric (n x n) matrix.
!
!  INPUT
!    imode      : 0 = eigenvalues only; 1 = eigenvalues + eigenvectors
!    n          : matrix dimension
!    a(n,n)     : input symmetric matrix (not modified)
!
!  OUTPUT
!    eival(n)   : eigenvalues in DESCENDING order (largest first)
!    eivec(n,n) : eigenvectors as columns, matching eival order
!    nrank_eff  : number of eigenvalues above the zero threshold
!
!=======================================================================
  IMPLICIT NONE
  INTEGER,      INTENT(IN)  :: imode, n
  REAL(r_size), INTENT(IN)  :: a(n,n)
  REAL(r_size), INTENT(OUT) :: eival(n)
  REAL(r_size), INTENT(OUT) :: eivec(n,n)
  INTEGER,      INTENT(OUT) :: nrank_eff

  ! LAPACK workspace
  REAL(r_dble) :: a8(n,n)
  REAL(r_dble) :: w8(n)
  REAL(r_dble) :: work(3*n)
  INTEGER      :: lwork, info, i
  CHARACTER(1) :: jobz

  ! dsyev returns eigenvalues in ASCENDING order; we reverse below.
  jobz  = MERGE('V', 'N', imode /= 0)
  lwork = 3*n
  a8    = REAL(a, r_dble)

  CALL dsyev(jobz, 'U', n, a8, n, w8, work, lwork, info)

  IF (info /= 0) THEN
    WRITE(*,*) 'ERROR (mtx_eigen): dsyev returned info = ', info
  END IF

  ! ---- reverse to descending order -------------------------------------
  DO i = 1, n
    eival(i)    = REAL(w8(n+1-i),        r_size)
    eivec(:, i) = REAL(a8(:, n+1-i),     r_size)
  END DO

  ! ---- count effective rank (positive eigenvalues) --------------------
  nrank_eff = n
  IF (w8(n) > 0.0d0) THEN
    DO i = 1, n
      IF (w8(i) < ABS(w8(n)) * SQRT(EPSILON(w8(n)))) THEN
        nrank_eff = nrank_eff - 1
      END IF
    END DO
  ELSE
    WRITE(*,*) 'WARNING (mtx_eigen): all eigenvalues <= 0'
  END IF

END SUBROUTINE mtx_eigen

END MODULE common_mtx
