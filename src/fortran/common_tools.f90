MODULE common_tools
!=======================================================================
!
! [PURPOSE:] Precision parameters, physical constants, and basic
!            statistical utilities used by the DA modules.
!
!=======================================================================
  IMPLICIT NONE
  PUBLIC

  ! ---- floating-point precision kinds ----------------------------------
  INTEGER, PARAMETER :: r_size = kind(0.0d0)   ! double precision
  INTEGER, PARAMETER :: r_dble = kind(0.0d0)   ! alias (kept for clarity)
  INTEGER, PARAMETER :: r_sngl = kind(0.0e0)   ! single precision

  ! ---- physical constants ----------------------------------------------
  REAL(r_size), PARAMETER :: pi      = 3.14159265358979d0
  REAL(r_size), PARAMETER :: gg      = 9.81d0       ! gravity        [m/s^2]
  REAL(r_size), PARAMETER :: rd      = 287.0d0      ! dry air gas constant [J/kg/K]
  REAL(r_size), PARAMETER :: cp      = 1005.7d0     ! specific heat  [J/kg/K]
  REAL(r_size), PARAMETER :: re      = 6371300.0d0  ! Earth radius   [m]
  REAL(r_size), PARAMETER :: t0c     = 273.15d0     ! 0 C in Kelvin

CONTAINS

!=======================================================================
SUBROUTINE com_mean(ndim, var, amean)
! Arithmetic mean of a 1-D array.
  IMPLICIT NONE
  INTEGER,      INTENT(IN)  :: ndim
  REAL(r_size), INTENT(IN)  :: var(ndim)
  REAL(r_size), INTENT(OUT) :: amean
  amean = SUM(var) / REAL(ndim, r_size)
END SUBROUTINE com_mean

END MODULE common_tools
