#!/bin/bash
# Revisar antes de ejecutar. Mueve las REF viejas a *_v1.
set -euo pipefail
cd /home/jorge.gacitua/salidas/WRF_Single_Cycle_Assimilation/data
[ -d WS_A_multiobs_1800_LOC2.0_REF ] && [ ! -e WS_A_multiobs_1800_LOC2.0_REF_v1 ] && mv WS_A_multiobs_1800_LOC2.0_REF WS_A_multiobs_1800_LOC2.0_REF_v1
[ -d WS_A_multiobs_1900_LOC2.0_REF ] && [ ! -e WS_A_multiobs_1900_LOC2.0_REF_v1 ] && mv WS_A_multiobs_1900_LOC2.0_REF WS_A_multiobs_1900_LOC2.0_REF_v1
[ -d WS_A_multiobs_2000_LOC2.0_REF ] && [ ! -e WS_A_multiobs_2000_LOC2.0_REF_v1 ] && mv WS_A_multiobs_2000_LOC2.0_REF WS_A_multiobs_2000_LOC2.0_REF_v1
[ -d WS_A_multiobs_2100_LOC2.0_REF ] && [ ! -e WS_A_multiobs_2100_LOC2.0_REF_v1 ] && mv WS_A_multiobs_2100_LOC2.0_REF WS_A_multiobs_2100_LOC2.0_REF_v1
[ -d WS_B_multiobs_1800_LOC2.0_REF ] && [ ! -e WS_B_multiobs_1800_LOC2.0_REF_v1 ] && mv WS_B_multiobs_1800_LOC2.0_REF WS_B_multiobs_1800_LOC2.0_REF_v1
[ -d WS_B_multiobs_1900_LOC2.0_REF ] && [ ! -e WS_B_multiobs_1900_LOC2.0_REF_v1 ] && mv WS_B_multiobs_1900_LOC2.0_REF WS_B_multiobs_1900_LOC2.0_REF_v1
[ -d WS_B_multiobs_2000_LOC2.0_REF ] && [ ! -e WS_B_multiobs_2000_LOC2.0_REF_v1 ] && mv WS_B_multiobs_2000_LOC2.0_REF WS_B_multiobs_2000_LOC2.0_REF_v1
[ -d WS_B_multiobs_2100_LOC2.0_REF ] && [ ! -e WS_B_multiobs_2100_LOC2.0_REF_v1 ] && mv WS_B_multiobs_2100_LOC2.0_REF WS_B_multiobs_2100_LOC2.0_REF_v1
[ -d WS_C_multiobs_1800_LOC2.0_REF ] && [ ! -e WS_C_multiobs_1800_LOC2.0_REF_v1 ] && mv WS_C_multiobs_1800_LOC2.0_REF WS_C_multiobs_1800_LOC2.0_REF_v1
[ -d WS_C_multiobs_1900_LOC2.0_REF ] && [ ! -e WS_C_multiobs_1900_LOC2.0_REF_v1 ] && mv WS_C_multiobs_1900_LOC2.0_REF WS_C_multiobs_1900_LOC2.0_REF_v1
[ -d WS_C_multiobs_2000_LOC2.0_REF ] && [ ! -e WS_C_multiobs_2000_LOC2.0_REF_v1 ] && mv WS_C_multiobs_2000_LOC2.0_REF WS_C_multiobs_2000_LOC2.0_REF_v1
[ -d WS_C_multiobs_2100_LOC2.0_REF ] && [ ! -e WS_C_multiobs_2100_LOC2.0_REF_v1 ] && mv WS_C_multiobs_2100_LOC2.0_REF WS_C_multiobs_2100_LOC2.0_REF_v1
