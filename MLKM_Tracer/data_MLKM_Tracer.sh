#!/bin/bash

N=$1;
w=$(echo $2 | awk '{printf("%.2f", $1)}');
K=$(echo $3 | awk '{printf("%.4f", $1)}');
ft=$4;
dt=$(echo $5 | awk '{printf("%.3f", $1)}');
e=$6;
L=$7;
Emax=1000;
data=./Phaset_w${w}_N${N}_K${K}_ft${ft}_dt${dt}_E${e}.dat

mkdir -p ./N${N}K${K}L${L}


Tmax=$(echo ${ft} ${dt} ${L} ${Emax} | awk '{print int(($1/$2 - $3 - 1)/$4)}')
# T=$8;
for((T=0;T<${Tmax};T++)); do
    for((E=0;E<${Emax};E++));do
        awk -v T=${T} -v E=${E} -v L=${L} -v Emax=${Emax} 'T*Emax+E+1<=NR&&NR<=T*Emax+E+L{printf("%s ", $0)}' ${data}
        echo ''
    done > ./N${N}K${K}L${L}/Phaset_w${w}_N${N}_K${K}_ft${ft}_dt${dt}_E${e}_${L}_${T}.dat
    
    awk -v T=${T} -v E=${E} -v L=${L} -v Emax=${Emax} 'T*Emax+L+1<=NR&&NR<=T*Emax+L+Emax' ${data} > ./N${N}K${K}L${L}/Phaset_w${w}_N${N}_K${K}_ft${ft}_dt${dt}_E${e}_${L}_${T}.label

done