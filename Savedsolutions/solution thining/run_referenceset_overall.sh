#!/bin/bash
MYPATH=""
FEATURE=600K/ # $2/
FILENAME=variables_outcomes  #  $3
JAVA_ARGS="-classpath ./MOEAFramework-2.9-Demo.jar:."
EPS=0.16
NOBJS=5
NVARS=24  #$1
NSEEDS=1  # 15
SEEDS=$(seq 1 ${NSEEDS})

for (S in ${SEEDS}){
	echo "#" >> ${MYPATH}${FEATURE}${FILENAME}_S${S}.set
	
}
java ${JAVA_ARGS} org.moeaframework.analysis.sensitivity.ResultFileMerger -e ${EPS} -d ${NOBJS} -v ${NVARS} -r ${FEATURE}${FILENAME}*.set -r ${MYPATH}${FEATURE}${FILENAME}.reference 