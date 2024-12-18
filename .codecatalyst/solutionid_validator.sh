#!/bin/sh

#set -e
export SOLUTIONID="SO9251"

echo "grep -nr --exclude-dir='.codecatalyst' "${SOLUTIONID}" ./.."
result=$(grep -nr --exclude-dir='.codecatalyst' "${SOLUTIONID}" ./..)
if [ $? -eq 0 ] 
then
  echo "Solution ID ${SOLUTIONID} found\n"
  echo "$result"
  exit 0
else
  echo "Solution ID ${SOLUTIONID} not found"
  exit 1
fi

export result
