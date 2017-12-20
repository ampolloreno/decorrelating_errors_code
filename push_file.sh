#!/bin/bash
for node in node2
do
  scp -r $1 $node:$1
done
