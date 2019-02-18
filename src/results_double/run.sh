#!/bin/bash
dir=results_double
params=$dir/params
env=$dir/simple_rand
N=1000

bin/ql    $params $env $N > $dir/ql
bin/dql   $params $env $N > $dir/dql
bin/qrl   $params $env $N > $dir/qrl
bin/qqrl  $params $env $N > $dir/qqrl
bin/dqrl  $params $env $N > $dir/dqrl
bin/dqqrl $params $env $N > $dir/dqqrl
