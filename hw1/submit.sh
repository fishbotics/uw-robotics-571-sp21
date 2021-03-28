#!/usr/bin/env bash
rm -rf submit/
mkdir -p submit

prepare () {
    cp $1 submit/
}

echo "Creating tarball..."
prepare ../nn/layers/losses/softmax_cross_entropy_loss_layer.py
prepare ../nn/layers/linear_layer.py
prepare ../nn/layers/relu_layer.py
prepare ../nn/optimizers/sgd_optimizer.py

tar cvzf submit.tar.gz submit
rm -rf submit/
echo "Done. Please upload submit.tar.gz to Canvas."
