
This repository contains original Caffe-BLVC code with multi GPU feature.

Modifications Done:

1. Naive Caffe implementations for Convolutions and Batch-Normalization are memory hungry. Hence, A "MemoryBank" class has been added and is declared as a member in Layer.hpp. In this way, any Layer can demand memory which can be resued or non-reusable by specifying a bank_id and a shareable flag.

2. Col2Im and Im2Col implementations are also memory hungry. For feature maps having large spatial dimensions, Col2Im and Im2Col demands excessively large memory. e.g. for a kernel size 3x3, stride = 1, Pad = 0 and feature_map of size 1x1600x256x256, it would require approx 3.7 GB.
