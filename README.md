
This repository contains original Caffe-BLVC code with multi GPU feature.

Modifications Done:

1. Naive Caffe implementations for Convolutions and Batch-Normalization are memory hungry. Hence, A "MemoryBank" class has been added and is declared as a member in Layer.hpp. In this way, any Layer can demand memory which can be resued or non-reusable by specifying a bank_id and a shareable flag.

2. Col2Im and Im2Col implementations are also memory hungry. For feature maps having large spatial dimensions, Col2Im and Im2Col demands excessively large memory. e.g. for a kernel size 3x3, stride = 1, Pad = 0 and feature_map of size 1x1600x256x256, it would require approx 3.7 GB. However, preallocating such a large memory doesn't show improvement in convolution operation. Hence, to avoid this, the convolution is broken into smaller operations during forward and backward pass.

    Forward Pass: In this case, only a chunk of spatial locations out of HxW is processed.
    Backward pass: While in this case, a chunk of channels is processed with all HxW locations.
    
    To achieve this, the caffe operations: caffe_gpu_gemm is extended as "caffe_gpu_gemm_ld" in which one can specify the lda,ldb and ldc required for cublas operations.
    
3. A memory optimization layer has been added. It can be inserted between two consecutive layers. During Forward pass the layer doesn't optimize tha data memory but optimizes the gradient memory by specifying a bank id.

4. Also, an optimized version of both Split and Concat layer has been added. Orignally split layer keeps different gradient memory for each split. But split optimized layer simply assign the same memories for data and diff. To accomplish its effect, a flag "is_shared" in class blob has been added. By checking its value, any layer can accumulate the gradient in memory. This functionality has been added for Convolution, eltwise and pooling layer. User can add this functionality by looking in the source files of the above layers.
   
    Similarly, ConcatOptimized layer allocates a contigous memory block to all input blobs. It can acheive the memory allocation recursively i.e. in case of splitted bottoms.
    
5. Finally, the Memory eater: Batch normalization. Two features in the batch normalization layer has been added

   i) BNLayer is extended for multi gpu syncing. Earlier a batch size of 3 on one GPU was only 3 i.e. batch normalization for intermediate feature maps was different in different GPUs. But with this feature, batch size is actually multiplied with number of GPUs and all examples across all GPUs have same batch statistics.
   
   ii) Secondly, when BN parameters are unfrozen, normalized inputs are preserved for backwards pass. This approach requires double the memory which a network would require without BN. Hence, Considering the high bandwidth of CPU-GPU memory transfer, normalized inputs are computed in a common memory space and is sent to CPU. Later in the backward pass, it is copied from CPU to GPU. A slight overhead in training time is added however test time remains unaffected.
   
    
Advantages:

1. Any network architecture is benefitted as a very less amount of gradient memory is required due to memory reuse.
2. ResNet, DenseNets are greatly benefitted with this approach and one can even go more deeper than the reported depths in their respective papers.
  
