# Human Abnormal Behaviors Detection Using X3D

Implementation of X3D for violence detection

<table style="border:0px">
   <tr>
       <td><img src="x3d_from_hvd_official/assets/video01.gif" frame=void rules=none></td>
       <td><img src="x3d_from_hvd_official/assets/video02.gif" frame=void rules=none></td>
   </tr>
   <tr>
      <td><div aligh=center><img src="x3d_from_hvd_official/assets/video06_nonef.gif" frame=void rules=none></td>
      <td><div aligh=center><img src="x3d_from_hvd_official/assets/video08_nonef.gif" frame=void rules=none></td>
   </tr>
</table>




# To install Horovod with NCCL support on the cluster:

1.Activate the python virtual environment

2.Use pip to install CMake

3.Use module load to load gcc8/8.4.0 and nccl2-cuda10.2-gcc/2.7.8 and openmpi-geib-cuda10.2-gcc/4.0.5 (which are already installed on the cluster)

4.Since gcc8 and NCCL are already installed on the cluster, run this command (make sure to specify your path to python virtual environment):

    HOROVOD_WITH_GLOO=1 HOROVOD_WITH_MPI=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_CMAKE={Path to python virtual environment}/bin/cmake HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_HOME=/cm/shared/apps/nccl2-cuda10.2-gcc/2.7.8 pip install --no-cache-dir horovod[pytorch]

To check if horovod is installed with NCCL correctly, use:

    horovodrun --check-build

if it is installed correctly, the information should be shown as below:

    Available Frameworks:
        [ ] TensorFlow
        [X] PyTorch
        [ ] MXNet

    Available Controllers:
        [X] MPI
        [X] Gloo

    Available Tensor Operations:
        [X] NCCL
        [ ] DDL
        [ ] CCL
        [X] MPI
        [X] Gloo 
