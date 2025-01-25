echo "starting building"
nvcc qdbmp.c gpu.cu -o gpu
echo "build finished"
echo "launching"
./gpu "1920-1080-sample.bmp"