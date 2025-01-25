echo "starting building"
nvcc qdbmp.c cpu.cu -o cpu
echo "build finished"
echo "launching"
./cpu "480-360-sample.bmp"