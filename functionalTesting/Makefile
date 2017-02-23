SOURCE := uma.cu
OBJECT := uma

uma: uma.cu
	nvcc -O3 -gencode arch=compute_35,code=sm_35      \
             -gencode arch=compute_50,code=sm_50      \
             -gencode arch=compute_50,code=compute_50 \
             -gencode arch=compute_60,code=sm_60      \
             -gencode arch=compute_61,code=sm_61      \
             uma.cu -o uma

.PHONY: clean
clean:
	@rm -rf ${OBJECT}
