// cuda_exec.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#define BLOCK_SIZE 256

__global__ void gpu_matrix_vector_mult(cuFloatComplex *matrix, cuFloatComplex *vector, cuFloatComplex *result, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
        for (int j = 0; j < cols; j++) {
            sum = cuCaddf(sum, cuCmulf(matrix[row * cols + j], vector[j]));
        }
        result[row] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: ./cuda_exec <dimensao>\n");
        return 1;
    }

    int dim = atoi(argv[1]);
    int rows = dim, cols = dim;
    size_t size = rows * cols * sizeof(cuFloatComplex);
    size_t vecSize = cols * sizeof(cuFloatComplex);

    cuFloatComplex *h_matrix = (cuFloatComplex*)malloc(size);
    cuFloatComplex *h_vector = (cuFloatComplex*)malloc(vecSize);
    cuFloatComplex *h_result = (cuFloatComplex*)malloc(vecSize);

    FILE *f = fopen("matrix.bin", "rb");
    fread(h_matrix, sizeof(cuFloatComplex), rows * cols, f);
    fclose(f);

    f = fopen("vector.bin", "rb");
    fread(h_vector, sizeof(cuFloatComplex), cols, f);
    fclose(f);

    cuFloatComplex *d_matrix, *d_vector, *d_result;
    cudaMalloc((void**)&d_matrix, size);
    cudaMalloc((void**)&d_vector, vecSize);
    cudaMalloc((void**)&d_result, vecSize);

    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, vecSize, cudaMemcpyHostToDevice);

    int gridSize = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gpu_matrix_vector_mult<<<gridSize, BLOCK_SIZE>>>(d_matrix, d_vector, d_result, rows, cols);

    cudaMemcpy(h_result, d_result, vecSize, cudaMemcpyDeviceToHost);

    FILE *out = fopen("result.bin", "wb");
    fwrite(h_result, sizeof(cuFloatComplex), cols, out);
    fclose(out);

    free(h_matrix); free(h_vector); free(h_result);
    cudaFree(d_matrix); cudaFree(d_vector); cudaFree(d_result);

    return 0;
}
