#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256  // Ajuste baseado no hardware

// Kernel GPU: Multiplicação de matriz por vetor
__global__ void gpu_matrix_vector_mult(float *matrix, float *vector, float *result, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += matrix[row * cols + j] * vector[j];
        }
        result[row] = sum;
    }
}

// Função CPU para comparação
void cpu_matrix_vector_mult(float *matrix, float *vector, float *result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j] * vector[j];
        }
        result[i] = sum;
    }
}

int main() {
    int rows, cols;

    // Solicita ao usuário as dimensões da matriz e do vetor
    printf("Digite o número de linhas da matriz: ");
    scanf("%d", &rows);
    printf("Digite o número de colunas da matriz: ");
    scanf("%d", &cols);

    size_t matrixSize = rows * cols * sizeof(float);
    size_t vectorSize = cols * sizeof(float);
    size_t resultSize = rows * sizeof(float);

    // Alocação na CPU
    float *h_matrix = (float*)malloc(matrixSize);
    float *h_vector = (float*)malloc(vectorSize);
    float *h_resultCPU = (float*)malloc(resultSize);
    float *h_resultGPU = (float*)malloc(resultSize);

    // Solicita ao usuário os valores da matriz
    printf("Digite os valores da matriz (%d x %d):\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("Matriz[%d][%d]: ", i, j);
            scanf("%f", &h_matrix[i * cols + j]);
        }
    }

    // Solicita ao usuário os valores do vetor
    printf("Digite os valores do vetor (%d elementos):\n", cols);
    for (int i = 0; i < cols; i++) {
        printf("Vetor[%d]: ", i);
        scanf("%f", &h_vector[i]);
    }

    // Alocação na GPU
    float *d_matrix, *d_vector, *d_result;
    cudaMalloc((void**)&d_matrix, matrixSize);
    cudaMalloc((void**)&d_vector, vectorSize);
    cudaMalloc((void**)&d_result, resultSize);

    // Cópia CPU -> GPU
    cudaMemcpy(d_matrix, h_matrix, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, vectorSize, cudaMemcpyHostToDevice);

    // Tempo CPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cpu_matrix_vector_mult(h_matrix, h_vector, h_resultCPU, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cpuTime;
    cudaEventElapsedTime(&cpuTime, start, stop);

    // Tempo GPU
    int gridSize = (rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEventRecord(start);
    gpu_matrix_vector_mult<<<gridSize, BLOCK_SIZE>>>(d_matrix, d_vector, d_result, rows, cols);
    cudaMemcpy(h_resultGPU, d_result, resultSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpuTime;
    cudaEventElapsedTime(&gpuTime, start, stop);

    // Exibe os resultados
    printf("\nResultado da multiplicação (CPU):\n");
    for (int i = 0; i < rows; i++) {
        printf("ResultadoCPU[%d] = %f\n", i, h_resultCPU[i]);
    }

    printf("\nResultado da multiplicação (GPU):\n");
    for (int i = 0; i < rows; i++) {
        printf("ResultadoGPU[%d] = %f\n", i, h_resultGPU[i]);
    }

    // Comparação de desempenho
    printf("\nCPU Time: %f ms\n", cpuTime);
    printf("GPU Time: %f ms\n", gpuTime);
    printf("Speedup (CPU/GPU): %f\n", cpuTime / gpuTime);

    // Validação dos resultados
    int correct = 1;
    for (int i = 0; i < rows; i++) {
        if (abs(h_resultCPU[i] - h_resultGPU[i]) > 1e-5) {
            correct = 0;
            break;
        }
    }
    printf("\nResultado %s\n", correct ? "CORRETO" : "INCORRETO");

    // Liberação de memória
    free(h_matrix); free(h_vector); free(h_resultCPU); free(h_resultGPU);
    cudaFree(d_matrix); cudaFree(d_vector); cudaFree(d_result);

    return 0;
}
