#include <stdio.h>
#include <cuda.h>

__global__ void somaVetores(int *A, int *B, int *C, int n){ //entrada: vetores a e b, saida: c, tamanho dos vetores: n
    int i = threadIdx.x + blockIdx.x * blockDim.x; // calculando cada indice global unico de cada thread

    if (i < n) // verifica se o indice está dentro do tamanho do vetor
        C[i] = A[i] + B[i];
}

int main(){
    int n = 1000; // tamanho do vetor
    int threadsPerBlock = 256; //tamanho basico
    int blocksPerGrid = (n + threadsPerBlock - 1)/threadsPerBlock;

    int *h_A, *h_B, *h_C; // ponteiros p cpu (host)
    int *d_A, *d_B, *d_C; // ponteiros p gpu (device)

    // alocar os espaços do ponteiros (cpu)
    h_A = (int*)malloc(n * sizeof(int));
    h_B = (int*)malloc(n * sizeof(int));
    h_C = (int*)malloc(n* sizeof(int));

    // alocar os espaços dos ponteiros (gpu)
    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_B, n *sizeof(int));
    cudaMalloc(&d_C, n * sizeof(int));

    // copiar os dados da cpu na gpu p executar a operação na gpu
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);

    // executar o kernel
    somaVetores<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // devolver o resultado para o host
    cudaMemcpy(h_C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    //verificar resultados (5 primeiros elems)
    for (int i = 0; i < 5; i++) {
        printf("C[%d] = %d (esperado: %d)\n", i, h_C[i], h_A[i] + h_B[i]);
    }
    
    //liberar memoria
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
