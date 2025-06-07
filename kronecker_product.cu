#include <stdio.h>
#include <cuda.h>
#include <cuComplex.h>

__global__ void kernelKronecker(cuComplex* A, int m, int n, cuComplex* B, int p, int q, cuComplex* C){
    // coordenadas das threads na matriz c
    int linha = threadIdx.x + blockIdx.x * blockDim.x;
    int coluna = threadIdx.y + blockIdx.y * blockDim.y;

    // dimensoes da matriz resultante
    int r_linhas = m * p;
    int r_colunas = n * q;

    if (linha < r_linhas && coluna < r_colunas){ // para n ultrapassar o limite
        // mapeia quais elementos das matrizes originais a e b estao sendo usadas pra calcular o prod kronecker
        int i_A = linha / p;
        int j_A = coluna / q;
        int k_B = linha  % p;
        int l_B = coluna % q;

        // obtem os elementos complexos
        cuComplex a = A[i_A * n + j_A];
        cuComplex b = B[k_B * q + l_B];

        // multiplicacao complexa
        C[linha*r_colunas+coluna] = cuCmulf(a, b);
    }
}

//funcao principal
void produtoKronecker(cuComplex* h_A, int m, int n, cuComplex* h_B, int p, int q, cuComplex* h_C){
    
    int r_linhas = m * p;
    int r_colunas = n * q;

    // alocacao na gpu
    cuComplex *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n * sizeof(cuComplex));
    cudaMalloc(&d_B, p * q * sizeof(cuComplex));
    cudaMalloc(&d_C, r_linhas * r_colunas * sizeof(cuComplex));

    // pega os dados do host p device
    cudaMemcpy(d_A, h_A, m * n * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, p * q * sizeof(cuComplex), cudaMemcpyHostToDevice);

    // config da execucao
    dim3 threadsPerBlock(16, 16); // def o tamanho dos blocos de threads a serem exec em paralelo
    dim3 blocksPerGrid((r_colunas+threadsPerBlock.x-1)/threadsPerBlock.x,
                       (r_linhas+threadsPerBlock.y-1)/threadsPerBlock.y);
    
    //executa o kernel
    kernelKronecker<<<blocksPerGrid, threadsPerBlock>>>(d_A, m, n, d_B, p, q, d_C);

    // copia o resultado
    cudaMemcpy(h_C, d_C, r_linhas * r_colunas * sizeof(cuComplex), cudaMemcpyDeviceToHost);

    //libera a memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// main p teste (gerada por ia)
/*
int main() {
    int m = 2, n = 2, p = 2, q = 2;
    
    // Cria matrizes complexas
    cuComplex h_A[] = {
        make_cuComplex(1, 2), make_cuComplex(3, 4),
        make_cuComplex(5, 6), make_cuComplex(7, 8)
    };
    cuComplex h_B[] = {
        make_cuComplex(9, 10), make_cuComplex(11, 12),
        make_cuComplex(13, 14), make_cuComplex(15, 16)
    };
    
    int result_size = (m * p) * (n * q);
    cuComplex* h_C = (cuComplex*)malloc(result_size * sizeof(cuComplex));
    
    // Calcula
    produtoKronecker(h_A, m, n, h_B, p, q, h_C);
    
    // Imprime
    printf("Resultado:\n");
    for (int i = 0; i < m*p; i++) {
        for (int j = 0; j < n*q; j++) {
            cuComplex val = h_C[i * (n*q) + j];
            printf("(%.1f + %.1fi) ", val.x, val.y);
        }
        printf("\n");
    }
    
    free(h_C);
    return 0;
}
    */
