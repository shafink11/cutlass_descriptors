// splitk_serial_gemm_fp64_reduction.cu
// -----------------------------------------------------------------------------
// CUTLASS Split-K **serial** GEMM benchmark
// A  : column-major  (M × K)
// B  : row-major     (K × N)
// C/D: column-major  (M × N)
// -----------------------------------------------------------------------------
//nvcc  -I$CUTLASS_HOME/include -I$CUTLASS_HOME/tools/util/include -arch=sm_75 -std=c++17 -O3 -lcublas bigchungus.cu -o splitk_serial
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

/****************************************************************
 * device helpers
 ****************************************************************/
__device__ inline void atomicMaxDouble(double *addr, double val) {
  unsigned long long *ull = reinterpret_cast<unsigned long long *>(addr);
  unsigned long long old  = *ull, assumed;
  unsigned long long newval = __double_as_longlong(val);
  do {
    assumed = old;
    if (__longlong_as_double(assumed) >= val) break;
    old = atomicCAS(ull, assumed, newval);
  } while (assumed != old);
}

/****************************************************************
 *  kernel to cast {half|bf16} -> double
 ****************************************************************/
template <typename In>
__global__ void cast_to_double(int64_t n, const In *src, double *dst) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += gridDim.x * blockDim.x) {
    dst[i] = static_cast<double>(src[i]);
  }
}

/****************************************************************
 *  reduction kernel for L2 / max-abs error
 ****************************************************************/
__global__ void compare_results(int64_t n,
                                const double *cutlass,
                                const double *cublas,
                                double *acc) {
  extern __shared__ double s[];
  double diff2 = 0.0, ref2 = 0.0, m = 0.0;

  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n; i += gridDim.x * blockDim.x) {
    double d = cutlass[i] - cublas[i];
    diff2 += d * d;
    ref2  += cublas[i] * cublas[i];
    m      = fmax(m, fabs(d));
  }

  s[threadIdx.x] = diff2;
  __syncthreads();
  for (int s2 = blockDim.x >> 1; s2; s2 >>= 1) {
    if (threadIdx.x < s2) s[threadIdx.x] += s[threadIdx.x + s2];
    __syncthreads();
  }

  if (!threadIdx.x) {
    atomicAdd(&acc[0], s[0]);   // Σ diff²
    atomicAdd(&acc[1], ref2);   // Σ ref²
    atomicMaxDouble(&acc[2], m);// max |diff|
  }
}

// -----------------------------------------------------------------------------
// 1.  Datatype selection -------------------------------------------------------
// -----------------------------------------------------------------------------
#ifdef USE_BF16
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using SmArchTag = cutlass::arch::Sm80;
#else
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using SmArchTag = cutlass::arch::Sm75;
#endif

using ElementOutput      = double;   // final C/D tensor
using ElementAccumulator = float;    // MMA accumulator
using ElementCompute     = double;   // epilogue compute (FP64)

using OpClassTag = cutlass::arch::OpClassTensorOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 32>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
static int const kStages = 2;
static int const kAlignmentA = cutlass::gemm::device::DefaultGemmConfiguration<
                                  OpClassTag, SmArchTag,
                                  ElementA, ElementB,
                                  ElementOutput, ElementAccumulator>::kAlignmentA;
static int const kAlignmentB = cutlass::gemm::device::DefaultGemmConfiguration<
                                  OpClassTag, SmArchTag,
                                  ElementA, ElementB,
                                  ElementOutput, ElementAccumulator>::kAlignmentB;
static bool const kSplitKSerial = true;

// CUTLASS GEMM ---------------------------------------------------------------
using Gemm = cutlass::gemm::device::Gemm<
    ElementA,                      cutlass::layout::ColumnMajor,  // A
    ElementB,                      cutlass::layout::RowMajor,     // B
    ElementOutput,                 cutlass::layout::ColumnMajor,  // C/D
    ElementAccumulator,
    OpClassTag,
    SmArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    kAlignmentA,
    kAlignmentB,
    kSplitKSerial>;

// -----------------------------------------------------------------------------
// 2.  Helper: fill tensor with random data ------------------------------------
// -----------------------------------------------------------------------------
template <typename Element>
void fill_random_device(Element *ptr, size_t num_elements) {
  std::vector<Element> host(num_elements);
  std::mt19937 gen(2025);
  std::uniform_real_distribution<float> dist(-99999999.f, 9999999999.f);
  for (size_t i = 0; i < num_elements; ++i) host[i] = Element(dist(gen));
  cudaMemcpy(ptr, host.data(), num_elements * sizeof(Element),
             cudaMemcpyHostToDevice);
}

// -----------------------------------------------------------------------------
// 3.  Verify with cuBLAS -------------------------------------------------------
// -----------------------------------------------------------------------------
void verify_vs_cublas(int M, int N, int K,
                      const void *A_src, const void *B_src,
                      const double *C_cutlass_d,
                      bool print_tile = true) {

  int64_t elemsA = int64_t(M) * K;
  int64_t elemsB = int64_t(K) * N;
  int64_t elemsC = int64_t(M) * N;

  // Convert A & B to FP64 for cuBLAS -----------------------------------------
  double *A_d64, *B_d64;
  cudaMalloc(&A_d64, elemsA * sizeof(double));
  cudaMalloc(&B_d64, elemsB * sizeof(double));

#ifdef USE_BF16
  cast_to_double<<<256,256>>>(elemsA,
      static_cast<const cutlass::bfloat16_t*>(A_src), A_d64);
  cast_to_double<<<256,256>>>(elemsB,
      static_cast<const cutlass::bfloat16_t*>(B_src), B_d64);
#else
  cast_to_double<<<256,256>>>(elemsA,
      static_cast<const cutlass::half_t*>(A_src), A_d64);
  cast_to_double<<<256,256>>>(elemsB,
      static_cast<const cutlass::half_t*>(B_src), B_d64);
#endif

  // cuBLAS DGEMM -------------------------------------------------------------
  double *C_ref_d;  cudaMalloc(&C_ref_d, elemsC * sizeof(double));
  cudaMemset(C_ref_d, 0, elemsC * sizeof(double));

  cublasHandle_t handle;  cublasCreate(&handle);
  const double alpha = 1.0, beta = 0.0;

  /*  A: column-major (M×K)  -> CUBLAS_OP_N, lda = M
   *  B: row-major (K×N)     -> interpret as (N×K) column-major, so pass
   *                            it with CUBLAS_OP_T and ldb = N
   *  C: column-major (M×N)  -> ldc = M
   */
  cublasDgemm(handle,
              CUBLAS_OP_N, CUBLAS_OP_T,
              M, N, K,
              &alpha,
              A_d64, M,
              B_d64, N,
              &beta,
              C_ref_d, M);

  cublasDestroy(handle);

  // Error metrics ------------------------------------------------------------
  double *acc_d;  cudaMalloc(&acc_d, 3 * sizeof(double));
  cudaMemset(acc_d, 0, 3 * sizeof(double));

  compare_results<<<256, 256, 256 * sizeof(double)>>>(
      elemsC, C_cutlass_d, C_ref_d, acc_d);

  double acc_h[3];
  cudaMemcpy(acc_h, acc_d, 3 * sizeof(double), cudaMemcpyDeviceToHost);

  std::cout << std::scientific << std::setprecision(3)
            << "\nL2  relative error = "
            << std::sqrt(acc_h[0]) / std::sqrt(acc_h[1])
            << "\nMax absolute  diff = " << acc_h[2] << "\n";

  // Optional: print 5×5 tile -------------------------------------------------
  if (print_tile) {
    constexpr int T = 5;
    double h_cutlass[T*T], h_cublas[T*T];

    cudaMemcpy2D(h_cutlass, T * sizeof(double),
                 C_cutlass_d, M * sizeof(double),
                 T * sizeof(double), T, cudaMemcpyDeviceToHost);

    cudaMemcpy2D(h_cublas,  T * sizeof(double),
                 C_ref_d,   M * sizeof(double),
                 T * sizeof(double), T, cudaMemcpyDeviceToHost);

    std::cout << "\nFirst 5×5 tile of C (CUTLASS | cuBLAS):\n";
    for (int i = 0; i < T; ++i) {
      for (int j = 0; j < T; ++j)
        std::cout << std::setw(11) << h_cutlass[i + j * T] << " ";
      std::cout << " | ";
      for (int j = 0; j < T; ++j)
        std::cout << std::setw(11) << h_cublas[i + j * T] << " ";
      std::cout << "\n";
    }
  }

  cudaFree(acc_d);  cudaFree(C_ref_d);
  cudaFree(A_d64);  cudaFree(B_d64);
}

// -----------------------------------------------------------------------------
// 4.  Main --------------------------------------------------------------------
// -----------------------------------------------------------------------------
int main() {
  // Problem size -------------------------------------------------------------
  int const M = 80000;
  int const N = 80000;
  int const K = 8192;
  int const split_k_slices = 32;   // serial Split-K

  ElementAccumulator alpha = 1.0f;
  ElementAccumulator beta  = 0.0f;

  size_t bytes_A = sizeof(ElementA)      * size_t(M) * size_t(K);
  size_t bytes_B = sizeof(ElementB)      * size_t(K) * size_t(N);
  size_t bytes_C = sizeof(ElementOutput) * size_t(M) * size_t(N);

  ElementA        *A;
  ElementB        *B;
  ElementOutput   *C;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Allocating ~" << (bytes_A + bytes_B + bytes_C) / double(1 << 20)
            << " MiB …" << std::flush;
  cudaMalloc(&A, bytes_A);
  cudaMalloc(&B, bytes_B);
  cudaMalloc(&C, bytes_C);
  std::cout << " done\n";

  // Init tensors -------------------------------------------------------------
  std::cout << "Filling tensors with random values …" << std::flush;
  fill_random_device(A, size_t(M) * size_t(K));
  fill_random_device(B, size_t(K) * size_t(N));
  cudaMemset(C, 0, bytes_C);
  std::cout << " done\n";

  // GEMM arguments -----------------------------------------------------------
  typename Gemm::Arguments args({M, N, K},           // GemmCoord
                                {A, M},              // A ptr, lda (col-major)
                                {B, N},              // B ptr, ldb (row-major)
                                {C, M},              // C ptr, ldc (col-major)
                                {C, M},              // D ptr, ldd
                                {alpha, beta},
                                split_k_slices);

  size_t workspace_bytes = Gemm::get_workspace_size(args);
  void *workspace = nullptr;
  cudaMalloc(&workspace, workspace_bytes);

  Gemm gemm_op;
  auto status = gemm_op.initialize(args, workspace);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "\nGEMM init failed: "
              << cutlassGetStatusString(status) << "\n";
    return -1;
  }

  // Warm-up ------------------------------------------------------------------
  gemm_op();
  cudaDeviceSynchronize();

  // Timed run ----------------------------------------------------------------
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  gemm_op();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms = 0.f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);

  double gflops = (2.0 * double(M) * double(N) * double(K))
                  / (elapsed_ms * 1.0e6);

  std::cout << "\n=== Split-K Serial GEMM ===\n";
#ifdef USE_BF16
  std::cout << "A/B = BF16, C = FP64, accum = FP32";
#else
  std::cout << "A/B = FP16, C = FP64, accum = FP32";
#endif
  std::cout << "\nProblem: " << M << "×" << N << "×" << K;
  std::cout << ", split-K = " << split_k_slices;
  std::cout << "\nElapsed: " << elapsed_ms << " ms";
  std::cout << "\nPerf   : " << gflops << " GFLOP/s\n";

  verify_vs_cublas(M, N, K, A, B, C);

  // Cleanup ------------------------------------------------------------------
  cudaFree(workspace);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  return 0;
}
