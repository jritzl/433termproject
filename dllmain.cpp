#include "pch.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <cmath>
#include <cstdlib>
#include <ctime>
#define PI 3.14159265
#define FRAME_LENGTH 2048

// Function prototypes
extern "C" __declspec(dllexport) int AddNumbers(int a, int b);
extern "C" __declspec(dllexport) void polyfit(double* x, double* y, int size, int degree, double* coefficients);
extern "C" __declspec(dllexport) void auto_correlation(double* signal, double* result, int N);
extern "C" __declspec(dllexport) void calculate_psd(double* signal, int N, double* PSD, double* freqs, int fs);
extern "C" __declspec(dllexport) double generateGaussian();
extern "C" __declspec(dllexport) void generateGaussianArray(double* array, int n);
extern "C" __declspec(dllexport) double random_double();
extern "C" __declspec(dllexport) void generate_pink_noise(int n, double* pink_noise);
extern "C" __declspec(dllexport) int isWhiteNoise(double* noiseSignal, int length, int fs);
extern "C" __declspec(dllexport) double calculateStdDeviation(double* noiseSignal, int length);
extern "C" __declspec(dllexport) int isSpeech(double* frame, int frameLength, double threshold);
extern "C" __declspec(dllexport) void checkSegmentsForWhiteNoise(double* signal, int totalLength, int fs, double* results);
extern "C" __declspec(dllexport) double EnergyisSpeech(double* frame, int frameLength, double threshold);
extern "C" __declspec(dllexport) void wienerFilter(double* signal, int length, double std_dev, double* filtered_signal, int P);
extern "C" __declspec(dllexport) void processSignal(double* noisySignal, int signalLength, double* filteredSignal, int P, double* speecher, double* noisetype, double* noisestd);
static void generateIncreasingWaveform(int N, double dc, double* waveform);
static void fft(double* realIn, double* imagIn, double* realOut, double* imagOut, int n);
static void buildToeplitz(double* arr, int N, double** matrix);
static void convolve(double* x, int N, double* h, int M, double* y);
static void matrixVectorMultiply(double** matrix, double* vector, double* result, int P);

static void luDecomposition(double** A, double** L, double** U, int n);
static void forwardSubstitution(double** L, double* y, double* b, int n);
static void backwardSubstitution(double** U, double* x, double* y, int n);
static void invertMatrix(double** A, double** invA, int n);

static double** allocateMatrix(int n);
static void freeMatrix(double** matrix, int n);
static double determinantFromLU(double** L, double** U, int n);
static void printArray(double* array, int size);

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
        // Code here runs when the DLL is loaded
        break;
    case DLL_THREAD_ATTACH:
        // Code here runs when a new thread is created
        break;
    case DLL_THREAD_DETACH:
        // Code here runs when a thread exits cleanly
        break;
    case DLL_PROCESS_DETACH:
        // Code here runs when the DLL is unloaded
        break;
    }
    return TRUE;
}

// Implementation of the AddNumbers function
extern "C" __declspec(dllexport) int AddNumbers(int a, int b) {
    return a + b;
}

extern "C" __declspec(dllexport) void polyfit(double* x, double* y, int size, int degree, double* coefficients) {
    // Allocate memory for the coefficient matrix and the constants vector
    double** A = (double**)malloc((degree + 1) * sizeof(double*));
    double* b = (double*)malloc((degree + 1) * sizeof(double));

    // Fill the coefficient matrix
    for (int i = 0; i <= degree; i++) {
        A[i] = (double*)malloc((degree + 1) * sizeof(double));
        for (int j = 0; j <= degree; j++) {
            A[i][j] = 0;
            for (int k = 0; k < size; k++) {
                A[i][j] += pow(x[k], i + j);
            }
        }
    }

    // Fill the constants vector
    for (int i = 0; i <= degree; i++) {
        b[i] = 0;
        for (int k = 0; k < size; k++) {
            b[i] += pow(x[k], i) * y[k];
        }
    }

    // Solve the system of equations using Gaussian elimination
    for (int i = 0; i <= degree; i++) {
        double max = fabs(A[i][i]);
        int maxIndex = i;
        for (int k = i + 1; k <= degree; k++) {
            if (fabs(A[k][i]) > max) {
                max = fabs(A[k][i]);
                maxIndex = k;
            }
        }

        // Swap rows
        for (int k = i; k <= degree; k++) {
            double temp = A[maxIndex][k];
            A[maxIndex][k] = A[i][k];
            A[i][k] = temp;
        }
        double temp = b[maxIndex];
        b[maxIndex] = b[i];
        b[i] = temp;

        // Perform row operations to make the matrix upper triangular
        for (int k = i + 1; k <= degree; k++) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j <= degree; j++) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back substitution to get the coefficients
    for (int i = degree; i >= 0; i--) {
        coefficients[i] = b[i];
        for (int j = i + 1; j <= degree; j++) {
            coefficients[i] -= A[i][j] * coefficients[j];
        }
        coefficients[i] /= A[i][i];
    }

    // Free dynamically allocated memory
    for (int i = 0; i <= degree; i++) {
        free(A[i]);
    }
    free(A);
    free(b);
}

// Function to calculate the auto-correlation of an input signal
extern "C" __declspec(dllexport) void auto_correlation(double* signal, double* result, int N) {
    for (int lag = 0; lag < N; lag++) {
        result[lag] = 0;
        for (int i = 0; i < N - lag; i++) {
            result[lag] += signal[i] * signal[i + lag];
        }
        result[lag] /= (N);
    }
}

// Function to calculate the Power Spectral Density (PSD) of an input signal
extern "C" __declspec(dllexport) void calculate_psd(double* signal, int N, double* PSD, double* freqs, int fs) {
    double* autocorr = (double*)malloc(sizeof(double) * N);
    double* real = (double*)malloc(sizeof(double) * N);
    double* imag = (double*)calloc(N, sizeof(double));  // Initialize to zero

    // Calculate auto-correlation using the provided function
    auto_correlation(signal, autocorr, N);

    // Compute FFT of the auto-correlation
    fft(autocorr, imag, real, imag, N);

    // Calculate PSD
    for (int i = 0; i < N / 2 + 1; i++) {
        PSD[i] = (real[i] * real[i] + imag[i] * imag[i]) / N;
        freqs[i] = (double)(i * fs / N);
    }

    // Clean up
    free(autocorr);
    free(real);
    free(imag);
}

// Simple FFT implementation assuming input is real
static void fft(double* realIn, double* imagIn, double* realOut, double* imagOut, int n) {
    if (n <= 1) return;

    // Divide the array and conquer
    double* evenReal = (double*)malloc(sizeof(double) * (n / 2));
    double* oddReal = (double*)malloc(sizeof(double) * (n / 2));
    double* evenImag = (double*)malloc(sizeof(double) * (n / 2));
    double* oddImag = (double*)malloc(sizeof(double) * (n / 2));

    for (int i = 0; i < n / 2; i++) {
        evenReal[i] = realIn[2 * i];
        oddReal[i] = realIn[2 * i + 1];
        evenImag[i] = imagIn[2 * i];
        oddImag[i] = imagIn[2 * i + 1];
    }

    fft(evenReal, evenImag, evenReal, evenImag, n / 2);
    fft(oddReal, oddImag, oddReal, oddImag, n / 2);

    for (int k = 0; k < n / 2; k++) {
        double cosTerm = cos(-2 * PI * k / n);
        double sinTerm = sin(-2 * PI * k / n);
        realOut[k] = evenReal[k] + cosTerm * oddReal[k] - sinTerm * oddImag[k];
        imagOut[k] = evenImag[k] + cosTerm * oddImag[k] + sinTerm * oddReal[k];
        realOut[k + n / 2] = evenReal[k] - (cosTerm * oddReal[k] - sinTerm * oddImag[k]);
        imagOut[k + n / 2] = evenImag[k] - (cosTerm * oddImag[k] + sinTerm * oddReal[k]);
    }

    free(evenReal);
    free(oddReal);
    free(evenImag);
    free(oddImag);
}

extern "C" __declspec(dllexport) double generateGaussian() {
    static int hasSpare = 0;
    static double spare;

    if (hasSpare) {
        hasSpare = 0;
        return spare;
    }

    hasSpare = 1;

    double u, v, s;
    do {
        u = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
        v = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return u * s;
}

// Function to generate an array of Gaussian distributed numbers
extern "C" __declspec(dllexport) void generateGaussianArray(double* array, int n) {
    for (int i = 0; i < n; ++i) {
        array[i] = generateGaussian();
    }
}

// Function to generate random numbers between -1 and 1
extern "C" __declspec(dllexport) double random_double() {
    return (double)rand() / RAND_MAX * 2 - 1;
}

// Function to generate pink noise of length n
extern "C" __declspec(dllexport) void generate_pink_noise(int n, double* pink_noise) {
    // Seed the random number generator
    srand((unsigned int)time(NULL));

    // Initialize variables
    double b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0;
    double white;

    // Generate pink noise samples
    for (int i = 0; i < n; i++) {
        white = random_double();

        // Apply 1/f^3 filter (6dB per octave)
        b0 = 0.99886 * b0 + white * 0.0555179;
        b1 = 0.99332 * b1 + white * 0.0750759;
        b2 = 0.96900 * b2 + white * 0.1538520;
        b3 = 0.86650 * b3 + white * 0.3104856;
        b4 = 0.55000 * b4 + white * 0.5329522;
        b5 = -0.7616 * b5 - white * 0.0168980;

        // Combine all the filtered random numbers
        pink_noise[i] = b0 + b1 + b2 + b3 + b4 + b5 + white * 0.5362;
    }
}

extern "C" __declspec(dllexport) int isWhiteNoise(double* noiseSignal, int length, int fs) {
    double* psd = (double*)malloc((length / 2 + 1) * sizeof(double));
    double* freqs = (double*)malloc((length / 2 + 1) * sizeof(double));

    double* dbpsd = (double*)malloc((length / 2) * sizeof(double));
    double* logfreqs = (double*)malloc((length / 2) * sizeof(double));

    // Calculate PSD (assuming you have a calculate_psd function)
    calculate_psd(noiseSignal, length, psd, freqs, fs);

    // Convert PSD to dB scale
    for (int i = 1; i < length / 2 + 1; i++) {
        dbpsd[i - 1] = 10 * log10(psd[i]);
        logfreqs[i - 1] = log10(freqs[i]);
    }

    // Perform linear regression on log-log PSD data
    double coeffs[2]; // Slope (coeffs[1]) and intercept
    polyfit(logfreqs, dbpsd, length / 2, 1, coeffs);

    free(psd);
    free(freqs);
    free(dbpsd);
    free(logfreqs);

    // Check if the slope is within the white noise range
    double slopeThreshold = -6; // dB/octave (adjust this as needed)
    return (coeffs[1] > slopeThreshold); // Returns 1 if white, 0 if not
}

extern "C" __declspec(dllexport) void checkSegmentsForWhiteNoise(double* signal, int totalLength, int fs, double* results) {
    int segmentLength = totalLength / 10;

    for (int i = 0; i < 10; i++) {
        double* segment = (double*)malloc(segmentLength * sizeof(double));
        for (int j = 0; j < segmentLength; j++) {
            segment[j] = signal[i * segmentLength + j];
        }

        results[i] = (double)isWhiteNoise(segment, segmentLength, fs);
        free(segment);
    }
}

// Function to calculate standard deviation of the noise
extern "C" __declspec(dllexport) double calculateStdDeviation(double* noiseSignal, int length) {
    double sum = 0.0, variance = 0.0;
    for (int i = 0; i < length; i++) {
        sum += noiseSignal[i];
    }
    double mean = sum / length;

    for (int i = 0; i < length; i++) {
        variance += pow(noiseSignal[i] - mean, 2);
    }
    variance /= length;

    return sqrt(variance);
}

extern "C" __declspec(dllexport) int isSpeech(double* frame, int frameLength, double threshold) {
    double energy = 0.0;

    // Calculate short-term energy
    for (int i = 0; i < frameLength; i++) {
        energy += frame[i] * frame[i];
    }

    energy /= frameLength;
   
    // Compare energy to threshold
    return (energy > threshold); // Returns 1 if speech, 0 otherwise
}


extern "C" __declspec(dllexport) double EnergyisSpeech(double* frame, int frameLength, double threshold) {
    double energy = 0.0;

    // Calculate short-term energy
    for (int i = 0; i < frameLength; i++) {
        energy += frame[i] * frame[i];
    }

    energy /= frameLength;
    // Compare energy to threshold
    return energy; // Returns the energy value
}

static void buildToeplitz(double* arr, int N, double** matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i > j) {
                matrix[i][j] = arr[i - j];
            }
            else if (i < j) {
                matrix[i][j] = arr[j - i];
            }
            else {
                matrix[i][j] = arr[0];
            }
        }
    }
}

static void convolve(double* x, int N, double* h, int M, double* y) {
    int resultSize = N+M-1;

    for (int n = 0; n < resultSize; n++) {
        y[n] = 0.0;
        for (int k = 0; k <= n; k++) {
            if (k < N && n - k < M) {
                y[n] += x[k] * h[n - k];
            }
        }
    }
}

static void matrixVectorMultiply(double** matrix, double* vector, double* result, int P) {
    for (int i = 0; i < P; i++) {
        result[i] = 0.0;
        for (int j = 0; j < P; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

// Function to perform LU decomposition
// Function to perform LU decomposition
static void luDecomposition(double** A, double** L, double** U, int n) {
    // Initialize L and U matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            U[i][j] = 0;
            L[i][j] = (i == j) ? 1 : 0; // Set diagonal of L to 1, others to 0
        }
    }

    // Perform the LU decomposition
    for (int i = 0; i < n; i++) {
        // Compute the upper triangular matrix U
        for (int k = i; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++) {
                sum += (L[i][j] * U[j][k]);
            }
            U[i][k] = A[i][k] - sum;
        }

        // Compute the lower triangular matrix L
        for (int k = i + 1; k < n; k++) {
            double sum = 0;
            for (int j = 0; j < i; j++) {
                sum += (L[k][j] * U[j][i]);
            }
            if (U[i][i] == 0) {
                return; // Handle division by zero or add error reporting
            }
            L[k][i] = (A[k][i] - sum) / U[i][i];
        }
    }
}

// Function to perform forward substitution
static void forwardSubstitution(double** L, double* y, double* b, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }
}

// Function to perform backward substitution
static void backwardSubstitution(double** U, double* x, double* y, int n) {
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = i + 1; j < n; j++) {
            sum += U[i][j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i][i];
    }
}

// Function to invert a matrix using LU decomposition
static void invertMatrix(double** A, double** invA, int n) {
    double** L = (double**)malloc(n * sizeof(double*));
    double** U = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        L[i] = (double*)malloc(n * sizeof(double));
        U[i] = (double*)malloc(n * sizeof(double));
    }

    luDecomposition(A, L, U, n);

    double* y = (double*)malloc(n * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    double* b = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b[j] = (i == j) ? 1 : 0;
        }

        forwardSubstitution(L, y, b, n);
        backwardSubstitution(U, x, y, n);

        for (int j = 0; j < n; j++) {
            invA[j][i] = x[j];
        }
    }

    for (int i = 0; i < n; i++) {
        free(L[i]);
        free(U[i]);
    }
    free(L);
    free(U);
    free(y);
    free(x);
    free(b);
}

static double** allocateMatrix(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

static void freeMatrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

static double determinantFromLU(double** L, double** U, int n) {
    double det = 1.0;
    for (int i = 0; i < n; i++) {
        det *= U[i][i];
    }
    return det;
}

static void printArray(double* array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%.2f\n", array[i]);
    }
}

extern "C" __declspec(dllexport) void wienerFilter(double* signal, int length, double std_dev, double* filtered_signal,int P) {
    
    double* Rv = (double*)calloc(P, sizeof(double));
    double* Rx = (double*)calloc(P, sizeof(double));
    double* Rs = (double*)calloc(P, sizeof(double));

    Rv[0] = std_dev * std_dev;

    double** toeplitz = allocateMatrix(P);
    double** Rx_matrix_reg = allocateMatrix(P);

    auto_correlation(signal, Rx, P);
    buildToeplitz(Rx, P, toeplitz);

    double lambda = 5 * std_dev;
    double** identityMatrixP = allocateMatrix(P);
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < P; j++) {
            identityMatrixP[i][j] = (i == j) ? 1.0 : 0.0;

        }
    }

    // Calculate Rs vector (Rx - Rv)
    for (int i = 0; i < P; i++) {
        Rs[i] = Rx[i] - Rv[i];
    }

    // Calculate regularized Rx matrix
    // Rx_matrix_reg = Toeplitz + lambda * identityMatrixP
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < P; j++) {
            Rx_matrix_reg[i][j] = toeplitz[i][j] + lambda * identityMatrixP[i][j];
        }
    }

   /* for (int i = 0; i < P; i++) {
        for (int j = 0; j < P; j++) {
            identityMatrixP[i][j] = (i == j) ? 1.0 : 0.0;

        }
    }

    for (int i = 0; i < P; i++) {
        Rs[i] = Rx[i] - Rv[i];
        for (int j = 0; j < P; j++) {
            Rx_matrix_reg[i][j] = toeplitz[i][j] + lambda * identityMatrixP[i][j];
        }
    }*/





    double** Rx_matrix_reg_inv = allocateMatrix(P);
    invertMatrix(Rx_matrix_reg, Rx_matrix_reg_inv, P);

    double* h = (double*)malloc(sizeof(double) * P);
    matrixVectorMultiply(Rx_matrix_reg_inv, Rs, h, P);

    double* y = (double*)calloc(length+P-1, sizeof(double));
    convolve(signal, length, h, P, y);

    for (int i = 0; i < length+P-1; i++) {
        filtered_signal[i] = y[i];
    }

    free(Rv);
    free(Rx);
    free(Rs);
    freeMatrix(toeplitz, P);
    freeMatrix(Rx_matrix_reg, P);
    freeMatrix(Rx_matrix_reg_inv, P);
    freeMatrix(identityMatrixP, P);
    free(h);
    free(y);
}


extern "C" __declspec(dllexport) void processSignal(double* noisySignal, int signalLength, double* filteredSignal, int P, double* speecher, double* noisetype, double* noisestd) {
    double threshold = 0.03;
    double stdDev = 0.05;
    int fs = 44100;
    double* waveform = (double*)malloc(FRAME_LENGTH * sizeof(double));
    double energy = 0;
    double avg_energy = 0;
    for (int i = 0; i < signalLength; i += FRAME_LENGTH) {
        double* filtered_frame = (double*)malloc((FRAME_LENGTH+P-1) * sizeof(double));
        double* frame = (double*)malloc(FRAME_LENGTH * sizeof(double));

        // Copy signal into frame, handle end of signal
        for (int j = 0; j < FRAME_LENGTH; j++) {
            frame[j] = noisySignal[i+j];
        }

        // Common filter operation
        wienerFilter(frame, FRAME_LENGTH, stdDev, filtered_frame,P);

        // Process the filtered frame
        
        int is_speech = isSpeech(frame, FRAME_LENGTH, threshold);
        energy = EnergyisSpeech(frame, FRAME_LENGTH, threshold);
        speecher[i/2048] = energy;
        avg_energy += energy;

       
        threshold = 0.03 * max(1.0, energy / 0.03);

        


        for (int j = 0; j < (FRAME_LENGTH+P-1) && (i + j < signalLength); j++) {
            filteredSignal[i + j] += filtered_frame[j];
            noisetype[i / 2048] = 2;
        }
        if (!is_speech) {
            // Analyze the type of noise
            int iswhite = isWhiteNoise(frame, FRAME_LENGTH, fs);
            noisetype[i / 2048] = iswhite;
            stdDev = calculateStdDeviation(frame, FRAME_LENGTH);
            
            
        }
        noisestd[i / 2048] = stdDev;



        

        

    }
}
