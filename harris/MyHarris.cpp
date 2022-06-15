#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace cv;

void Harris(Mat &src, Mat &output, int kernel_size, float k, int threshold) {
    // 1. Calcola le componenti del vettore gradiente
    Mat dx, dy;
    Sobel(src, dx, CV_32FC1, 1, 0, kernel_size, BORDER_DEFAULT);
    Sobel(src, dy, CV_32FC1, 0, 1, kernel_size, BORDER_DEFAULT);

    // 2. Calcolare le componenti della matrice E
    // Dx^2, Dy^2 e Dx*Dy
    Mat dx2, dy2, dxdy;
    pow(dx, 2.0, dx2);
    pow(dy, 2.0, dy2);
    multiply(dx, dy, dxdy);

    // 3. Applicare un filtro Gaussiano alle tre componenti
    Mat dx2g, dy2g, dxdyg;
    GaussianBlur(dx2, dx2g, Size(7, 7), 2.0, 0.0, BORDER_DEFAULT);
    GaussianBlur(dy2, dy2g, Size(7, 7), 0.0, 2.0, BORDER_DEFAULT);
    GaussianBlur(dxdy, dxdyg, Size(7, 7), 2.0, 2.0, BORDER_DEFAULT);

    // 4. Calcolare l'indice R
    Mat det, trace, R;
    // Calcoliamo il determinante
    Mat diag1, diag2;
    multiply(dx2g, dy2g, diag1);
    multiply(dxdyg, dxdyg, diag2);
    det = diag1 - diag2;
    // Calcoliamo la traccia
    pow(dx2g + dy2g, 2, trace);
    R = det - k * trace;
    imshow("R", R);
    waitKey(0);

    // 5. Normalizziamo l'indice R tra [0, 255]
    normalize(R, R, 0, 255, NORM_MINMAX, CV_8U);
    imshow("R norm", R);
    waitKey(0);

    // 6. Sogliamo R
    output = src.clone();
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (R.at<uchar>(i, j) > threshold) {
                circle(output, Point(j, i), 6, Scalar(0), 2, 8, 0);
            }
        }
    }
}

int main(int argc, char **argv) {
    int kernelSize, threshold;
    float k;
    // Controllo argomenti riga di comando
    if (argc != 5) {
        cout << "Usage: " << argv[0] << " image_name kernelSize k threshold" << endl;
        return -1;
    }

    // Lettura dell'immagine
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "Could not read the image with name " << argv[1] << endl;
        return -1;
    }

    kernelSize = stoi(argv[2]);
    k = stof(argv[3]);
    threshold = stoi(argv[4]);
    
    Mat out;
    Harris(src, out, kernelSize, k, threshold);
    
    imshow("Harris", out);
    waitKey(0);
    return 0;
}