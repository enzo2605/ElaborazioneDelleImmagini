#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

const int kernel_size = 3;

void thresholding(Mat &img, Mat &out, int lowThreshold, int highThreshold) {
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            if (img.at<uchar>(i, j) > highThreshold) {
                out.at<uchar>(i, j) = 255;
                // Prumuoviamo tutti i pixel nel suo intorno 3x3 a pixel forti
                for (int u = -1; u <= 1; u++) {
                    for (int v = -1; v <= 1; v++) {
                        if (img.at<uchar>(i + u, j + v) > lowThreshold && img.at<uchar>(i + u, j + v) < highThreshold) {
                            out.at<uchar>(i + u, j + v) = 255;
                        }
                    }
                }
            }
            else if (img.at<uchar>(i, j) < lowThreshold) {
                out.at<uchar>(i, j) = 0;
            }
        }
    }
}

void noMaximaSuppression(Mat &magnitude, Mat &orientations, Mat &nms) {
    for (int i = 1; i < magnitude.rows - 1; i++) {
        for (int j = 1; j < magnitude.cols - 1; j++) {
            // Ricaviamo l'angolo del pixel in posizione (i, j)
            float angle = orientations.at<float>(i, j);
            // Facciamo in modo che gli angoli varino tra -180 e 180
            angle = (angle > 180) ? angle - 360 : angle;

            // orizzontale
            if ((angle > -22.5) && (angle <= 22.5) || (angle > -157.5) && (angle <= 157.5)) {
                if (magnitude.at<uchar>(i, j) >= magnitude.at<uchar>(i, j - 1) && magnitude.at<uchar>(i, j) >= magnitude.at<uchar>(i, j + 1)) {
                    nms.at<uchar>(i, j) = magnitude.at<uchar>(i, j);
                }
            }
            // diagonale dx
            else if ((angle > -67.5) && (angle <= -22.5) || (angle > 112.5) && (angle <= 157.5)) {
                if (magnitude.at<uchar>(i, j) >= magnitude.at<uchar>(i - 1, j - 1) && magnitude.at<uchar>(i, j) >= magnitude.at<uchar>(i + 1, j + 1)) {
                    nms.at<uchar>(i, j) = magnitude.at<uchar>(i, j);
                }                      
            }
            // verticale
            else if ((angle > -112.5) && (angle <= -67.5) || (angle > 67.5) && (angle <= 112.5)) {
                if (magnitude.at<uchar>(i, j) >= magnitude.at<uchar>(i - 1, j) && magnitude.at<uchar>(i, j) >= magnitude.at<uchar>(i + 1, j)) {
                    nms.at<uchar>(i, j) = magnitude.at<uchar>(i, j);
                }
            }   
            // diagonale sx
            else if ((angle > -157.5) && (angle <= -112.5) || (angle > 22.5) && (angle <= 67.5)) {
                if (magnitude.at<uchar>(i, j) >= magnitude.at<uchar>(i + 1, j - 1) && magnitude.at<uchar>(i, j) >= magnitude.at<uchar>(i - 1, j + 1)) {
                    nms.at<uchar>(i, j) = magnitude.at<uchar>(i, j);
                }        
            }
        }
    }
}

void Canny(Mat &src, Mat &output, int kernelSize, int lowThreshold, int highThreshold) {
    Mat gauss;
    /* 1. Convolvere l'immagine con il filtro Gaussiano */
    GaussianBlur(src, gauss, Size(5, 5), 0, 0);

    /* 2. Calcolare magnitudo e angolo di fase del vettore gradiente */
    // Calcolo del vettore gradiente
    Mat Dx, Dy;
    Sobel(gauss, Dx, CV_32FC1, 1, 0, kernel_size);
    Sobel(gauss, Dy, CV_32FC1, 0, 1, kernel_size);
    // Calcolo della magnitudo con formula standard
    Mat Dx2, Dy2, magnitude;
    pow(Dx, 2, Dx2);
    pow(Dy, 2, Dy2);
    sqrt(Dx2 + Dy2, magnitude);
    // Normalizzazione della magnitudo
    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8U);
    //imshow("magnitudo", magnitude);
    // Calcolo dell'angolo di fase con la funzione phase
    Mat orientations;
    phase(Dx, Dy, orientations, true);
    //imshow("angolo di fase", orientations);
    //waitKey(0);
    /* 3. Applicare la non maxima suppression */
    Mat nms = Mat::zeros(magnitude.rows, magnitude.cols, CV_8U);
    noMaximaSuppression(magnitude, orientations, nms);
    //imshow("no maxima suppression", nms);

    /* 4. Applicare il tresholding con isteresi */
    Mat out = Mat::zeros(nms.rows, nms.cols, CV_8U);
    thresholding(nms, out, lowThreshold, highThreshold);
    //imshow("threshold", out);
    //waitKey(0);
    output = out;
}

int main(int argc, char **argv) {
    Mat img, output;
    String img_name;
    int lowThreshold, highThreshold;

    // Controllo valori passati da riga di comando
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " img_name lowThreshold highThreshold" << endl;
        return -1;
    }
    img_name = argv[1];

    // Lettura immagine come parametro da riga di comando
    img = imread(img_name, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Could not open " << img_name << endl;
        return -1;
    }

    lowThreshold = stoi(argv[2]);
    highThreshold = stoi(argv[3]);

    Canny(img, output, kernel_size, lowThreshold, highThreshold);

    imshow("Canny", output);
    waitKey(0);
    return 0;
}