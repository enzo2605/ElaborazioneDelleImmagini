#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace cv;

void houghCircles(Mat src, Mat &out, Mat edgeCanny, int r_min, int r_max, int threshold) {
    /* 2. Creiamo lo spazio dei voti
        Lo spazio dei voti sarà matrice tridimensionale dove le prime
        due dimensioni sono dettate dalla dimensione della matrice
        di Canny. La terza è il range di valori che variano tra 
        il raggio minimo e il raggio massimo.
    */
    int sizes[] = {edgeCanny.rows, edgeCanny.cols, r_max - r_min + 1};
    // Allochiamo una matrice tridimensionale, le cui dimensioni sono in sizes
    // di profondità 8 bit e inizializzata a 0.
    Mat votes = Mat(3, sizes, CV_8U, Scalar(0));

    /* 3. Per ogni punto di edge (x, y) */
    for (int x = 0; x < edgeCanny.rows; x++) {
        for (int y = 0; y < edgeCanny.cols; y++) {
            if (edgeCanny.at<uchar>(x, y) == 255) {
                /* 4. Per ogni raggio che varia da r_min ad r_max */
                for (int radius = r_min; radius <= r_max; radius++) {
                    /* 5. Per ogni angolo theta che varia da 0 a 360 */
                    for (int theta = 0; theta < 360; theta++) {
                        /* 6. Calcola a e b */
                        int a = y - radius * cos(theta * M_PI / 180);
                        int b = x - radius * sin(theta * M_PI / 180);

                        // Se le coordinate del centro sono interne all'immagine
                        if (a >= 0 && a < edgeCanny.cols && b >= 0 && b < edgeCanny.rows) {
                            /* 7. Effettua la votazione */
                            votes.at<uchar>(b, a, radius - r_min)++;
                        }
                    }
                }
            }
        }
    }
    out = src.clone();
    /* 7. Andiamo a prendere i valori (a, b, r) maggiori di una certa soglia */
    for (int r = r_min; r < r_max; r++) {
        for (int b = 0; b < edgeCanny.rows; b++) {
            for (int a = 0; a < edgeCanny.cols; a++) {
                if (votes.at<uchar>(b, a, r - r_min) > threshold) {
                    // La prima chiamata disegna il centro del cerchio, di raggio 3 px
                    circle(out, Point(a, b), 3, Scalar(0), 2, 8, 0);
                    // La seconda chiamata disegna il cerchio effettivo
                    circle(out, Point(a, b), r, Scalar(0), 2, 8, 0);
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    // Controllo argomenti riga di comando
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " image_name" << endl;
        return -1;
    }

    // Lettura dell'immagine
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "Could not read the image with name " << argv[1] << endl;
        return -1;
    }

    // Blurring dell'immagine per attenuare l'eventuale rumore
    GaussianBlur(src, src, Size(5, 5), 0, 0);

    /* 1. Applichiamo l'algoritmo di Canny per ottenere l'immagine degli edge */
    Mat edgeCanny;
    Canny(src, edgeCanny, 90, 160, 3);
    imshow("Canny", edgeCanny);
    waitKey(0);

    int r_min = 40;
    int r_max = 90;
    int threshold = 140;
    
    Mat out;
    houghCircles(src, out, edgeCanny, r_min, r_max, threshold);
    
    imshow("output", out);
    waitKey(0);
    
    return 0;
}