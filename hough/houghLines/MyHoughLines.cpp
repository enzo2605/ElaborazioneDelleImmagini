#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace cv;

void houghLines(Mat src, Mat &out, Mat edgeCanny, int threshold) {
    /* 2. Creiamo lo spazio dei voti
        Lo spazio dei voti sarà matrice con tutti zeri e dovrà
        avere una dimensione tale da poter considerare tutte le
        possibili rette che passano per il piano immagine.
    */
    // Calcolo della distanza massima tra due punti nell'immagine
    int dist = hypot(src.rows, src.cols);
    // Inizializzazione della matrice dei voti
    Mat votes = Mat::zeros(dist * 2, 180, CV_8U);

    double rho, theta;
    /* 3. Per ogni punto (x,y) di edge */
    for (int x = 0; x < edgeCanny.rows; x++) {
        for (int y = 0; y < edgeCanny.cols; y++) {
            // Se il punto (x,y) è un punto di edge
            if (edgeCanny.at<uchar>(x, y) == 255) {
                /* 4. Per ogni angolo theta che varia tra 0 e 180 */
                for (theta = 0; theta < 180; theta++) {
                    /* 5. Calcola rho */
                    // (theta - 90) poiché l'intervallo theta varia da -90 a 90
                    // Le funzioni cos() e sin() vogliono l'argomento in radianti perciò 
                    // bisogna moltiplicare per pi greco / 180
                    rho = dist + y * cos((theta - 90) * CV_PI / 180) + x * sin((theta - 90) * CV_PI / 180);
                    /* 6. Effettua la votazione */
                    votes.at<uchar>(rho, theta)++;
                }
            }
        }
    }
    out = src.clone();
    /* 7. Andiamo a prendere i valori (rho, theta) maggiori di una certa soglia */
    for (int r = 0; r < votes.rows; r++) {
        for (int t = 0; t < votes.cols; t++) {
            // Se la retta caratterizzata dai parametri (rho, theta) è stata votata 
            // da un numero di pixel maggiore della soglia
            if (votes.at<uchar>(r, t) >= threshold) {
                theta = (t - 90) * CV_PI / 180;
                double sin_t = sin(theta), cos_t = cos(theta);
                // Calcola i valori di x e di y del punto
                int x = (r - dist) * cos_t;
                int y = (r - dist) * sin_t;
                
                // Calcoliamo i due estremi della retta
                Point pt1(cvRound(x + dist * (-sin_t)), cvRound(y + dist * (cos_t)));
                Point pt2(cvRound(x - dist * (-sin_t)), cvRound(y - dist * (cos_t))); 
                line(out, pt1, pt2, Scalar(0), 2, 0);
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

    int threshold = 150;

    // Blurring dell'immagine per attenuare l'eventuale rumore
    GaussianBlur(src, src, Size(5, 5), 0, 0);

    /* 1. Applichiamo l'algoritmo di Canny per ottenere l'immagine degli edge */
    Mat edgeCanny;
    Canny(src, edgeCanny, 90, 160, 3);
    imshow("Canny", edgeCanny);
    waitKey(0);

    Mat out;
    houghLines(src, out, edgeCanny, threshold);

    imshow("output", out);
    waitKey(0);

    return 0;
}