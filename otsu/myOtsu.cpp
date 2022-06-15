#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int Otsu(vector<double> his) {
    double mediaCumGlob = 0.0f;
    // Calcolo della media cumulativa globale mG
    for (int i = 0; i < 256; i++) {
        mediaCumGlob += i * his[i];
    }
    
    double prob = 0.0f;
    double currMediaCum = 0.0f;
    double currVar = 0.0f;
    double maxVar = 0.0f;
    int thresh;
    for (int i = 0; i < 256; i++) {
        // Calcolo somma cumulativa P1(k)
        prob += his[i];
        // Calcolo media cumulativa m(k)
        currMediaCum += i * his[i];
        // Calcolo della varianza interclasse sigma(k)
        currVar = pow(mediaCumGlob * prob - currMediaCum, 2) / (prob * (1 - prob));
        // Massimizza la varianza interclasse
        if (currVar > maxVar) {
            maxVar = currVar;
            thresh = i;
        }
    }

    return thresh;
}

vector<double> NormalizedHistogram(Mat img) {
    // Inizializziamo il vettore di double
    vector<double> his(256, 0.0f);

    // Calcoliamo il numero di occorrenze in termini
    // di valore di intensità per ogni pixel
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            his[img.at<uchar>(y, x)]++;
        }
    }

    // Normalizzazione dell'istogramma
    for (int i = 0; i < 256; i++) {
        his[i] /= img.rows *img.cols;
    }

    return his;
}

vector<int> OtsuMultipleThresh(vector<double> his) {
    // Calcolo della media cumulativa globale mG
    double mediaCumGlob = 0.0f;
    for (int i = 0; i < 256; i++) {
        mediaCumGlob += i * his[i];
    }

    // Abbiamo 3 classi, quindi 3 probabilità e 3 medie cumulative
    vector<double> prob(3, 0.0f);
    vector<double> currMediaCum(3, 0.0f);
    double currVar = 0.0f;
    double maxVar = 0.0f;
    // Abbiamo due soglie poiché abbiamo 3 classi
    vector<int> thresh(2, 0);
    for (int i = 0; i < 256 - 2; i++) {
        // Calcolo somma cumulativa P1(k)
        prob[0] += his[i];
        // Calcolo media cumulativa m1(k)
        currMediaCum[0] += i * his[i];

        for (int j = i + 1; j < 256 - 1; j++) {
            // Calcolo somma cumulativa P2(k)
            prob[1] += his[j];
            // Calcolo media cumulativa m2(k)
            currMediaCum[1] += j * his[j];

            for (int k = j + 1; k < 256; k++) {
                // Calcolo somma cumulativa P3(k)
                prob[2] += his[k];
                // Calcolo media cumulativa m3(k)
                currMediaCum[2] += k * his[k];

                // Calcolo della varianza interclasse sigma(k1, k2)
                currVar = 0.0f;
                for (int w = 0; w < 3; w++) {
                    currVar += prob[w] * pow(currMediaCum[w] / prob[w] - mediaCumGlob, 2);
                }

                // Calcolo della varianza massima
                if (currVar > maxVar) {
                    maxVar = currVar;
                    thresh[0] = i;
                    thresh[1] = j;
                }
            }
            prob[2] = currMediaCum[2] = 0.0f;
        }
        prob[1] = currMediaCum[1] = 0.0f;
    }

    return thresh;
}

void MultipleThreshold(Mat img, Mat &out, vector<int> thresh) {
    out = Mat::zeros(img.size(), img.type());
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            if (img.at<uchar>(y, x) >= thresh[1]) {
                out.at<uchar>(y, x) = 255;
            }
            else if (img.at<uchar>(y, x) >= thresh[0]) {
                out.at<uchar>(y, x) = 127;
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
    
    /** 1. Applichiamo un filtro Gaussiano per sfocare l'immagine **/
    GaussianBlur(src, src, Size(5, 5), 0, 0);

    /** 2. Calcoliamo l'istogramma normalizzato **/
    vector<double> hist = NormalizedHistogram(src);

    /** 3. Algoritmo di Otsu con una singola soglia **/
    int otsuThresh = Otsu(hist);

    /** Applicazione del thresholding **/
    Mat out;
    threshold(src, out, otsuThresh, 255, THRESH_BINARY);
    imshow("Otsu", out);
    waitKey(0);

    /** Otsu con soglie multiple **/
    vector<int> thresh = OtsuMultipleThresh(hist);
    cout << "min: " << thresh[0] << " max: " << thresh[1] << endl;
    MultipleThreshold(src, out, thresh);
    imshow("Otsu2", out);
    waitKey(0);

    return 0;
}