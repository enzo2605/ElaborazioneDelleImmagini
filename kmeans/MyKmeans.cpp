/*
  K-MEANS
  STEP:
  - Inizializzo i centri dei cluster.
  -	Assegno ogni pixel al centro più vicino:
        Per ogni pixel Pj calcolare la distanza dai k centri Ci ed assegnare Pj al cluster con il centro Ci più vicino.
  -	Aggiornare i centri:
        Calcolare la media dei pixel in ogni cluster.
  - Ripetere i punti 2 e 3 finchè il centro (media) di ogni cluster non viene più modificato (ovvero i gruppi non vengono modificati).
*/

#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>  

using namespace std;
using namespace cv;

double euclideanDistance(Scalar p1, Scalar p2) {
    double diffBlue = p1.val[0] - p2[0];
    double diffGreen = p1.val[1] - p2[1];
    double diffRed = p1.val[2] - p2[2];

    double distance = sqrt(pow(diffBlue, 2) + pow(diffGreen, 2) + pow(diffRed, 2));
    
    return distance;
}

void myKmeans(Mat &src, Mat &dst, int nClusters, double threshold) {
    // Vettore che contiene i colori dei centri
    vector<Scalar> centersColors;
    // Vettore che contiene i punti dei cluster
    vector<vector<Point>> clusters;

    RNG random(getTickCount());

    /* 1. Inizializzo i centri del cluster in maniera random */
    for (int k = 0; k < nClusters; k++) {
        // Calcolo delle coordinate del centro
        Point center;
        center.x = random.uniform(0, src.cols);
        center.y = random.uniform(0, src.rows);

        // Memorizzo i colori del centro appena calcolato in uno scalare
        Scalar center_color(src.at<Vec3b>(center)[0], src.at<Vec3b>(center)[1], src.at<Vec3b>(center)[2]);
        // Aggiungo il colore del centro al vettore che contiene i colori dei centri
        centersColors.push_back(center_color);

        // Vettore che conterrà i pixel assegnati ad ogni cluster
        vector<Point> pointInClusterK;
        // Aggiungo un vettore di punti al vettore che conterrà i punti del cluster
        clusters.push_back(pointInClusterK);
    }

    //* 2. Assegno i pixel ai cluster, ricalcolo i centri usando le medie, fino a che la differenza > 0.1 */
    double oldCenterSum = 0.0;
    double diffOldNewAvg = INFINITY; // Differenza tra la vecchia e la nuova media

    // Itera finché la differenza tra le vecchie medie e le nuove supera una certa soglia
    while (diffOldNewAvg > threshold) {
        // Pulizia dei vettori di punti presenti nei cluster
        for (int k = 0; k < nClusters; k++) {
            clusters[k].clear();
        }

        // Assegno i pixel ai cluster
        for (int x = 0; x < src.rows; x++) {
            for (int y = 0; y < src.cols; y++) {
                // Calcolo le distanze da ogni centro dei cluster e posiziono il pixel nel cluster più vicino.
                double minDistance = INFINITY;
                int clusterIndex = 0; // Serve per memorizzare quale cluster è più vicino al pixel

                // Estrazione del pixel in posizione x, y
                Scalar point = src.at<Vec3b>(x, y);

                // Ciclo su ogni cluster per capire il pixel a quale
                // cluster deve essere assegnato
                for (int k = 0; k < nClusters; k++) {
                    // Estrazione del centro del cluster (vettore di 3 componenti RGB)
                    Scalar center = centersColors[k];

                    // Calcolo della distanza Euclidea tra il centro del cluster e il pixel x,y
                    double distance = euclideanDistance(point, center);

                    // Se la distanza è la minore, la si aggiorna
                    if (distance < minDistance) {
                        minDistance = distance;
                        clusterIndex = k;
                    }
                }

                // Aggiunta del pixel nel vettore k dei cluster
                clusters[clusterIndex].push_back(Point(y, x));
            }
        }

        // Aggiornamento dei centri, ovvero ricalcolo delle medie
        double newCenterSum = 0;

        for (int k = 0; k < nClusters; k++) {
            // Estrazione di tutti i pixel presenti nel cluster k
            vector<Point> clusterKPoints = clusters[k];
            double blue = 0.0, green = 0.0, red = 0.0;

            // Somma dei colori di tutti i punti del cluster k
            for (int i = 0; i < clusterKPoints.size(); i++) {
                Point pixel = clusterKPoints[i];
                blue += src.at<Vec3b>(pixel)[0];
                green += src.at<Vec3b>(pixel)[1];
                red += src.at<Vec3b>(pixel)[2];
            }

            // Calcolo della medie dei colori del nuovo centro
            blue /= clusterKPoints.size();
            green /= clusterKPoints.size();
            red /= clusterKPoints.size();

            // Estrazione del vecchio centro
            Scalar center = centersColors[k];
            // Calcolo del nuovo centro
            Scalar newCenter(blue, green, red);

            // Calcolo distanza tra vecchio e nuovo centro
            newCenterSum += euclideanDistance(newCenter, center);
            // Aggiornamento nuovo centro
            centersColors[k] = newCenter;
        }

        // Calcolo della media dividendo la media per il numero di cluster
        newCenterSum /= nClusters;
        // Calcolo della differenza tra la vecchia somma e la nuova somma
        diffOldNewAvg = abs(oldCenterSum - newCenterSum);
        // Aggiornamento della somma
        oldCenterSum = newCenterSum;
    }

    // Nell'immagine di output, bisogna assegnare ad ogni pixel nel cluster
    // k il livello di intensità del centro del cluster
    for (int k = 0; k < nClusters; k++) {
        // Estrazione di tutti i punti del cluster
        vector<Point> clusterPoints = clusters[k];
        // Estrazione del centro del cluster
        Scalar center = centersColors[k];
        for (int i = 0; i < clusterPoints.size(); i++) {
            // Ad ogni pixel viene assegnata l'intensità del centro del cluster
            Point pixel = clusterPoints[i];
            dst.at<Vec3b>(pixel)[0] = center[0];
            dst.at<Vec3b>(pixel)[1] = center[1];
            dst.at<Vec3b>(pixel)[2] = center[2];
        }
    }
}

int main(int argc, char **argv) {
    // Controllo argomenti riga di comando
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " image_name number_of_clusters" << endl;
        return -1;
    }

    // Lettura dell'immagine
    Mat src = imread(argv[1], IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not read the image with name " << argv[1] << endl;
        return -1;
    }

    // Il numero di cluster è passato da riga di comando come secondo argomento
    int clusters_number = stoi(argv[2]);

    Mat dst(src.size(), src.type());
    myKmeans(src, dst, clusters_number, 0.1);
    
    imshow("Source image", src);
	imshow("K-Means", dst);
	waitKey(0);
    return 0;
}