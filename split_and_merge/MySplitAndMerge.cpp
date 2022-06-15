#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

double minStdev;
int maxArea;

struct Region {
    vector<Region> adj;
    bool valid;
    Scalar label;
    Rect area;
};

/**
 * Permette di applicare il predicato su una regione. Il predicato deve essere deciso a priori.
 * 
 * @param src la matrice che rappresenta l'immagine di input
 *
 * @return true se il predicato è verificato, false se il predicato non è verificato
 **/
bool predicate(Mat src) {
    Scalar stdDev;
    meanStdDev(src, Scalar(), stdDev); // Funzione che calcola media e deviazione standard della matrice
    return (stdDev[0] < minStdev || src.rows * src.cols < maxArea);
}

/**
 * Divide l'immagine in regioni rettangolari. La suddivisione continua fintanto
 * che il predicato applicato ad una regione risulta falso.
 * 
 * @param src la matrice che rappresenta l'immagine di input
 * @param rect il rettangolo che inizialmente ha dimensioni pari a tutta la matrice.
 * 
 * @return La radice del quadtree ottenuto.
 **/
Region split(Mat src, Rect area) {
    Region R;
    R.valid = true;
    R.area = area;

    // Se il predicato è vero
    if (predicate(src)) {
        Scalar mean;
        meanStdDev(src, mean, Scalar());
        R.label = mean; // La label della regione diventa la media
    }
    else {
        int width = src.cols / 2;
        int height = src.rows / 2;

        // Suddivisione di ogni regione in 4 regioni in maniera ricorsiva
        Region upperBoundLeft = split(src(Rect(0, 0, width, height)), Rect(area.x, area.y, width, height));
        Region upperBoundRight = split(src(Rect(width, 0, width, height)), Rect(area.x + width, area.y, width, height));
        Region lowerBoundLeft = split(src(Rect(0, height, width, height)), Rect(area.x, area.y + height, width, height));
        Region lowerBoundRight = split(src(Rect(width, height, width, height)), Rect(area.x + width, area.y + height, width, height));

        // Aggiunta di ogni regione al vettore delle regioni adiacenti di R
        R.adj.push_back(upperBoundLeft);
        R.adj.push_back(upperBoundRight);
        R.adj.push_back(lowerBoundLeft);
        R.adj.push_back(lowerBoundRight);
    }

    return R;
}

/**
 * Effettua l'unione di regioni adiacenti, se il predicato applicato all'unione delle regioni è vero.
 * 
 * @param src la matrice che rappresenta l'immagine di input
 * @param r1 una regione
 * @param r2 una regione adiacente a r2
 **/
void mergeRegion(Mat src, Region &r1, Region &r2) {
    // Se le due regioni non hanno regioni adiacenti
    if (r1.adj.size() == 0 && r2.adj.size() == 0) {
        // Unione delle regioni
        Rect r12 = r1.area | r2.area;

        // Se il predicato applicato all'unione delle due regioni è vero
        if (predicate(src(r12))) {
            // Unisci le due regioni
            r1.area = r12;
            r1.label = (r1.label + r2.label) / 2;
            // Dato che la regione r2 fa parte di r1, invalida r2
            r2.valid = false;
        }
    }
}

void merge(Mat src, Region &r) {
    if (r.adj.size() > 0) {
        // Prova ad unire le regioni
        mergeRegion(src, r.adj.at(0), r.adj.at(1));
        mergeRegion(src, r.adj.at(2), r.adj.at(3));
        mergeRegion(src, r.adj.at(0), r.adj.at(2));
        mergeRegion(src, r.adj.at(1), r.adj.at(3));

        // Chiama ricorsivamente la funzione su ogni sottoregione
        // della regione r
        for (int i = 0; i < r.adj.size(); i++) {
            merge(src, r.adj.at(i));
        }
    }
}

void displayOutput(Mat& out, Region R) {
    //Se la regione master non ha sotto-regioni e la regione master è valida
	//disegna un rettangolo.
	if (R.adj.size() == 0 && R.valid) {
        rectangle(out, R.area, R.label, FILLED); //Area=Zona di interesse; Label=Colore; FILLED=Tipo di rettangolo
    }
	//Visualizza ricorsivamente le sotto-regioni della regione master.
	for (int i = 0; i < R.adj.size(); i++) {
        displayOutput(out, R.adj.at(i));
    }
}

int main(int argc, char **argv) {
    // Controllo argomenti riga di comando
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " image_name minStdev maxArea" << endl;
        return -1;
    }

    // Lettura dell'immagine
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "Could not read the image with name " << argv[1] << endl;
        return -1;
    }

    // I parametri del predicato li acquisiamo da riga di comando
    minStdev = stod(argv[2]);
    maxArea = stoi(argv[3]);

    Region r = split(src, Rect(0, 0, src.cols, src.rows));
    merge(src, r);

    Mat out = src.clone();
    displayOutput(out, r);

    imshow("src", src);
    imshow("out", out);
    waitKey(0);
    
    return 0;
}