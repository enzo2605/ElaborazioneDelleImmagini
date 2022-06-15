#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

int tsize;
double smthreshold;

class TNode {
    private:
        Rect region;
        TNode *UL, *UR, *LL, *LR;
        vector<TNode *> merged;
        vector<bool> mergedB = vector<bool>(4, false);
        double stddev, mean;
    public:
        TNode(Rect R) { 
            region = R; 
            UL = nullptr;
            UR = nullptr;
            LL = nullptr;
            LR = nullptr; 
        }

        TNode *getUL() { return UL; }
        TNode *getUR() { return UR; }
        TNode *getLL() { return LL; }
        TNode *getLR() { return LR; }

        void setUL(TNode *N) { UL = N; }
        void setUR(TNode *N) { UR = N; }
        void setLL(TNode *N) { LL = N; }
        void setLR(TNode *N) { LR = N; }

        double getStdDev() { return stddev; }
        double getMean() { return mean; }

        void setStdDev(double stddev) { this->stddev = stddev; }
        void setMean(double mean) { this->mean = mean; }

        void addRegion(TNode *R) { merged.push_back(R); }
        vector<TNode *> &getMerged() { return merged; }

        Rect &getRegion() { return region; }

        void setMergedB(int i) { mergedB[i] = true; }
        bool getMergedB(int i) { return mergedB[i]; }
};

TNode *split(Mat &img, Rect R) {
    // Si imposta la regione
    TNode *root = new TNode(R);
    
    // Calcolo media e deviazione standard
    Scalar stddev, mean;
    meanStdDev(img(R), mean, stddev);
    
    root->setMean(mean[0]);
    root->setStdDev(stddev[0]);
    
    // Si controlla se la dimensione della regione è maggiore della 
    // dimensione minima e se la deviazione standard è maggiore della
    // sogliatura impostata. Questo è il predicato
    if (R.width > tsize && root->getStdDev() > smthreshold) {
        // Fase di split
        Rect ul(R.x, R.y, R.height / 2, R.width / 2);
        root->setUL(split(img, ul));

        Rect ur(R.x, R.y + R.width / 2, R.height / 2, R.width / 2);
        root->setUR(split(img, ur));

        Rect ll(R.x + R.height / 2, R.y, R.height / 2, R.width / 2);
        root->setLL(split(img, ll));

        Rect lr(R.x + R.height / 2, R.y + R.width / 2, R.height / 2, R.width / 2);
        root->setLR(split(img, lr));
    }

    rectangle(img, R, Scalar(0));
    return root;
}

/**
 * -----------   ---------
 * | UL | UR |   | 0 | 1 |
 * -----------   ---------
 * | LL | LR |   | 3 | 2 |
 * -----------   ---------
 * 
 * Considera le sottoregioni adiacenti all'interno di una 
 * regione e prova ad unirle a coppie.
 * **/
void merge(TNode *root) {
    // Se il predicato è falso significa che la regione è stata splittata
    if (root->getRegion().width > tsize && root->getStdDev() > smthreshold) {
        // Proviamo ad unire UL e UR
        // Se il predicato per UL e per UR è vero
        if (root->getUL()->getStdDev() <= smthreshold && root->getUR()->getStdDev() <= smthreshold) {
            root->addRegion(root->getUL()); root->setMergedB(0);
            root->addRegion(root->getUR()); root->setMergedB(1);
            // Proviamo ad unire LL e LR
            if (root->getLL()->getStdDev() <= smthreshold && root->getLR()->getStdDev() <= smthreshold) {
                root->addRegion(root->getLL()); root->setMergedB(2);
                root->addRegion(root->getLR()); root->setMergedB(3);
            }
            else {
                merge(root->getLL());
                merge(root->getLR());
            }
        }

        // Proviamo ad unire UR e LR
        else if(root->getUR()->getStdDev() <= smthreshold && root->getLR()->getStdDev() <= smthreshold) {
            root->addRegion(root->getUR()); root->setMergedB(1);
            root->addRegion(root->getLR()); root->setMergedB(2);
            // Proviamo ad unire UL e UR
            if (root->getUL()->getStdDev() <= smthreshold && root->getLL()->getStdDev() <= smthreshold) {
                root->addRegion(root->getUL()); root->setMergedB(0);
                root->addRegion(root->getLL()); root->setMergedB(3);
            }
            else {
                merge(root->getUL());
                merge(root->getLL());
            }
        }

        // Proviamo ad unire LR e LL
        else if (root->getLL()->getStdDev() <= smthreshold && root->getLR()->getStdDev() <= smthreshold) {
            root->addRegion(root->getLL()); root->setMergedB(3);
            root->addRegion(root->getLR()); root->setMergedB(2);
            // Proviamo ad unire UL e UR
            if (root->getUL()->getStdDev() <= smthreshold && root->getUR()->getStdDev() <= smthreshold) {
                root->addRegion(root->getUL()); root->setMergedB(0);
                root->addRegion(root->getUR()); root->setMergedB(1);
            }
            else {
                merge(root->getUL());
                merge(root->getUR());
            }
        }

        // Proviamo ad unire UL e LL
        else if (root->getUL()->getStdDev() <= smthreshold && root->getLL()->getStdDev() <= smthreshold) {
            root->addRegion(root->getUL()); root->setMergedB(0);
            root->addRegion(root->getLL()); root->setMergedB(3);
            // Proviamo ad unire UL e UR
            if (root->getUR()->getStdDev() <= smthreshold && root->getLR()->getStdDev() <= smthreshold) {
                root->addRegion(root->getUR()); root->setMergedB(1);
                root->addRegion(root->getLR()); root->setMergedB(2);
            }
            else {
                merge(root->getUR());
                merge(root->getLR());
            }
        }

        // Scendiamo nei 4 figli e proviamo a fare i merge
        else {
            merge(root->getUL());
            merge(root->getUR());
            merge(root->getLL());
            merge(root->getLR());
        }
    }
    // La regione non è stata splittata
    else {
        root->addRegion(root);
        root->setMergedB(0);
        root->setMergedB(1);
        root->setMergedB(2);
        root->setMergedB(3);
    }
}

void segment(TNode *root, Mat &img) {
    vector<TNode *> tmp = root->getMerged();

    if (!tmp.size()) {
        segment(root->getUL(), img);
        segment(root->getUR(), img);
        segment(root->getLR(), img);
        segment(root->getLL(), img);
    }
    // Quando identifichiamo una regione valida
    else {
        double val = 0;
        // Tutti i pixel della regione assumono valore medio
        for (auto x : tmp) {
            val += (int) x->getMean();
        }

        val /= tmp.size();

        for (auto x : tmp) {
            img(x->getRegion()) = (int)val;
        }
        // Se la sottoregione non è stata ancora segmentata
        // Andiamo a segmentare ulteriormente
        if (tmp.size() > 1) {
            if (!root->getMergedB(0)) {
                segment(root->getUL(), img);
            }
            if (!root->getMergedB(1)) {
                segment(root->getUR(), img);
            }
            if (!root->getMergedB(2)) {
                segment(root->getLR(), img);
            }
            if (!root->getMergedB(3)) {
                segment(root->getLL(), img);
            }
        }
    }
}

int main(int argc, char **argv) {
    // Controllo numero argomenti da riga di comando
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " image_name area stddev" << endl;
        return -1;
    }

    // Lettura dell'immagine
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "Could not read the image with name " << argv[1] << endl;
        return -1;
    }

    // Inizializzazione dei parametri acquisiti da riga di comando
    tsize = stoi(argv[2]);
    smthreshold = stod(argv[3]);

    // Blurring per attenuare il rumore
    GaussianBlur(src, src, Size(5, 5), 0, 0);
    // Troviamo la dimensione ottimale dell'immagine
    int exponent = log(min(src.cols, src.rows)) / log(2);
    int s = pow(2.0, (double)exponent);
    Rect square = Rect(0, 0, s, s);
    src = src(square).clone();
    Mat srcSeg = src.clone();

    // Fase di split
    TNode *root = split(src, Rect(0, 0, src.rows, src.cols));
    // Conclusa la fase di split effettuiamo la fase di merge
    merge(root);
    // Segmentazione
    segment(root, srcSeg);  

    imshow("src", src);
    imshow("segmented", srcSeg);
    waitKey(0);
    return 0;
}