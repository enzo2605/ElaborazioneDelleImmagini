/*
 Region Growing
 STEP:
 - Sia f(x,y) l'immagine di input.
 - Sia S(x,y) la matrice dei seed che assegna il valore 1 alle posizioni dei seed e 0 alle altre posizioni.
 - Sia Q un predicato da applicare ad ogni pixel.
 - Formare l'immagine fQ che nel punto (x,y) contiene il valore 1 se Q(f(x,y)) � vero altrimenti contiene il valore 0.
 - Aggiungere ad ogni seed i pixel impostati ad 1 in fQ che risultano [4,8]-connessi al seed stesso.
 - Marcare ogni componente connessa con un'etichetta diversa.
 */

#include <iostream>
#include <stack>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void grow(Mat& src, Mat& dest, Mat& mask, Point seed, int threshold2);

const int threshold2 = 200;
const uchar max_region_num = 100;

//Una regione, per essere considerata tale deve avere almeno l'1% dei pixel dell'immagine totale.
const double min_region_area_factor = 0.01;

//Definisco l'8-intorno del pixel che vado a considerare.
const Point PointShift2D[8] =
{
    Point(1, 0),
    Point(1, -1),
    Point(0, -1),
    Point(-1, -1),
    Point(-1, 0),
    Point(-1, 1),
    Point(0, 1),
    Point(1, 1)
};


int main(int argc, char** argv) {
    

    //Lettura immagine    
    Mat src = imread(argv[1], IMREAD_COLOR);

    if (src.cols > 500 || src.rows > 500) 
    {
        resize(src, src, Size(0, 0), 0.5, 0.5); //Resize dell'immagine nel caso in cui sia troppo grande per ottenere un risultato in tempi ridotti.
    }
    
    //namedWindow("src", CV_WINDOW_NORMAL);


    /*
      Calcolo l'area minima che deve avere una regione per essere considerata tale
      in modo che regioni molto piccole (es: 2-3 pixel) non vengono considerate, non sono significative.
    */
    int min_region_area = int(min_region_area_factor * src.cols * src.rows);
    //namedWindow("mask", CV_WINDOW_NORMAL);

    //Padding � l'etichetta; � un contatore che viene incrementato ogni volta che viene trovata una nuova regione.
    uchar padding = 1;
    
    Mat dest = Mat::zeros(src.rows, src.cols, CV_8UC1); //Immagine destinazione.

    //Maschera che consente di individuare tutti i pixel che si trovano all'interno di una determinata regione nella fase di accrescimento.
    Mat mask = Mat::zeros(src.rows, src.cols, CV_8UC1); 

    for (int x = 0; x < src.cols; ++x) {
        for (int y = 0; y < src.rows; ++y) {
            /*
              Se dest = 0 vuol dire che il pixel in posizione (x,y) non � stato ancora considerato e quindi � il prossimo seed della regione;
              viceversa vuol dire che gi� fa parte di una regione e quindi non pu� essere considerato.
             */
            if (dest.at<uchar>(Point(x, y)) == 0) { 
                //A partire dal pixel (x,y) provo ad accrescere la regione.
                grow(src, dest, mask, Point(x, y), threshold2);

                //Sommo tutti i pixel ad 1 nella regione.
                int mask_area = (int)sum(mask).val[0];
                //Verifico se l'area della regione sia maggiore dell'area minima.
                if (mask_area > min_region_area) 
                { 
                    dest = dest + mask * padding;  //Etichetto i pixel della regione.
                    //imshow("mask", mask * 255);
                    //imshow("dest", dest * 255);
                    //waitKey();
                    if (++padding > max_region_num) //Se ottengo pi� di 100 regioni mi fermo, perch� sto over segmentando le immagini.
                    { 
                        cout << "Numero di regioni molto alto." << endl; 
                        return -1; 
                    }
                }
                else 
                {
                    dest = dest + mask * 255; //Etichetta che identifica tutte quelle regioni che hanno un'area troppo piccola per essere un'unica regione.
                }
                mask = mask - mask; //Azzero mask e passo alla prossima regione.
            }
        }
    }
    imshow("src", src);
    imshow("dest", dest);
    waitKey(0);
    return 0;
}

void grow(Mat& src, Mat& dest, Mat& mask, Point seed, int threshold2) {

    //Utilizzo lo stack per simulare la visita in profondit� di un grafo.
    stack<Point> point_stack;
    point_stack.push(seed); //Inserisco il seed nello stack.

    while (!point_stack.empty()) { //Continuo la visita finch� lo stack non � vuoto.
        
        Point center = point_stack.top(); //Verifico il pixel che si trova in cima allo stack.
        mask.at<uchar>(center) = 1;
        point_stack.pop(); //Estraggo il pixel che si trova in cima allo stack.

        //Analizzo l'8-intorno.
        for (int i = 0; i < 8; ++i) {
            Point estimating_point = center + PointShift2D[i]; //Scostamenti rispetto al punto centrale.

            //Cerco le componenti connesse all'interno dell'immagine.
            //Verifico se il punto da considerare � esterno all'immagine, in quel caso si passa al pixel successivo nell'8-intorno:
            if (estimating_point.x < 0
                || estimating_point.x > src.cols - 1
                || estimating_point.y < 0
                || estimating_point.y > src.rows - 1) {
                continue;
            } 
            // Se il punto da considerare � interno all'immagine:
            else {
                //Distanza
                int delta = int(pow(src.at<Vec3b>(center)[0] - src.at<Vec3b>(estimating_point)[0], 2)
                            + pow(src.at<Vec3b>(center)[1] - src.at<Vec3b>(estimating_point)[1], 2)
                            + pow(src.at<Vec3b>(center)[2] - src.at<Vec3b>(estimating_point)[2], 2));
                //PREDICATO.
                if (dest.at<uchar>(estimating_point) == 0 //Verifico che estimating_point non � gi� stato preso in dest, altrimenti vuol dire che appartiene ad un'altra regione;
                    && mask.at<uchar>(estimating_point) == 0 //che non � gi� stato considerato all'interno di questa regione;
                    && delta < threshold2) { //e che la distanza tra il pixel centrale e il pixel che sto accrescendo sia minore della threshold.
                    mask.at<uchar>(estimating_point) = 1; //Lo aggiungo alla mia regione.
                    point_stack.push(estimating_point); //Faccio il push del pixel all'interno dello stack.
                }
            }
        }
    }
}
