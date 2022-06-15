all: my ferone

my:
	g++ MyKmeans.cpp -o MyKmeans.out `pkg-config --cflags --libs opencv`

ferone:
	g++ kmeansF.cpp -o kmeansF.out `pkg-config --cflags --libs opencv`

clean:
	rm *.out