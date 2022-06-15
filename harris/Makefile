all: ferone my

ferone:
	g++ Harris.cpp -o Harris.out `pkg-config --cflags --libs opencv`

my:
	g++ MyHarris.cpp -o MyHarris.out `pkg-config --cflags --libs opencv`

clean:
	rm *.out