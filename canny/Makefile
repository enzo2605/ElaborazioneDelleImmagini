all: ferone my

ferone:
	g++ Canny.cpp -o Canny.out `pkg-config --cflags --libs opencv`

my:
	g++ MyCanny.cpp -o MyCanny.out `pkg-config --cflags --libs opencv`

clean:
	rm *.out