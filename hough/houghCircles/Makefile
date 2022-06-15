all: ferone my

ferone:
	g++ HoughCircle_Demo.cpp -o HoughCircle_Demo.out `pkg-config --cflags --libs opencv`
	
my:
	g++ MyHoughCircles.cpp -o MyHoughCircles.out `pkg-config --cflags --libs opencv`

clean:
	rm *.out