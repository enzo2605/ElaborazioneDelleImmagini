all: ferone my

ferone:
	g++ HoughLines_Demo.cpp -o HoughLines_Demo.out `pkg-config --cflags --libs opencv`
	
my:
	g++ MyHoughLines.cpp -o MyHoughLines.out `pkg-config --cflags --libs opencv`

clean:
	rm *.out