all: ferone my

ferone:
	g++ RegionGrowing.cpp -o RegionGrowing.out `pkg-config --cflags --libs opencv`
	
my:
	g++ MyRegionGrowing.cpp -o MyRegionGrowing.out `pkg-config --cflags --libs opencv`

clean:
	rm -f *.out