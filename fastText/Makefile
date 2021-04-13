CPPFLAGS += -std=c++11 -Iinclude

libfasttext-wrapper.a: libfasttext-wrapper.o
	ar rcs lib/$@ obj/*.o

libfasttext-wrapper.o: fasttext-wrapper.cpp
	g++ -c $(CPPFLAGS) $^ -o obj/$@

.PHONY: clean

clean:
	rm -rf obj/libfasttext-wrapper.o lib/libfasttext-wrapper.a
