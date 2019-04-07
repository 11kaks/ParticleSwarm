CC = g++
CFLAGS = -c
SOURCES =  ParticleSwarm.cpp OP.cpp Rastriging.cpp Particle.cpp Swarm.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = ParticleSwarm

all: $(OBJECTS) $(EXECUTABLE)

$(EXECUTABLE) : $(OBJECTS)
		$(CC) $(OBJECTS) -o $@  

.cpp.o: *.h
	$(CC) $(CFLAGS) $< -o $@

clean :
	-rm -f $(OBJECTS) $(EXECUTABLE)

.PHONY: all clean